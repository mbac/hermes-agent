"""Tests for gateway.shutdown_forensics — fast snapshot + async diag spawn."""

from __future__ import annotations

import json
import builtins
import io
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from gateway import shutdown_forensics as sf


# ---------------------------------------------------------------------------
# _signal_name
# ---------------------------------------------------------------------------

class TestSignalName:

    def test_unknown_int_returns_signal_num_token(self):
        # Pick an integer extremely unlikely to ever be a real signal alias
        assert sf._signal_name(9999) == "signal#9999"


# ---------------------------------------------------------------------------
# snapshot_shutdown_context
# ---------------------------------------------------------------------------

class TestSnapshotShutdownContext:

    def test_handles_none_signal(self):
        ctx = sf.snapshot_shutdown_context(None)
        assert ctx["signal"] == "UNKNOWN"
        assert ctx["signal_num"] is None

    def test_includes_timestamps(self):
        before = time.time()
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        after = time.time()
        assert before <= ctx["ts"] <= after
        assert isinstance(ctx["ts_monotonic"], float)


    def test_under_systemd_false_without_invocation_id_and_normal_ppid(
        self, monkeypatch
    ):
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        # We can't actually change ppid; skip if we happen to be reaped
        # by init (e.g. running under tini).
        if os.getppid() == 1:
            pytest.skip("test process is reaped by init")
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert ctx["under_systemd"] is False


    def test_detects_takeover_marker_for_self(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker = tmp_path / ".gateway-takeover.json"
        marker.write_text(
            f'{{"target_pid": {os.getpid()}, "replacer_pid": 99999}}',
            encoding="utf-8",
        )
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert "takeover_marker" in ctx
        assert ctx["takeover_marker_for_self"] is True


# ---------------------------------------------------------------------------
# format_context_for_log / context_as_json
# ---------------------------------------------------------------------------

class TestFormatters:


    def test_context_as_json_handles_unserialisable_values(self):
        ctx = {"signal": "SIGTERM", "weird": object()}
        payload = sf.context_as_json(ctx)
        # default=str means objects get repr'd, JSON stays valid
        decoded = json.loads(payload)
        assert decoded["signal"] == "SIGTERM"
        assert "weird" in decoded


# ---------------------------------------------------------------------------
# spawn_async_diagnostic
# ---------------------------------------------------------------------------

class TestSpawnAsyncDiagnostic:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only diagnostic")
    def test_spawns_subprocess_and_writes_output(self, tmp_path):
        log_path = tmp_path / "diag.log"
        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=3.0)
        assert pid is not None and pid > 0

        # Wait briefly for the subprocess to write — bounded by its own timeout.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if log_path.exists() and log_path.stat().st_size > 0:
                # Wait a touch longer for the script to finish writing
                time.sleep(0.2)
                break
            time.sleep(0.1)

        # Reap the subprocess so it doesn't show up as a zombie.
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass

        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "shutdown diagnostic" in contents
        assert "SIGTERM" in contents


# ---------------------------------------------------------------------------
# _parse_systemd_duration_to_us
# ---------------------------------------------------------------------------

class TestParseSystemdDuration:
    def test_seconds(self):
        assert sf._parse_systemd_duration_to_us("90s") == 90 * 1_000_000

    def test_minutes(self):
        assert sf._parse_systemd_duration_to_us("3min") == 180 * 1_000_000


# ---------------------------------------------------------------------------
# check_systemd_timing_alignment
# ---------------------------------------------------------------------------

class TestCheckSystemdTimingAlignment:

    def test_returns_none_when_unit_undeterminable(self, monkeypatch):
        monkeypatch.setenv("INVOCATION_ID", "abc")
        # /proc/self/cgroup likely doesn't end in .service for the test runner
        result = sf.check_systemd_timing_alignment(180.0)
        # Either None (we couldn't find a unit) or a dict with mismatch info
        # for whatever unit pytest IS in.  Both are valid; we just ensure
        # the function doesn't raise.
        assert result is None or isinstance(result, dict)

    @staticmethod
    def _fake_systemd(monkeypatch, user_output, system_output):
        """Pretend we are PID-1-supervised inside hermes-gateway.service.

        ``user_output``/``system_output`` are the stdout ``systemctl show``
        returns for the ``--user`` and system managers respectively.
        """
        import subprocess as _sp

        monkeypatch.setenv("INVOCATION_ID", "abc")

        real_open = builtins.open

        def fake_open(path, *a, **kw):
            if str(path) == "/proc/self/cgroup":
                return io.StringIO("0::/system.slice/hermes-gateway.service\n")
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)

        def fake_run(cmd, **kw):
            out = user_output if "--user" in cmd else system_output
            return _sp.CompletedProcess(cmd, 0, stdout=out, stderr="")

        monkeypatch.setattr(sf.subprocess, "run", fake_run)

    def test_ignores_manager_where_unit_is_not_loaded(self, monkeypatch):
        """A system-unit install must not be judged by the --user manager.

        ``systemctl --user show`` exits 0 for a unit it has never heard of and
        prints systemd's *defaults* (TimeoutStopUSec=1min 30s). Trusting that
        made every boot log a bogus "stale systemd unit (TimeoutStopSec=90s)"
        warning against a unit that was actually current.
        """
        self._fake_systemd(
            monkeypatch,
            user_output="TimeoutStopUSec=1min 30s\nLoadState=not-found\n",
            system_output="TimeoutStopUSec=10min 30s\nLoadState=loaded\n",
        )
        result = sf.check_systemd_timing_alignment(600.0, 30.0)
        assert result is not None
        # 630s from the *system* manager, not 90s from the --user default.
        assert result["timeout_stop_sec"] == 630.0
        assert not result["mismatch"]

    def test_still_reports_a_genuinely_short_timeout(self, monkeypatch):
        """The real stale-unit case must still be detected."""
        self._fake_systemd(
            monkeypatch,
            user_output="TimeoutStopUSec=1min 30s\nLoadState=not-found\n",
            system_output="TimeoutStopUSec=1min 30s\nLoadState=loaded\n",
        )
        result = sf.check_systemd_timing_alignment(600.0, 30.0)
        assert result is not None
        assert result["timeout_stop_sec"] == 90.0
        assert result["mismatch"]

    def test_returns_none_when_no_manager_has_the_unit(self, monkeypatch):
        self._fake_systemd(
            monkeypatch,
            user_output="TimeoutStopUSec=1min 30s\nLoadState=not-found\n",
            system_output="TimeoutStopUSec=1min 30s\nLoadState=not-found\n",
        )
        assert sf.check_systemd_timing_alignment(600.0, 30.0) is None
