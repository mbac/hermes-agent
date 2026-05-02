"""Tests for web backend runtime fallback.

Covers four tiers:
  Tier 1 — Callable-level unit tests (fn_map, bare lambdas)
  Tier 2 — HTTP-level failure simulation (httpx mocks)
  Tier 3 — Firecrawl gateway / SDK-level fallback
  Tier 4 — Async extract / crawl fallback
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tools.web_tools import (
    _get_backend,
    _get_backend_candidates,
    _try_backend_with_fallback,
    _try_backend_with_fallback_async,
    _is_backend_available,
    web_search_tool,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────


def _make_mock_response(status_code: int, json_data=None, text=""):
    """Build a mock ``httpx.Response`` whose ``raise_for_status()``
    raises ``HTTPStatusError`` when *status_code* >= 400."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code} error", request=MagicMock(), response=resp,
        )
    return resp


def _fake_search_result(url: str):
    """Minimal search-result dict matching the standard search schema."""
    return {
        "success": True,
        "data": {"web": [{"url": url, "title": "Test", "description": "desc", "position": 1}]},
    }


def _fake_exa_search_result(url: str):
    return _fake_search_result(url)


def _fake_parallel_search_result(url: str):
    return _fake_search_result(url)


# ─── Tier 1: Backend Candidates ──────────────────────────────────────────────


class TestBackendCandidates:
    """``_get_backend_candidates()`` returns a prioritised list."""

    def test_explicit_config_first(self, monkeypatch):
        """When ``web.backend=parallel``, parallel is first candidate."""
        monkeypatch.setenv("EXA_API_KEY", "sk-fake")
        monkeypatch.setenv("PARALLEL_API_KEY", "sk-fake")
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            candidates = _get_backend_candidates()
            assert candidates[0] == "parallel"
            assert "exa" in candidates[1:]

    def test_no_config_priority_order(self, monkeypatch):
        """When no explicit config, firecrawl is first (if available)."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-fake")
        monkeypatch.setenv("PARALLEL_API_KEY", "p-fake")
        with patch("tools.web_tools._load_web_config", return_value={}):
            candidates = _get_backend_candidates()
            assert candidates[0] == "firecrawl"
            assert candidates[1] == "parallel"

    def test_unconfigured_backend_not_in_candidates(self, monkeypatch):
        """Backend without API key is not in candidates."""
        for var in ("EXA_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY",
                     "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"):
            monkeypatch.delenv(var, raising=False)
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            candidates = _get_backend_candidates()
            assert len(candidates) == 1  # only firecrawl default
            assert candidates[0] == "firecrawl"

    def test_legacy_get_backend(self, monkeypatch):
        """``_get_backend()`` still returns first candidate."""
        monkeypatch.setenv("PARALLEL_API_KEY", "p-fake")
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert _get_backend() == "parallel"


# ─── Tier 2: Callable-Level Fallback ──────────────────────────────────────────


class TestCallableFallback:
    """``_try_backend_with_fallback()`` dispatches correctly."""

    def test_first_backend_succeeds(self):
        fn_map = {
            "firecrawl": lambda: _fake_search_result("fc"),
            "parallel": lambda: _fake_search_result("p"),
        }
        with patch("tools.web_tools._get_backend_candidates",
                   return_value=["firecrawl", "parallel"]):
            result = _try_backend_with_fallback("search", fn_map, "Error")
            assert result["data"]["web"][0]["url"] == "fc"

    def test_fallback_on_runtime_error(self):
        fn_map = {
            "firecrawl": lambda: (_ for _ in ()).throw(RuntimeError("timeout")),
            "parallel": lambda: _fake_search_result("fallback"),
        }
        with patch("tools.web_tools._get_backend_candidates",
                   return_value=["firecrawl", "parallel"]):
            result = _try_backend_with_fallback("search", fn_map, "Error")
            assert result["data"]["web"][0]["url"] == "fallback"

    def test_skips_missing_api_key(self):
        fn_map = {
            "firecrawl": lambda: (_ for _ in ()).throw(ValueError("API key not set")),
            "parallel": lambda: _fake_search_result("ok"),
        }
        with patch("tools.web_tools._get_backend_candidates",
                   return_value=["firecrawl", "parallel"]):
            result = _try_backend_with_fallback("search", fn_map, "Error")
            assert result["data"]["web"][0]["url"] == "ok"

    def test_all_backends_fail(self):
        fn_map = {
            "firecrawl": lambda: (_ for _ in ()).throw(RuntimeError("e1")),
            "parallel": lambda: (_ for _ in ()).throw(RuntimeError("e2")),
        }
        with patch("tools.web_tools._get_backend_candidates",
                   return_value=["firecrawl", "parallel"]):
            result = _try_backend_with_fallback("search", fn_map, "Error prefix")
            assert isinstance(result, str)
            data = json.loads(result)
            assert "Error prefix" in data.get("error", "")

    def test_skip_backend_without_handler(self):
        fn_map = {
            "parallel": lambda: _fake_search_result("ok"),
        }
        with patch("tools.web_tools._get_backend_candidates",
                   return_value=["exa", "parallel"]):
            result = _try_backend_with_fallback("search", fn_map, "Error")
            assert result["data"]["web"][0]["url"] == "ok"


# ─── Tier 3: HTTP-Level Fallback ──────────────────────────────────────────────


class TestHTTPLevelFallback:
    """End-to-end fallback via the actual dispatch path with HTTP mocks."""

    @patch("tools.web_tools._load_web_config", return_value={})
    def test_tavily_429_falls_back_to_exa(self, mock_config, monkeypatch):
        """Tavily rate-limited (429) → Exa succeeds."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
        monkeypatch.setenv("EXA_API_KEY", "exa-fake")
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        with patch("tools.web_tools.httpx.post") as mock_post, \
             patch("tools.web_tools._get_exa_client") as mock_exa_client, \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):

            # Tavily returns 429
            mock_post.return_value = _make_mock_response(429, text="rate limited")

            # Exa succeeds — need a mock with `.results` attribute (SDK shape)
            mock_exa = MagicMock()
            mock_exa.search.return_value = MagicMock(results=[
                MagicMock(url="exa-page", title="Test", highlights=["desc"])
            ])
            mock_exa_client.return_value = mock_exa

            result = web_search_tool("test query")
            data = json.loads(result)
            assert data["success"] is True
            assert data["data"]["web"][0]["url"] == "exa-page"

    @patch("tools.web_tools._load_web_config", return_value={})
    def test_tavily_timeout_falls_back_to_parallel(self, mock_config, monkeypatch):
        """Tavily connection timeout → Parallel succeeds."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
        monkeypatch.setenv("PARALLEL_API_KEY", "p-fake")
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        with patch("tools.web_tools.httpx.post") as mock_post, \
             patch("tools.web_tools._get_parallel_client") as mock_p_client, \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):

            mock_post.side_effect = httpx.ConnectTimeout("connection timed out")

            # Parallel succeeds — need `.results` on the returned search object
            mock_p = MagicMock()
            mock_p.beta.search.return_value = MagicMock(results=[
                MagicMock(url="parallel-page", title="Test", excerpts=["desc"])
            ])
            mock_p_client.return_value = mock_p

            result = web_search_tool("test query")
            data = json.loads(result)
            assert data["success"] is True
            assert data["data"]["web"][0]["url"] == "parallel-page"

    @patch("tools.web_tools._load_web_config", return_value={})
    def test_all_backends_fail_http_errors(self, mock_config, monkeypatch):
        """Every backend throws → tool_error returned."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
        monkeypatch.setenv("EXA_API_KEY", "exa-fake")
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        with patch("tools.web_tools.httpx.post") as mock_post, \
             patch("tools.web_tools._get_exa_client") as mock_exa_client, \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):

            # Tavily returns 503
            mock_post.return_value = _make_mock_response(503)
            # Exa SDK throws
            mock_exa_client.side_effect = httpx.ConnectError("no route")

            result = web_search_tool("test query")
            data = json.loads(result)
            assert "Error searching web" in data.get("error", "")

    @patch("tools.web_tools._load_web_config", return_value={})
    def test_firecrawl_503_falls_back_to_parallel(self, mock_config, monkeypatch):
        """Firecrawl returns 503 → Parallel succeeds."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-fake")
        monkeypatch.setenv("PARALLEL_API_KEY", "p-fake")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        with patch("tools.web_tools._get_firecrawl_client") as mock_fc, \
             patch("tools.web_tools._get_parallel_client") as mock_p_client:

            # Firecrawl search throws
            mock_fc.return_value.search.side_effect = httpx.HTTPStatusError(
                "503", request=MagicMock(), response=MagicMock(status_code=503),
            )

            # Parallel succeeds
            mock_p = MagicMock()
            mock_p.beta.search.return_value = MagicMock(results=[
                MagicMock(url="parallel-fallback", title="Test", excerpts=["desc"])
            ])
            mock_p_client.return_value = mock_p

            result = web_search_tool("test query")
            data = json.loads(result)
            assert data["success"] is True
            assert data["data"]["web"][0]["url"] == "parallel-fallback"


# ─── Tier 4: Async Extract / Crawl Fallback ───────────────────────────────────


class TestAsyncFallback:
    """Async fallback via ``_try_backend_with_fallback_async``."""

    @pytest.mark.asyncio
    async def test_async_helper_handles_mixed_sync_async(self, monkeypatch):
        """Helper awaits coroutine results from lambdas."""
        monkeypatch.setenv("PARALLEL_API_KEY", "p-fake")
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        async def _async_ok():
            return {"ok": True}

        fn_map = {
            "parallel": lambda: _async_ok(),
            "firecrawl": lambda: {"sync": True},
        }
        with patch("tools.web_tools._get_backend_candidates",
                   return_value=["parallel", "firecrawl"]):
            result = await _try_backend_with_fallback_async("test", fn_map, "Error")
            assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_extract_fallback_tavily_to_exa(self, monkeypatch):
        """Tavily extract fails → Exa succeeds."""
        from tools.web_tools import web_extract_tool

        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
        monkeypatch.setenv("EXA_API_KEY", "exa-fake")
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        with patch("tools.web_tools.httpx.post") as mock_post, \
             patch("tools.web_tools._get_exa_client") as mock_exa_client, \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False), \
             patch("tools.web_tools.check_auxiliary_model", return_value=False):

            # Tavily returns 500
            mock_post.return_value = _make_mock_response(500)

            # Exa succeeds
            mock_exa = MagicMock()
            mock_exa.get_contents.return_value = MagicMock(results=[
                MagicMock(url="https://ex.com", title="Exa Page", text="content")
            ])
            mock_exa_client.return_value = mock_exa

            result = await web_extract_tool(["https://example.com"])
            data = json.loads(result)
            assert data["results"][0]["url"] == "https://ex.com"
            assert data["results"][0]["title"] == "Exa Page"
            assert data["results"][0]["content"] == "content"
            assert mock_exa.get_contents.called

    @pytest.mark.asyncio
    async def test_extract_fallback_firecrawl_scrape_error_to_exa(self, monkeypatch):
        """Firecrawl scrape errors are backend failures, so Exa gets a chance."""
        from tools.web_tools import web_extract_tool

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-fake")
        monkeypatch.setenv("EXA_API_KEY", "exa-fake")
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        with patch("tools.web_tools._get_firecrawl_client") as mock_fc_client, \
             patch("tools.web_tools._get_exa_client") as mock_exa_client, \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False), \
             patch("tools.web_tools.check_auxiliary_model", return_value=False):

            mock_fc_client.return_value.scrape.side_effect = httpx.HTTPStatusError(
                "503", request=MagicMock(), response=MagicMock(status_code=503),
            )

            mock_exa = MagicMock()
            mock_exa.get_contents.return_value = MagicMock(results=[
                MagicMock(url="https://ex.com", title="Exa Page", text="exa content")
            ])
            mock_exa_client.return_value = mock_exa

            result = await web_extract_tool(["https://example.com"], use_llm_processing=False)
            data = json.loads(result)
            assert data["results"][0]["url"] == "https://ex.com"
            assert data["results"][0]["content"] == "exa content"
            assert mock_exa.get_contents.called
