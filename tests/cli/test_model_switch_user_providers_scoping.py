"""Regression test for ``/model <name> --provider <user-provider>`` scoping.

In ``HermesCLI._handle_model_switch``, the config-loaded ``providers:`` and
``custom_providers:`` sections were previously only populated inside the
"no args" branch that opens the interactive picker. When the user ran
``/model foo --provider bar`` (explicit --provider, model given), that
branch was skipped and ``user_provs`` / ``custom_provs`` stayed ``None``,
causing ``switch_model`` to receive no user-config context and returning
"Unknown provider 'bar'..." for any user-defined provider.

This test locks in the scoping fix so future refactors cannot regress it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_bare_cli():
    """Construct a HermesCLI instance without running ``__init__``.

    Only the attributes ``_handle_model_switch`` touches before the
    ``switch_model`` call are populated.
    """
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "some-current-model"
    cli.provider = "openrouter"
    cli.base_url = ""
    cli.api_key = ""
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    return cli


def _user_providers_fixture():
    """A config.yaml ``providers:`` section — e.g. a local LiteLLM proxy."""
    return {
        "litellm": {
            "name": "LiteLLM Proxy",
            "base_url": "http://127.0.0.1:4000/v1",
            "key_env": "LITELLM_MASTER_KEY",
            "transport": "openai_chat",
            "models": {
                "kimi": {"model": "kimi"},
                "grok": {"model": "grok"},
            },
        }
    }


def _fail_result():
    """A ``switch_model`` result that makes the handler return early.

    Returning ``success=False`` short-circuits the post-switch state
    mutations, so the test only needs to assert on the kwargs captured
    by the ``switch_model`` mock.
    """
    return MagicMock(success=False, error_message="forced-early-return",
                     is_global=False)


def test_model_switch_with_explicit_provider_loads_user_providers_from_config():
    """``/model kimi --provider litellm`` must see ``providers.litellm`` from config.

    Regression: previously ``user_provs`` / ``custom_provs`` were loaded
    from config only inside the no-args picker branch, so explicit-model
    switches received ``user_providers=None`` and could not resolve
    user-defined providers.
    """
    cli = _make_bare_cli()
    user_provs = _user_providers_fixture()

    with patch("hermes_cli.config.load_config",
               return_value={"providers": user_provs}), \
         patch("hermes_cli.config.get_compatible_custom_providers",
               return_value=[]), \
         patch("hermes_cli.model_switch.switch_model",
               return_value=_fail_result()) as mock_switch:
        cli._handle_model_switch("/model kimi --provider litellm")

    assert mock_switch.called, "switch_model must be invoked"
    kwargs = mock_switch.call_args.kwargs
    assert kwargs.get("explicit_provider") == "litellm"
    assert kwargs.get("user_providers") == user_provs, (
        "user_providers must be populated from config.yaml before "
        "switch_model() is called — this is the scoping fix."
    )
    assert kwargs.get("custom_providers") == [], (
        "custom_providers should be populated (empty list here), not None"
    )


def test_model_switch_bare_model_loads_user_providers_from_config():
    """``/model kimi`` (no --provider) must also see ``providers:`` from config.

    Same scoping bug: alias resolution consulted ``user_providers=None``
    and fell through to the default provider catalog, so the local proxy
    alias was never seen and Hermes picked a direct upstream model instead.
    """
    cli = _make_bare_cli()
    user_provs = _user_providers_fixture()

    with patch("hermes_cli.config.load_config",
               return_value={"providers": user_provs}), \
         patch("hermes_cli.config.get_compatible_custom_providers",
               return_value=[]), \
         patch("hermes_cli.model_switch.switch_model",
               return_value=_fail_result()) as mock_switch:
        cli._handle_model_switch("/model kimi")

    assert mock_switch.called
    kwargs = mock_switch.call_args.kwargs
    assert kwargs.get("explicit_provider") == ""
    assert kwargs.get("user_providers") == user_provs


def test_model_switch_config_load_failure_does_not_crash():
    """If ``load_config`` raises, the handler must fall through with ``None``.

    The ``try / except Exception`` wrapper around config loading is
    load-bearing: the switch path must remain callable even when the
    config is unreadable or malformed.
    """
    cli = _make_bare_cli()

    with patch("hermes_cli.config.load_config",
               side_effect=RuntimeError("boom")), \
         patch("hermes_cli.model_switch.switch_model",
               return_value=_fail_result()) as mock_switch:
        cli._handle_model_switch("/model kimi --provider litellm")

    assert mock_switch.called
    kwargs = mock_switch.call_args.kwargs
    # With a config-load failure we fall through to None — switch_model
    # will then return its own "Unknown provider" error, which is fine;
    # the point is the handler itself does not crash.
    assert kwargs.get("user_providers") is None
    assert kwargs.get("custom_providers") is None
