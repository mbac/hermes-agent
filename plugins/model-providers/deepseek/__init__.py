"""DeepSeek provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class DeepSeekProfile(ProviderProfile):
    """DeepSeek — reasoning_effort + extra_body.thinking pair."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate Hermes reasoning_config into DeepSeek-native params.

        Returns (extra_body_additions, top_level_kwargs).  The transport
        merges extra_body_additions into extra_body and top_level_kwargs
        directly into the API request.

        None / no config → let DeepSeek default (thinking enabled, medium).
        enabled=False    → reasoning_effort="none" + thinking disabled.
        enabled=True     → reasoning_effort=<effort> + thinking enabled.
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            # No config — DeepSeek's server defaults apply.
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        if enabled is False:
            extra_body["thinking"] = {"type": "disabled"}
            top_level["reasoning_effort"] = "none"
            return extra_body, top_level

        # Thinking enabled — map the effort level.
        extra_body["thinking"] = {"type": "enabled"}
        effort = (reasoning_config.get("effort") or "").strip().lower()
        if effort in ("minimal", "low", "medium", "high", "xhigh"):
            top_level["reasoning_effort"] = effort
        else:
            top_level["reasoning_effort"] = "medium"

        return extra_body, top_level


deepseek = DeepSeekProfile(
    name="deepseek",
    aliases=("deepseek-chat",),
    env_vars=("DEEPSEEK_API_KEY",),
    display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API",
    signup_url="https://platform.deepseek.com/",
    fallback_models=(
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    base_url="https://api.deepseek.com/v1",
)

register_provider(deepseek)
