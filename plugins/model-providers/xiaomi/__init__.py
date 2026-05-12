"""Xiaomi MiMo provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — binary thinking toggle only (no effort levels).

    MiMo's API supports a single ``thinking.type`` parameter:
    ``"enabled"`` or ``"disabled"``.  There is no reasoning_effort
    or max_reasoning_tokens control — the server decides effort
    internally when thinking is enabled.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate Hermes reasoning_config into MiMo's native params.

        Returns (extra_body_additions, top_level_kwargs).  The transport
        merges extra_body_additions into extra_body and top_level_kwargs
        directly into the API request.

        None / no config → let MiMo's server defaults apply.
        enabled=False    → thinking.type=disabled.
        enabled=True     → thinking.type=enabled (no effort tuning available).
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        if enabled is False:
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {"type": "enabled"}

        return extra_body, top_level


xiaomi = XiaomiProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
)

register_provider(xiaomi)
