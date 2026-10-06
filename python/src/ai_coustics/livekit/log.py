import logging

logger = logging.getLogger("ai_coustics.livekit")

PLUGIN_NAME = "ai-coustics"


def log_fields(component: str, **fields: object) -> dict[str, object]:
    """Return the common structured fields for an ai-coustics log record."""

    return {"plugin": PLUGIN_NAME, "component": component, **fields}
