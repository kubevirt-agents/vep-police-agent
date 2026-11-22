"""Constants and helper functions for configuration."""

import os

# Model tier constants for node configuration (overridable via env vars)
FAST_MODEL = os.getenv("FAST_MODEL", "gemini-3-flash-preview")
HEAVY_MODEL = os.getenv("HEAVY_MODEL", "gemini-3.1-pro-preview")

# Thinking level per tier; "" = model default.
FAST_THINKING_LEVEL = os.getenv("FAST_THINKING_LEVEL", "")
HEAVY_THINKING_LEVEL = os.getenv("HEAVY_THINKING_LEVEL", "")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", FAST_MODEL)

# Nodes on the heavy tier (powerful model, deeper thinking, longer timeout).
# Everything else is fast tier. The tier is resolved by node name, NOT by
# model name, so a config where FAST_MODEL == HEAVY_MODEL still works.
HEAVY_NODES = frozenset({"analyze_combined", "update_sheets", "alert_summary"})


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to the default on missing/invalid values."""
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a bool env var ("1", "true", "yes" - case-insensitive - mean True)."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes")

AVAILABLE_MODELS = [
    FAST_MODEL,
    HEAVY_MODEL,
]

# Global flag for fastest model mode
_USE_FASTEST_MODEL = False


def set_fastest_model(enabled: bool = True) -> None:
    """Force all nodes to use the fastest model."""
    global _USE_FASTEST_MODEL
    _USE_FASTEST_MODEL = enabled


def is_fastest_model_enabled() -> bool:
    """Check if fastest model mode is enabled."""
    return _USE_FASTEST_MODEL


def get_model_for_node(node_name: str) -> str:
    """Get the model name for a specific node."""
    from .config import NODE_MODELS
    if _USE_FASTEST_MODEL:
        return FAST_MODEL
    return NODE_MODELS.get(node_name, DEFAULT_MODEL)


def get_tier_for_node(node_name: str | None) -> str:
    """Resolve the LLM tier ("heavy" or "fast") for a node/operation name."""
    return "heavy" if node_name in HEAVY_NODES else "fast"


def get_llm_settings_for_node(node_name: str | None) -> tuple[str | None, int]:
    """Return (thinking_level, timeout_seconds) for a node/operation name.

    thinking_level is None when the tier's configured level is "" (model
    default). The timeout is the per-tier call timeout in seconds.
    """
    from .config import LLM_TIMEOUT_FAST, LLM_TIMEOUT_HEAVY
    if get_tier_for_node(node_name) == "heavy":
        return HEAVY_THINKING_LEVEL or None, LLM_TIMEOUT_HEAVY
    return FAST_THINKING_LEVEL or None, LLM_TIMEOUT_FAST


def set_node_model(node_name: str, model: str) -> None:
    """Set the model for a specific node."""
    from .config import NODE_MODELS
    if model not in AVAILABLE_MODELS:
        import warnings
        warnings.warn(f"Model '{model}' not in AVAILABLE_MODELS. Proceeding anyway.")
    NODE_MODELS[node_name] = model


def get_all_node_models() -> dict[str, str]:
    """Get all node model configurations."""
    from .config import NODE_MODELS
    return NODE_MODELS.copy()


def get_email_recipients() -> list[str]:
    """Get email recipients (env var takes precedence over config)."""
    from .config import EMAIL_RECIPIENTS
    env_recipients = os.environ.get("EMAIL_RECIPIENTS")
    if env_recipients:
        return [email.strip() for email in env_recipients.split(",") if email.strip()]
    return EMAIL_RECIPIENTS.copy() if EMAIL_RECIPIENTS else []


def board_search_patterns(version: str) -> list[str]:
    """Build progressively looser search patterns for finding a release board.

    Tries specific patterns first (e.g., "1.9 enhancements tracking"), then
    falls back to looser ones to be resilient against board naming changes.
    """
    v = version.lstrip('v')
    return [
        f"{v} enhancements tracking",
        f"{v} release tracking",
        f"{v} tracking",
        f"{v} enhancement",
        f"{v} release",
    ]


def get_project_board_for_version(version: str | None) -> int | None:
    """Get project board number for a version string like 'v1.8' or '1.8'.

    Args:
        version: Version string (e.g., "v1.8", "1.8", or None)

    Returns:
        Project board number if found, None otherwise
    """
    from .config import VERSION_PROJECT_BOARDS
    if not version:
        return None
    v = version.lstrip("v")
    return VERSION_PROJECT_BOARDS.get(v)


