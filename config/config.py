"""Configuration values for VEP governance agent."""


from .util import FAST_MODEL, HEAVY_MODEL, _env_int

# Default release version (used when release cannot be auto-detected)
DEFAULT_RELEASE: str = "v1.9"

# Known SIGs in the kubevirt project
KNOWN_SIGS: list[str] = ["compute", "network", "storage"]

# Project board numbers by release version
# Each kubevirt release has its own GitHub Project V2 board
VERSION_PROJECT_BOARDS: dict[str, int] = {
    "1.6": 15,
    "1.7": 18,
    "1.8": 19,
    "1.9": 21,
    "1.10": 22,
}

# Model configuration per node type
# Architecture: fetch nodes use lightweight LLM to gather data, analysis nodes use powerful LLM to reason
NODE_MODELS: dict[str, str] = {
    # Deep reasoning nodes - use powerful model for analysis
    "analyze_combined": HEAVY_MODEL,  # Single node that does ALL analysis with full context
    "update_sheets": HEAVY_MODEL,
    "alert_summary": HEAVY_MODEL,  # Generates summary-style notifications

    # Context fetch nodes - use fast model (Flash) to fetch data via GitHub MCP
    # These nodes ONLY fetch raw data, NO analysis - analysis is done by analyze_combined
    "fetch_veps": FAST_MODEL,
    "check_activity": FAST_MODEL,
    "check_compliance": FAST_MODEL,
    "check_deadlines": FAST_MODEL,
    "check_exceptions": FAST_MODEL,
    "merge_vep_updates": FAST_MODEL,  # Simple context merge, no deep reasoning

    # Utility/orchestration nodes omitted - they don't invoke LLMs.
    # (send_email, send_slack, send_notifications, save_state_cache, scheduler, run_monitoring)
    # If added here they'd just fall back to DEFAULT_MODEL via get_model_for_node(), which is unused.
}

# Email notification configuration
EMAIL_RECIPIENTS: list[str] = [
    "iholder@redhat.com",
]

# Agent operation interval (in seconds)
# After each fetch, the full pipeline runs: fetch_veps -> run_monitoring -> analyze_combined -> update_sheets -> alert_summary
FETCH_VEPS_INTERVAL_SECONDS: int = 12 * 60 * 60  # 12 hours

# LLM rate limit retry configuration
LLM_MAX_RETRIES: int = 3
LLM_INITIAL_DELAY: int = 2  # seconds

# Per-tier LLM call settings (tier resolved by node name, see util.HEAVY_NODES)
LLM_TIMEOUT_FAST: int = _env_int("LLM_TIMEOUT_FAST", 300)  # seconds
LLM_TIMEOUT_HEAVY: int = _env_int("LLM_TIMEOUT_HEAVY", 900)  # seconds
# Output token cap per call; thinking tokens count against it.
LLM_MAX_OUTPUT_TOKENS: int = _env_int("LLM_MAX_OUTPUT_TOKENS", 65536)
# SDK-level retries per call. Kept low because wrapper-level retries
# multiply into wall time.
LLM_SDK_MAX_RETRIES: int = _env_int("LLM_SDK_MAX_RETRIES", 1)
# Wrapper-level retries on call timeout (on top of the first attempt).
LLM_TIMEOUT_RETRIES: int = _env_int("LLM_TIMEOUT_RETRIES", 1)

# Fetch-loop context budget (input tokens). Past the budget the tool loop
# stops and the node answers with the data gathered so far; past the hard
# cap the newest tool results are also shrunk so the final call fits the
# model's context window.
FETCH_TOOL_RESULT_MAX_CHARS: int = _env_int("FETCH_TOOL_RESULT_MAX_CHARS", 20000)
FETCH_INPUT_TOKEN_BUDGET: int = _env_int("FETCH_INPUT_TOKEN_BUDGET", 650000)
FETCH_INPUT_TOKEN_HARD_CAP: int = _env_int("FETCH_INPUT_TOKEN_HARD_CAP", 900000)

# Per-node tool-loop iteration overrides. Default is 30 for fetch_veps and
# 10 for everything else (5 in the "test-sheets" debug mode).
NODE_MAX_ITERATIONS: dict[str, int] = {
    "check_compliance": 20,
}
