"""Tests for get_model(): model name, tier settings, and SDK kwargs.

get_model must forward the tier-resolved thinking level (None when empty),
the per-tier timeout (so the SDK aborts the HTTP call itself), the output
token cap, and a low SDK retry count (SDK retries multiply with the
wrapper-level retries in llm_helper). The client class is mocked because
constructing the real one needs an API key.
"""

from unittest.mock import patch

from services.utils import get_model


def test_get_model_passes_model_and_fast_tier_settings():
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm, \
         patch("services.utils.get_api_key", return_value="test-key"), \
         patch("config.DEFAULT_MODEL", "fast-model"), \
         patch("config.get_llm_settings_for_node", return_value=("medium", 300)), \
         patch("config.LLM_MAX_OUTPUT_TOKENS", 65536), \
         patch("config.LLM_SDK_MAX_RETRIES", 1):
        get_model()

    mock_llm.assert_called_once_with(
        model="fast-model",
        api_key="test-key",
        thinking_level="medium",
        timeout=300,
        max_output_tokens=65536,
        max_retries=1,
    )


def test_get_model_node_param_resolves_heavy_tier():
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm, \
         patch("services.utils.get_api_key", return_value="test-key"), \
         patch("config.get_llm_settings_for_node", return_value=("high", 900)) as mock_settings, \
         patch("config.LLM_MAX_OUTPUT_TOKENS", 65536), \
         patch("config.LLM_SDK_MAX_RETRIES", 1):
        get_model("heavy-model", node="analyze_combined")

    mock_settings.assert_called_once_with("analyze_combined")
    mock_llm.assert_called_once_with(
        model="heavy-model",
        api_key="test-key",
        thinking_level="high",
        timeout=900,
        max_output_tokens=65536,
        max_retries=1,
    )


def test_get_model_passes_none_when_thinking_level_empty():
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm, \
         patch("services.utils.get_api_key", return_value="test-key"), \
         patch("config.DEFAULT_MODEL", "fast-model"), \
         patch("config.get_llm_settings_for_node", return_value=(None, 300)), \
         patch("config.LLM_MAX_OUTPUT_TOKENS", 65536), \
         patch("config.LLM_SDK_MAX_RETRIES", 1):
        get_model()

    mock_llm.assert_called_once_with(
        model="fast-model",
        api_key="test-key",
        thinking_level=None,
        timeout=300,
        max_output_tokens=65536,
        max_retries=1,
    )
