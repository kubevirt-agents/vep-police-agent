"""Tests for LLM tier resolution: node name -> (thinking level, timeout).

The tier must be resolved by node name via the explicit HEAVY_NODES mapping,
NOT by model name - a config where FAST_MODEL == HEAVY_MODEL (e.g. a
single-model sweep) must still give heavy nodes heavy-tier settings.
"""

from unittest.mock import patch

import config
from config import util as config_util


def test_heavy_nodes_get_heavy_tier():
    for node in ("analyze_combined", "update_sheets", "alert_summary"):
        assert config.get_tier_for_node(node) == "heavy"


def test_everything_else_gets_fast_tier():
    for node in (
        "fetch_veps", "check_activity", "check_compliance", "check_deadlines",
        "check_exceptions", "merge_vep_updates", "some_unknown_node", None,
    ):
        assert config.get_tier_for_node(node) == "fast"


def test_llm_settings_follow_node_not_model_name():
    """FAST_MODEL == HEAVY_MODEL must not change tier resolution."""
    with patch.object(config_util, "HEAVY_THINKING_LEVEL", "high"), \
         patch.object(config_util, "FAST_THINKING_LEVEL", "medium"), \
         patch.object(config_util, "FAST_MODEL", "same-model"), \
         patch.object(config_util, "HEAVY_MODEL", "same-model"), \
         patch("config.config.LLM_TIMEOUT_FAST", 300), \
         patch("config.config.LLM_TIMEOUT_HEAVY", 900):
        assert config.get_llm_settings_for_node("analyze_combined") == ("high", 900)
        assert config.get_llm_settings_for_node("update_sheets") == ("high", 900)
        assert config.get_llm_settings_for_node("check_compliance") == ("medium", 300)
        # Unknown nodes and None fall back to the fast tier.
        assert config.get_llm_settings_for_node(None) == ("medium", 300)


def test_empty_thinking_level_maps_to_model_default():
    with patch.object(config_util, "FAST_THINKING_LEVEL", ""), \
         patch("config.config.LLM_TIMEOUT_FAST", 300):
        assert config.get_llm_settings_for_node("fetch_veps") == (None, 300)


def test_env_int_and_bool_helpers(monkeypatch):
    monkeypatch.setenv("TEST_X_INT", "42")
    assert config_util._env_int("TEST_X_INT", 1) == 42

    monkeypatch.setenv("TEST_X_INT", "not-a-number")
    assert config_util._env_int("TEST_X_INT", 7) == 7

    monkeypatch.delenv("TEST_X_INT")
    assert config_util._env_int("TEST_X_INT", 5) == 5

    monkeypatch.setenv("TEST_X_BOOL", "true")
    assert config_util._env_bool("TEST_X_BOOL") is True

    monkeypatch.setenv("TEST_X_BOOL", "YES")
    assert config_util._env_bool("TEST_X_BOOL") is True

    monkeypatch.setenv("TEST_X_BOOL", "no")
    assert config_util._env_bool("TEST_X_BOOL") is False

    monkeypatch.delenv("TEST_X_BOOL")
    assert config_util._env_bool("TEST_X_BOOL", True) is True
