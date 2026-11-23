"""Tests for the config-gated LLM paths.

- ANALYZE_USE_TOOLS=False (default): analyze_combined makes a direct
  structured-output call - no GitHub tools bound, no MCP server spawned.
- ANALYZE_USE_TOOLS=True: the historical tool loop with GitHub MCP.
- ALERT_SUMMARY_USE_LLM=False (default): alert_summary uses the deterministic
  fallback and never invokes the LLM (the historical behaviour - the old
  empty-mcp_names call returned an empty model without calling the LLM).
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from nodes.alert_summary import AlertSummaryResponse, alert_summary_node
from nodes.analyze_combined import analyze_combined_node
from services.response_models import AnalyzeAttentionResponse
from state import VEPActivity, VEPCompliance, VEPInfo, VEPMilestone


def _make_vep(tracking_issue_id: int, name: str) -> VEPInfo:
    return VEPInfo(
        tracking_issue_id=tracking_issue_id,
        name=name,
        title=f"{name} title",
        owner="owner1",
        owning_sig="compute",
        status="Tracked",
        last_updated=datetime.now(UTC),
        created_at=datetime.now(UTC),
        current_milestone=VEPMilestone(
            version="v1.10", status="Tracked", promotion_phase="Net New",
            exception_phase="None", target_stage="Beta", all_code_prs_merged=False,
        ),
        compliance=VEPCompliance(
            template_complete=True, all_sigs_signed_off=True, vep_merged=False,
            prs_linked=True, docs_pr_created=True, labels_valid=True,
        ),
        activity=VEPActivity(last_activity=datetime.now(UTC), days_since_update=1),
        tracking_issue=None,
        analysis={},
    )


def _indexed_context():
    return {
        "release_phase": "development",
        "release_deadlines": {},
        "board_veps": {},
        "phase_detail": {"fraction_through_phase": 0.5},
        "cycle_start_date": None,
    }


def _fake_llm(parsed):
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = {
        "raw": MagicMock(usage_metadata={"input_tokens": 100, "output_tokens": 50}),
        "parsed": parsed,
        "parsing_error": None,
    }
    llm.with_structured_output.return_value = structured_llm
    return llm


def test_analyze_combined_without_tools_spawns_no_mcp():
    vep = _make_vep(1, "vep-0001")
    parsed = AnalyzeAttentionResponse(
        analyses=[], alerts=[], general_insights=[], sheets_need_update=False,
    )
    llm = _fake_llm(parsed)

    with patch("nodes.analyze_combined.create_indexed_context", return_value=_indexed_context()), \
         patch("nodes.analyze_combined.ANALYZE_USE_TOOLS", False), \
         patch("services.llm_helper.get_mcp_tools_by_name") as mock_get_tools, \
         patch("services.llm_helper.get_model", return_value=llm), \
         patch("config.get_model_for_node", return_value="heavy-model"), \
         patch("config.get_llm_settings_for_node", return_value=("high", 900)):
        analyze_combined_node({"veps": [vep], "last_check_times": {}})

    mock_get_tools.assert_not_called()
    llm.bind_tools.assert_not_called()
    llm.with_structured_output.assert_called_once()


def test_analyze_combined_with_tools_binds_github_mcp():
    vep = _make_vep(1, "vep-0001")
    parsed = AnalyzeAttentionResponse(
        analyses=[], alerts=[], general_insights=[], sheets_need_update=False,
    )
    llm = _fake_llm(parsed)
    no_tool_response = MagicMock(tool_calls=[], content="done")
    llm.invoke.return_value = no_tool_response
    github_tool = MagicMock()
    github_tool.name = "get_issue"
    github_tool.func = lambda **kwargs: "issue data"

    with patch("nodes.analyze_combined.create_indexed_context", return_value=_indexed_context()), \
         patch("nodes.analyze_combined.ANALYZE_USE_TOOLS", True), \
         patch("services.llm_helper.get_mcp_tools_by_name", return_value=[github_tool]) as mock_get_tools, \
         patch("services.llm_helper.get_model", return_value=llm), \
         patch("config.get_model_for_node", return_value="heavy-model"), \
         patch("config.get_llm_settings_for_node", return_value=("high", 900)):
        analyze_combined_node({"veps": [vep], "last_check_times": {}})

    mock_get_tools.assert_called_once_with("github")
    llm.bind_tools.assert_called_once()


def test_alert_summary_disabled_never_invokes_llm():
    vep = _make_vep(1, "vep-0001")
    state = {
        "veps": [vep],
        "alerts": [],
        "general_insights": ["insight"],
        "last_check_times": {},
    }

    with patch("nodes.alert_summary.ALERT_SUMMARY_USE_LLM", False), \
         patch("nodes.alert_summary.invoke_llm_with_tools") as mock_invoke:
        result = alert_summary_node(state)

    mock_invoke.assert_not_called()
    # Deterministic fallback summary was produced.
    assert "VEP Police Report" in result["alert_summary_text"]
    assert result["executive_summary"] == ""


def test_alert_summary_enabled_uses_tool_free_llm_call():
    vep = _make_vep(1, "vep-0001")
    state = {
        "veps": [vep],
        "alerts": [],
        "general_insights": [],
        "last_check_times": {},
    }
    llm_result = AlertSummaryResponse(summary_text="LLM-formatted summary")

    with patch("nodes.alert_summary.ALERT_SUMMARY_USE_LLM", True), \
         patch("nodes.alert_summary.invoke_llm_with_tools", return_value=llm_result) as mock_invoke:
        result = alert_summary_node(state)

    mock_invoke.assert_called_once()
    assert mock_invoke.call_args.kwargs.get("mcp_names") == ()
    assert result["alert_summary_text"] == "LLM-formatted summary"
