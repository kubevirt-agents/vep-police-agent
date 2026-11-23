"""Tests for the design-phase impl_pr_summary guard data.

The "Already-designed and landed work" rule requires impl PR blockers to stay
visible during the design phase even though full implementation_prs are
stripped from the LLM context. These tests pin:
- blocking-label classification,
- the compact summary shape (labels, staleness, bot-only activity flag),
- current-release filtering (previous-release-only VEPs get an empty summary),
- the design-phase context swap (implementation_prs popped, impl_pr_summary
  added) and that other phases keep implementation_prs untouched.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from nodes.analyze_combined import (
    _days_since,
    _impl_pr_summary,
    _is_blocking_label,
    _is_bot_only_activity,
    analyze_combined_node,
)
from services.response_models import AnalyzeAttentionResponse
from state import PRInfo, VEPActivity, VEPCompliance, VEPInfo, VEPMilestone

CUTOFF = datetime(2026, 10, 21, tzinfo=UTC)
NOW = datetime(2026, 11, 22, tzinfo=UTC)


def _make_pr(number: int, state: str = "open", labels: list[str] | None = None,
             merged_at: datetime | None = None, conversation: dict | None = None) -> PRInfo:
    return PRInfo(
        number=number,
        title=f"PR #{number}",
        url=f"https://github.com/kubevirt/kubevirt/pull/{number}",
        state=state,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 11, 20, tzinfo=UTC),
        author="author1",
        merged=state == "merged",
        merged_at=merged_at,
        labels=labels or [],
        conversation=conversation or {},
    )


def _make_vep(impl_prs: list[PRInfo]) -> VEPInfo:
    return VEPInfo(
        tracking_issue_id=10,
        name="vep-0010",
        title="VEP 10 title",
        owner="owner1",
        owning_sig="compute",
        status="Tracked",
        last_updated=datetime.now(UTC),
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        current_milestone=VEPMilestone(
            version="v1.11", status="Tracked", promotion_phase="Net New",
            exception_phase="None", target_stage="Alpha", all_code_prs_merged=False,
        ),
        compliance=VEPCompliance(
            template_complete=True, all_sigs_signed_off=True, vep_merged=False,
            prs_linked=True, docs_pr_created=True, labels_valid=True,
        ),
        activity=VEPActivity(last_activity=datetime.now(UTC), days_since_update=1),
        tracking_issue=None,
        implementation_prs=impl_prs,
        analysis={},
    )


def test_is_blocking_label():
    assert _is_blocking_label("do-not-merge/hold")
    assert _is_blocking_label("do-not-merge/blocked")
    assert _is_blocking_label("lifecycle/rotten")
    assert _is_blocking_label("needs-rebase")
    assert not _is_blocking_label("lifecycle/stale")
    assert not _is_blocking_label("sig/compute")
    assert not _is_blocking_label("approved-vep")


def test_is_bot_only_activity():
    assert _is_bot_only_activity({"last_comment_author": "kubevirt-commenter-bot"})
    assert _is_bot_only_activity({"last_comment_author": "some-app[bot]"})
    assert _is_bot_only_activity({"last_comment_author": "kubevirt-prow"})
    assert not _is_bot_only_activity({"last_comment_author": "0xFelix"})
    assert not _is_bot_only_activity({})


def test_days_since():
    assert _days_since((NOW - timedelta(days=40)).isoformat(), NOW) == 40
    assert _days_since((NOW - timedelta(days=40)).replace(tzinfo=None).isoformat(), NOW) == 40
    assert _days_since(None, NOW) is None
    assert _days_since("not-a-date", NOW) is None
    assert _days_since((NOW + timedelta(days=5)).isoformat(), NOW) == 0


def test_impl_pr_summary_carries_blockers_and_staleness():
    conversation = {
        "last_commit_pushed_at": (NOW - timedelta(days=88)).isoformat(),
        "days_since_last_human_activity": 4,
        "last_comment_author": "kubevirt-commenter-bot",
        "approved_count": 0,
        "changes_requested_unaddressed": False,
    }
    held = _make_pr(19638, labels=["do-not-merge/hold", "sig/compute"], conversation=conversation)
    vep = _make_vep([held])

    summary = _impl_pr_summary(vep, CUTOFF)

    assert len(summary) == 1
    entry = summary[0]
    assert entry["number"] == 19638
    assert entry["state"] == "open"
    assert entry["blocking_labels"] == ["do-not-merge/hold"]
    assert entry["days_since_last_commit"] == 88
    assert entry["days_since_last_human_activity"] == 4
    assert entry["recent_activity_bot_only"] is True
    assert entry["approved_count"] == 0


def test_impl_pr_summary_filters_to_current_release():
    # Open PR (current) + merged-before-cutoff PR (previous) + abandoned PR.
    current_open = _make_pr(101, state="open", labels=["needs-rebase"])
    previous_merged = _make_pr(102, state="merged",
                               merged_at=datetime(2026, 9, 1, tzinfo=UTC))
    abandoned = _make_pr(103, state="closed")
    vep = _make_vep([current_open, previous_merged, abandoned])

    summary = _impl_pr_summary(vep, CUTOFF)

    assert [e["number"] for e in summary] == [101]
    assert summary[0]["blocking_labels"] == ["needs-rebase"]

    # All impl PRs predate the cycle -> no current-release impl work at all.
    vep_all_previous = _make_vep([
        _make_pr(102, state="merged", merged_at=datetime(2026, 9, 1, tzinfo=UTC)),
    ])
    assert _impl_pr_summary(vep_all_previous, CUTOFF) == []


def _indexed_context(release_phase: str) -> dict:
    return {
        "release_phase": release_phase,
        "release_deadlines": {},
        "board_veps": {},
        "phase_detail": {"fraction_through_phase": 0.4},
        "cycle_start_date": "2026-10-21",
    }


def _run_analyze(vep: VEPInfo, release_phase: str) -> dict:
    """Run analyze_combined_node with a stubbed LLM; return the batch context."""
    captured = {}

    def fake_invoke(op_name, batch_context, system_prompt, user_prompt, response_model, use_tools=False):
        captured["batch_context"] = batch_context
        captured["user_prompt"] = user_prompt
        return AnalyzeAttentionResponse(
            analyses=[], alerts=[], general_insights=[], sheets_need_update=False,
        )

    with patch("nodes.analyze_combined.create_indexed_context",
               return_value=_indexed_context(release_phase)), \
         patch("nodes.analyze_combined.invoke_llm_check", side_effect=fake_invoke):
        analyze_combined_node({"veps": [vep], "last_check_times": {}})
    return captured


def test_design_phase_swaps_impl_prs_for_summary():
    held = _make_pr(19638, labels=["do-not-merge/hold"], conversation={
        "last_commit_pushed_at": (NOW - timedelta(days=88)).isoformat(),
        "days_since_last_human_activity": 4,
        "last_comment_author": "kubevirt-commenter-bot",
        "approved_count": 0,
        "changes_requested_unaddressed": False,
    })
    vep = _make_vep([held])
    vep.board_fields = {"impl_prs": [{"number": 999}], "fields": {}}

    captured = _run_analyze(vep, "design")
    vep_data = captured["batch_context"]["veps"][0]

    assert "implementation_prs" not in vep_data
    assert "impl_prs" not in vep_data["board_fields"]
    summary = vep_data["impl_pr_summary"]
    assert [e["number"] for e in summary] == [19638]
    assert summary[0]["blocking_labels"] == ["do-not-merge/hold"]
    assert summary[0]["recent_activity_bot_only"] is True
    # The guard text must be present in the prompt the model receives.
    assert "BLOCKED IMPL" in captured["user_prompt"]
    assert "STAGE GAP" in captured["user_prompt"]


def test_non_design_phase_keeps_full_impl_prs():
    pr = _make_pr(19638, labels=["do-not-merge/hold"])
    vep = _make_vep([pr])

    captured = _run_analyze(vep, "development")
    vep_data = captured["batch_context"]["veps"][0]

    assert "implementation_prs" in vep_data
    assert vep_data["implementation_prs"][0]["labels"] == ["do-not-merge/hold"]
    assert "impl_pr_summary" not in vep_data
