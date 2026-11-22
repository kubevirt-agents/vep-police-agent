"""Tests for the --veps debug subset mode.

A subset run must: filter fetch_veps output to the named VEPs, force the
skip flags so it can never touch the board/sheets/notifications, write its
state cache to a separate file, and skip detect_changes + snapshots so the
next full run does not report the other VEPs as removed.
"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from main import apply_vep_filter_mode, get_initial_state
from nodes.fetch_veps import _filter_veps
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
    )


def test_filter_veps_by_number_and_name():
    veps = [
        _make_vep(491, "vep-0491"),
        _make_vep(465, "vep-0465"),
        _make_vep(500, "vep-0500"),
    ]

    kept = _filter_veps(veps, ["491", "vep-0465"])

    assert [v.tracking_issue_id for v in kept] == [491, 465]


def test_filter_veps_accepts_loose_name_forms():
    veps = [_make_vep(491, "vep-0491")]

    assert _filter_veps(veps, ["VEP 491"]) == veps
    assert _filter_veps(veps, ["vep-491"]) == veps


def test_filter_veps_no_match_returns_empty():
    assert _filter_veps([_make_vep(491, "vep-0491")], ["999"]) == []


def test_apply_vep_filter_mode_forces_skip_flags():
    args = SimpleNamespace(
        veps="491, vep-0465",
        skip_update_board=False, skip_sheets=False,
        skip_send_email=False, skip_send_slack=False,
    )

    vep_filter = apply_vep_filter_mode(args)

    assert vep_filter == ["491", "vep-0465"]
    assert args.skip_update_board is True
    assert args.skip_sheets is True
    assert args.skip_send_email is True
    assert args.skip_send_slack is True


def test_apply_vep_filter_mode_noop_without_flag():
    args = SimpleNamespace(
        veps=None,
        skip_update_board=False, skip_sheets=False,
        skip_send_email=False, skip_send_slack=False,
    )

    assert apply_vep_filter_mode(args) == []
    assert args.skip_update_board is False
    assert args.skip_sheets is False


def test_get_initial_state_carries_vep_filter():
    assert get_initial_state(vep_filter=["491"])["vep_filter"] == ["491"]
    assert get_initial_state()["vep_filter"] == []


def test_subset_run_writes_separate_state_cache(tmp_path, monkeypatch):
    from nodes import save_state_cache

    main_cache = tmp_path / "state_cache.json"
    subset_cache = tmp_path / "state_cache_subset.json"
    monkeypatch.setattr(save_state_cache, "CACHE_FILE", main_cache)
    monkeypatch.setattr(save_state_cache, "SUBSET_CACHE_FILE", subset_cache)
    monkeypatch.setattr(save_state_cache, "save_snapshot", lambda data: None)

    vep = _make_vep(491, "vep-0491")
    subset_state = {
        "veps": [vep], "vep_filter": ["491"], "last_check_times": {},
        "general_insights": [], "alerts": [],
    }
    save_state_cache.save_state_cache_node(subset_state)

    assert subset_cache.exists()
    assert not main_cache.exists()
    data = json.loads(subset_cache.read_text())
    assert len(data["veps"]) == 1

    # A full run still writes the prod cache.
    full_state = {
        "veps": [vep, _make_vep(465, "vep-0465")], "vep_filter": [],
        "last_check_times": {}, "general_insights": [], "alerts": [],
    }
    save_state_cache.save_state_cache_node(full_state)

    assert main_cache.exists()
    assert len(json.loads(main_cache.read_text())["veps"]) == 2


def test_subset_run_skips_history_snapshot(tmp_path, monkeypatch):
    from nodes import save_state_cache

    monkeypatch.setattr(save_state_cache, "CACHE_FILE", tmp_path / "main.json")
    monkeypatch.setattr(save_state_cache, "SUBSET_CACHE_FILE", tmp_path / "subset.json")

    def boom(data):
        raise AssertionError("subset run must not write a history snapshot")

    monkeypatch.setattr(save_state_cache, "save_snapshot", boom)
    save_state_cache.save_state_cache_node({
        "veps": [], "vep_filter": ["491"], "last_check_times": {},
    })


def test_subset_run_skips_detect_changes(monkeypatch):
    from nodes import detect_changes

    def boom():
        raise AssertionError("subset run must not load previous snapshots")

    monkeypatch.setattr(detect_changes, "load_previous_snapshot", boom)

    result = detect_changes.detect_changes_node({
        "vep_filter": ["491"], "last_check_times": {}, "veps": [],
    })

    assert "detected_changes" not in result


def test_subset_run_skips_snapshot_node(monkeypatch):
    from nodes import snapshot

    def boom(state, cycle_duration=0):
        raise AssertionError("subset run must not write snapshots")

    monkeypatch.setattr(snapshot, "dump_snapshot", boom)

    result = snapshot.snapshot_node({"vep_filter": ["491"]})

    assert result == {}
