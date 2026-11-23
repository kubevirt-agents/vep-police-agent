"""Replay analyze_combined against a cached state - LLM tuning harness.

Loads a state cache (default: the subset cache written by --veps runs),
strips each VEP's analysis so attention is recomputed from scratch, runs
analyze_combined_node directly (no graph, no scheduler, no board/sheets/
notification writes), and prints a per-VEP table plus wall time per batch.
Results are also written to --out as JSON.

Model, thinking level, and timeout come from the usual configuration
(HEAVY_MODEL, HEAVY_THINKING_LEVEL, LLM_TIMEOUT_HEAVY env vars; the
analyze_combined node is on the heavy tier). This script adds no overrides.

Usage (inside the container, from /workspace):
    python3 scripts/replay_analyze.py
    python3 scripts/replay_analyze.py --state-cache cache/state_cache.json --veps 491,465 --out output/replay.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Make the repo root importable when invoked as "python3 scripts/replay_analyze.py"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import load_state_cache
from services.utils import log


def _filter_veps(veps, tokens: list[str]) -> list:
    """Keep VEPs whose tracking issue number or name matches a token."""
    keep: set[str] = {t.strip().lower() for t in tokens if t.strip()}
    if not keep:
        return veps
    return [
        vep for vep in veps
        if str(vep.tracking_issue_id) in keep or vep.name.lower() in keep
    ]


def _collect_batch_timings():
    """Wrap analyze_combined's LLM call to record wall time per batch."""
    import nodes.analyze_combined as analyze_module

    timings: list[float] = []
    original = analyze_module.invoke_llm_check

    def timed_invoke(*args, **kwargs):
        start = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            timings.append(time.monotonic() - start)

    analyze_module.invoke_llm_check = timed_invoke
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay analyze_combined on a cached state (no external writes)"
    )
    parser.add_argument(
        "--state-cache", type=Path, default=Path("cache/state_cache_subset.json"),
        help="State cache file to load (default: cache/state_cache_subset.json)"
    )
    parser.add_argument(
        "--veps", type=str, default=None,
        help="Optional comma-separated VEP numbers/names to keep (e.g. '491,vep-0465')"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("output/replay_analyze.json"),
        help="JSON output path (default: output/replay_analyze.json)"
    )
    args = parser.parse_args()

    from nodes.analyze_combined import analyze_combined_node

    cached = load_state_cache(args.state_cache)
    if not cached:
        log(f"No valid state cache at {args.state_cache}", node="replay_analyze", level="ERROR")
        return 1

    veps = cached.get("veps", [])
    if args.veps:
        tokens = args.veps.split(",")
        veps = _filter_veps(veps, tokens)
        log(f"VEP filter: kept {len(veps)} VEP(s) ({', '.join(t.strip() for t in tokens if t.strip())})",
            node="replay_analyze")
    if not veps:
        log("No VEPs to analyze", node="replay_analyze", level="ERROR")
        return 1

    # Strip existing analysis (including _deterministic_attention) so
    # attention is fully recomputed.
    for vep in veps:
        vep.analysis = {}

    log(f"Replaying analyze_combined on {len(veps)} VEP(s) from {args.state_cache}", node="replay_analyze")

    state = {
        "veps": veps,
        "release_schedule": cached.get("release_schedule"),
        "current_release": cached.get("current_release"),
        "general_insights": [],
        "alerts": [],
        # Generous index-cache age: replay should reuse the index written by
        # the paired fetch run (same data for every replayed config) instead
        # of rebuilding it against live GitHub, which would need credentials
        # and shift the prompts between replay runs.
        "index_cache_minutes": 240,
        "mock_analyzed_combined": False,
        "skip_monitoring": False,
        "last_check_times": {},
    }

    batch_timings = _collect_batch_timings()
    start = time.monotonic()
    result = analyze_combined_node(state)
    total_seconds = time.monotonic() - start

    table = result.get("vep_summary_table", [])
    updated_veps = result.get("veps", veps)
    attention_by_id = {
        vep.tracking_issue_id: (vep.analysis or {}).get("attention", {})
        for vep in updated_veps
    }

    print()
    print(f"{'VEP':>6}  {'URGENCY':<8}  COMMENT")
    print("-" * 100)
    for row in sorted(table, key=lambda r: r.get("vep_number") or 0):
        comment = (row.get("status_comment") or "")[:80]
        print(f"{row.get('vep_number'):>6}  {row.get('urgency', '?'):<8}  {comment}")
    print("-" * 100)

    print(f"Total: {len(table)} VEP(s) in {total_seconds:.1f}s")
    for i, seconds in enumerate(batch_timings, start=1):
        print(f"Batch {i}/{len(batch_timings)}: {seconds:.1f}s")

    out = {}
    for row in table:
        number = row.get("vep_number")
        attention = attention_by_id.get(number, {})
        out[str(number)] = {
            "urgency": row.get("urgency"),
            "comment": row.get("status_comment"),
            "attention_level": attention.get("attention_level"),
            "health_summary": attention.get("health_summary"),
            "suggested_action": attention.get("suggested_action"),
            "phase": attention.get("phase"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Replay results written to {args.out}", node="replay_analyze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
