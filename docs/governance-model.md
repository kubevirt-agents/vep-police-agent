# KubeVirt VEP Governance Model

This document is the reference the VEP Police Agent reasons against. It is
injected verbatim into the analysis prompt so the agent's judgments are
anchored to a repo-versioned, reviewable policy rather than ad-hoc heuristics.

## Mission

The agent's job is to **steer maintainer attention**: for each in-scope VEP,
decide *whether a maintainer needs to look now, and what is being done wrong*,
judged against **exactly where we are in the release cycle**. It is **not** a
merge-probability predictor. Phase position sets expectations; the same state
(e.g. "no implementation PRs") is healthy early in a cycle and alarming late.

Signals that mean "needs attention": no agreement / going nowhere; stale for a
while; reviewer sentiment negative; changes requested but unaddressed; no impl
PRs *late* in the implementation phase; delivered work does not cover the
criteria the VEP promised; policy violations.

## Release cycle phases

Phases are derived from named, reliable freeze dates. `cycle_start` is the
**previous release's Code Freeze** (when master reopens for the next release).

| Phase | Window | Focus |
|---|---|---|
| Design | `cycle_start` → VEP (Enhancement) Freeze | Proposal review & approval |
| Implementation (development) | VEP Freeze → Code Freeze | Land implementation PRs |
| Stabilization | Code Freeze → GA | Testing, docs, bugfix only |
| Post-release | after GA | Cycle done |

Phases are **soft**: they steer focus and priority, they do not hard-gate.

The agent receives a rich temporal object per cycle: `phase`,
`days_into_phase`, `days_left_in_phase`, `fraction_through_phase` (0.0–1.0),
and `next_freeze` (name + date). Use `days_left_in_phase` and
`fraction_through_phase` to calibrate urgency: the same silence is far more
urgent at 90% through Implementation than at 10%.

## Phase playbook — healthy vs. needs-attention

| Phase / position | Healthy (no attention) | Needs attention |
|---|---|---|
| Design, early | proposal PR exists & under review | no proposal PR; VEP not yet Tracked on board |
| Design, near VEP Freeze | proposal reviewed, sign-offs landing | proposal unmerged, no SIG review, no sign-off |
| Impl, just started | no impl PRs yet | (none — normal) |
| Impl, mid | impl PRs open & progressing | impl PR open but silent for weeks; changes-requested unaddressed |
| Impl, near Code Freeze | bulk merged; only small backportable bugfix left | 0 impl PRs; core parts never opened; partial coverage of promised criteria |
| Stabilization | impl merged; docs/tests landing | impl unmerged (won't make it); no exception requested |
| Any | reviews addressed, activity recent | stale / going-nowhere; negative reviewer sentiment |

### Already-designed and landed work

The playbook rows above describe a VEP moving through the cycle. A VEP can
also be nearly done, or fully done, relative to this cycle's target when the
cycle starts (graduations, carry-overs). Judge those against the targeted
stage, not against whether a proposal PR is currently open:

- "Proposal PR exists & under review" includes proposals already **merged or
  approved**. What matters is that a proposal covers the cycle's **targeted
  stage** - the stage this cycle is graduating to (read the PR titles, the
  issue's stage/timeline, and impl PR titles - e.g. a "graduate X to GA" impl
  PR means GA is the target - not just "some proposal merged"). When the
  targeted stage's design is merged or approved, "no open proposal PR" is not
  a problem: the VEP is healthy (no attention) unless the implementation is
  demonstrably blocked or stalled (rotten/hold/needs-rebase labels,
  unaddressed change requests, weeks of silence). Bot activity is not human
  activity: a lifecycle bot commenting on an otherwise silent PR is itself a
  staleness/blocker marker. Implementation not having started yet is normal
  during the Design phase.
- A design merged before the cycle start still covers the stage it targeted.
  This does **not** extend to a VEP targeting a **new** stage this cycle
  (e.g. Beta to GA): earlier-stage merges do not cover the new stage's
  requirements. If no open or merged proposal defines the targeted stage's
  requirements, that is a real gap - flag it.
- Leftover hygiene from landed work - missing labels on the tracking issue,
  stale status fields, the VEP still sitting on the board after completion -
  is a cleanup note (at most "watch"), not needs-attention, once the
  targeted stage has landed.

## Policy & compliance checklist

Enumerated from `kubevirt/enhancements` (feature-lifecycle, templates, README
exceptions). Flag violations as attention items:

- A PR must **reference an existing VEP tracking issue** (`enhancements#N` or
  `enhancements/issues/N`); reject the literal placeholder.
- **Feature gate** required where applicable.
- **VEP template complete** (no unfilled required sections).
- **Release sign-off** checklist satisfied for the target stage.
- Stage duration guidance (warn only): Alpha ≤ 1–2 releases, Beta ≤ 1–3.
- Post-freeze work needs an **exception** requested via the kubevirt-dev list.

## Attribution rules (context for reasoning)

- VEP number == the GitHub **tracking issue** number (KEP model).
- **One PR → at most one VEP**, via the PR's referenced tracking issue only.
  No transitive "Depends on"; sub-VEP issue bodies listing a parent's design
  PRs must not leak onto the sub-VEP.
- The **current-cycle project board is the cycle oracle**: a VEP is in-scope
  iff it is on the board with a live Status (not Removed / Complete) and is not
  next-cycle-only. Ignore next-cycle VEPs entirely.
- Open PR referencing an in-scope VEP = current-cycle candidate **regardless of
  age** (staleness is an attention signal, not an exclusion). Merged PR belongs
  to the cycle its `mergedAt` falls in. Closed-unmerged = dropped.
