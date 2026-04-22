"""
DBS Task Definitions for the Parkinson's Motor Environment.
===========================================================

Three tasks ordered by clinical difficulty.  Each task defines:
  - task_id                 : unique string identifier
  - description             : plain-English clinical objective
  - n_steps                 : episode length (subset of the 100-step timeline)
  - start_step              : index into the 100-step calibrated timeline
  - target_force_preserved  : minimum fraction of healthy muscle force (clinical goal)
  - target_beta_arv         : maximum normalised beta ARV acceptable (0=suppressed)
  - max_dbs_amplitude       : hard cap on DBS amplitude (mA) to prevent side-effects
  - max_side_effect_load    : maximum allowed cumulative side-effect load
  - success_threshold       : minimum grader score to count as "success"

Clinical background
-------------------
The Fleming et al. (2023) simulation provides 100 time-steps (t=10.02–12.00 s,
20 ms intervals).  The three tasks slice progressively more of this window:

  Easy   → first 20 steps (t=10.02–10.40 s)  — beta oscillation just starting
  Medium → first 50 steps (t=10.02–11.00 s)  — tremor ramping rapidly
  Hard   → all 100 steps  (t=10.02–12.00 s)  — full clinical DBS episode

All difficulty increases come from *the real brain dynamics* in the calibrated
data — not from injected artificial difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DBSTask:
    """Immutable task specification."""
    task_id: str
    description: str

    # Episode extent
    start_step: int         # first calibrated window index (0-indexed)
    n_steps: int            # how many consecutive 20 ms windows to replay

    # Clinical success thresholds (used by graders)
    target_force_preserved: float   # must stay ABOVE this (fraction of healthy)
    target_beta_arv: float          # must stay BELOW this (normalised, 0=suppressed)
    max_dbs_amplitude: float        # hard cap; agent penalised if exceeded (mA)
    max_side_effect_load: float     # hard cap; agent penalised if exceeded (0-1)
    success_threshold: float        # grader score >= this → episode "success"

    # Human-readable difficulty
    difficulty: str                 # "easy" | "medium" | "hard"


# ── Task 1 — Easy ─────────────────────────────────────────────────────────────
#
# Clinical goal: the agent acts as a fresh DBS programmer at the very START of
# the episode (t=10.02–10.40 s, 20 windows).  Beta is near its pre-DBS peak
# (0.78 normalised) and tremor is still low (0.01–0.17).  Muscle force is still
# 93 % of healthy.  The agent must keep beta suppressed without overshooting on
# amplitude (≤1.0 mA) or inducing side-effects.  Because tremor has not yet
# built up, this is the most forgiving window — a strong DBS pulse is sufficient.
TASK_BETA_SUPPRESSION = DBSTask(
    task_id="beta_suppression",
    description=(
        "Static beta suppression: the episode begins before tremor builds up "
        "(t=10.02–10.40 s).  The agent must tune DBS Amplitude and Pulse Width "
        "to suppress the STN beta oscillation below the clinical threshold "
        "(beta_arv < 0.20) while keeping DBS ≤ 1.0 mA to limit side-effects.  "
        "Muscle force should not drop below 80 %% of the healthy baseline."
    ),
    start_step=0,
    n_steps=20,
    target_force_preserved=0.80,   # ≥ 80 % of healthy motor function
    target_beta_arv=0.20,          # suppress beta to ≤ 20 % of peak
    max_dbs_amplitude=1.0,         # ≤ 1 mA (low side-effect zone)
    max_side_effect_load=0.30,
    success_threshold=0.60,
    difficulty="easy",
)

# ── Task 2 — Medium ───────────────────────────────────────────────────────────
#
# Clinical goal: tremor is now actively ramping (0.17 → 0.80 normalised) while
# the closed-loop controller struggles to maintain force (~51–30 % of healthy).
# The agent must react dynamically — beta spikes require amplitude bursts, but
# sustained high amplitude exhausts the side-effect budget.  Success requires
# keeping force above 35 % AND staying within the side-effect budget.
TASK_TREMOR_CORRECTION = DBSTask(
    task_id="tremor_correction",
    description=(
        "Dynamic tremor correction: the episode covers the rapid tremor build-up "
        "phase (t=10.02–11.00 s, 50 steps).  The agent must balance DBS "
        "amplitude (0–2.0 mA) and pulse width to suppress beta oscillations while "
        "keeping muscle force above 35 %% of healthy.  It must avoid exhausting "
        "the side-effect budget (≤ 0.50) during sustained high stimulation."
    ),
    start_step=0,
    n_steps=50,
    target_force_preserved=0.35,   # ≥ 35 % of healthy (challenging)
    target_beta_arv=0.35,          # keep beta partially suppressed
    max_dbs_amplitude=2.0,         # up to 2 mA permissible
    max_side_effect_load=0.50,
    success_threshold=0.55,
    difficulty="medium",
)

# ── Task 3 — Hard ─────────────────────────────────────────────────────────────
#
# Clinical goal: full 100-step episode (t=10.02–12.00 s).  By step 80+, tremor
# approaches 0.99 and force collapses to ~4 % of healthy.  The agent must find
# a policy that uses aggressive DBS early to slow tremor progression and then
# sustains force through the end of the episode, all while managing the
# cumulative side-effect load across 100 steps.
TASK_FULL_EPISODE = DBSTask(
    task_id="full_episode",
    description=(
        "Full clinical episode: the agent manages DBS for the complete 100-step "
        "closed-loop simulation (t=10.02–12.00 s).  Tremor escalates from near-zero "
        "to near-maximum.  The agent must dynamically adapt amplitude (0–3.0 mA) "
        "and pulse width to maximise cumulative muscle force preservation while "
        "keeping side-effect load below 0.70 across the full episode.  This "
        "mirrors the clinical challenge faced by a DBS programmer optimising "
        "stimulation for a Parkinson's patient over a multi-second therapeutic window."
    ),
    start_step=0,
    n_steps=100,
    target_force_preserved=0.25,   # ≥ 25 % average over full episode
    target_beta_arv=0.45,          # average beta kept at or below 45 %
    max_dbs_amplitude=3.0,         # up to 3 mA allowed
    max_side_effect_load=0.70,
    success_threshold=0.50,
    difficulty="hard",
)

# ── Registry ──────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, DBSTask] = {
    TASK_BETA_SUPPRESSION.task_id:  TASK_BETA_SUPPRESSION,
    TASK_TREMOR_CORRECTION.task_id: TASK_TREMOR_CORRECTION,
    TASK_FULL_EPISODE.task_id:      TASK_FULL_EPISODE,
}

_DEFAULT_TASK_ID = TASK_FULL_EPISODE.task_id


def get_task(task_id: str | None = None) -> DBSTask:
    """
    Return the DBSTask for the given task_id.
    Falls back to the hard 'full_episode' task when task_id is None.
    Raises ValueError for unknown task_ids.
    """
    if task_id is None:
        return TASK_REGISTRY[_DEFAULT_TASK_ID]
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        raise ValueError(
            f"Unknown task_id {task_id!r}. "
            f"Available: {list(TASK_REGISTRY)}"
        )
    return task
