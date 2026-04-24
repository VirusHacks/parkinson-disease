# Phase 6-10 Completion Summary

This summary reflects the latest benchmark-calibration pass completed on `2026-04-23`.

## What was finished

- Reward and grading calibration for public vs held-out evaluation
- Scenario-specific success thresholds for the medium and hard tasks
- Constraint-aware benchmark penalties so over-cap stimulation is no longer rewarded
- Reproducible seeded public and held-out reporting
- A focused tremor-task policy search with saved artifacts in:
  - `outputs/search/tremor_policy_search.json`
  - `outputs/search/tremor_policy_search.md`

## Current benchmark snapshot

Source of truth:

- `outputs/benchmark/benchmark_eval.json`
- `outputs/benchmark/benchmark_eval.md`
- `outputs/search/tremor_policy_search.json`
- `outputs/search/tremor_policy_search.md`

### `beta_suppression`

- `no_dbs` fails all public runs
- all constant baselines now fail all public runs
- `safety_aware` passes all public runs

### `tremor_correction`

- `no_dbs` fails all public runs
- `const_high` now fails all public runs
- the searched rescue baseline used by `safety_aware` passes all public runs
- held-out `fragile` remains difficult and still fails, which is desirable for generalization pressure

This is the biggest improvement from the final calibration pass.

### `full_episode`

- `no_dbs` fails all public runs under the stricter threshold
- all constant baselines now fail all public runs
- `safety_aware` passes all public runs
- held-out `refractory` remains clearly harder and still fails

## What changed in the final pass

- added a recovery diagnostic to the benchmark score details
- rebalanced the medium-task score to reward rescue while still punishing non-treatment
- tightened hard-task success to prevent passive policies from passing on safety alone
- added explicit penalties for repeated task-envelope violation
- replaced the previous medium-task heuristic with the best rescue preset discovered by the saved search sweep

## Remaining known limitations

- held-out medium and hard scenarios remain intentionally tough enough that simple baselines still fail
- the easy task is now benchmark-clean, but still narrower and less diverse than the medium and hard tasks
- the hard task is calibrated for leaderboard separation rather than to mimic every hardware/programming constraint used clinically

## Files touched in this block

- `parkinsons_Motor/graders/dbs_graders.py`
- `parkinsons_Motor/tasks/dbs_tasks.py`
- `parkinsons_Motor/evaluation/eval_suite.py`
- `parkinsons_Motor/evaluation/tremor_policy_search.py`

## Output artifacts

- `outputs/benchmark/benchmark_eval.json`
- `outputs/benchmark/benchmark_eval.md`
- `outputs/search/tremor_policy_search.json`
- `outputs/search/tremor_policy_search.md`
- `outputs/phase6_10_summary.md`
