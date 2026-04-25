# Local Inference Report

- Model: `Qwen/Qwen2.5-72B-Instruct`
- Server: `http://localhost:8000`
- Tasks: `easy`
- Seeds per task: `[0]`
- Request sleep: `0.0` s
- Inter-task sleep: `0.0` s
- Mean score (all rollouts): `0.7742`

## Task Results (aggregated across seeds)

| Task | n | Mean ± Std | Min | Max | Pass | Threshold | Grader ran? |
|---|---:|---|---:|---:|---:|---:|---:|
| `easy` | 1 | 0.7742 ± 0.0000 | 0.7742 | 0.7742 | 1/1 | 0.55 | yes |

## Grader component means (only for tasks where grader ran)

### `easy`

| Component | Value |
|---|---:|
| `beta_score` | 0.8889 |
| `efficiency_score` | 0.2041 |
| `force_score` | 0.4749 |
| `hard_failure_penalty` | 0.0400 |
| `overall_score` | 0.7742 |
| `passes_motor_gate` | 1.0000 |
| `passes_safety_gate` | 1.0000 |
| `passes_symptom_gate` | 1.0000 |
| `pre_penalty_score` | 0.8142 |
| `recovery_score` | 0.0430 |
| `safety_score` | 1.0000 |
| `smoothness_score` | 0.9596 |
| `terminal_stability_score` | 0.3152 |
| `therapeutic_engagement` | 0.6920 |
| `tracking_score` | 0.9580 |
| `tremor_score` | 0.7843 |


## Per-seed detail

### `easy`

| Seed | Steps | Score | Pass | Mean amp (mA) | Max amp (mA) | Events fired |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 36/36 | 0.7742 | PASS | 1.062 | 1.250 | — |

