# Local Inference Report

- Model: `Qwen/Qwen2.5-72B-Instruct`
- Server: `http://localhost:8000`
- Tasks: `easy, medium, hard`
- Seeds per task: `[None]`
- Request sleep: `0.5` s
- Inter-task sleep: `4.0` s
- Mean score (all rollouts): `0.6140`

## Task Results (aggregated across seeds)

| Task | n | Mean ± Std | Min | Max | Pass | Threshold | Grader ran? |
|---|---:|---|---:|---:|---:|---:|---:|
| `easy` | 1 | 0.7951 ± 0.0000 | 0.7951 | 0.7951 | 1/1 | 0.55 | yes |
| `medium` | 1 | 0.4525 ± 0.0000 | 0.4525 | 0.4525 | 0/1 | 0.52 | NO (mean-reward fallback) |
| `hard` | 1 | 0.5944 ± 0.0000 | 0.5944 | 0.5944 | 0/1 | 0.68 | NO (mean-reward fallback) |

## Grader component means (only for tasks where grader ran)

### `easy`

| Component | Value |
|---|---:|
| `beta_score` | 0.9480 |
| `efficiency_score` | 0.2089 |
| `force_score` | 0.5228 |
| `hard_failure_penalty` | 0.0400 |
| `overall_score` | 0.7951 |
| `passes_motor_gate` | 1.0000 |
| `passes_safety_gate` | 1.0000 |
| `passes_symptom_gate` | 1.0000 |
| `pre_penalty_score` | 0.8351 |
| `recovery_score` | 0.0657 |
| `safety_score` | 1.0000 |
| `smoothness_score` | 0.9592 |
| `terminal_stability_score` | 0.3386 |
| `therapeutic_engagement` | 0.7253 |
| `tracking_score` | 0.9486 |
| `tremor_score` | 0.7726 |


## Per-seed detail

### `easy`

| Seed | Steps | Score | Pass | Mean amp (mA) | Max amp (mA) | Events fired |
|---:|---:|---:|---:|---:|---:|---|
| · | 36/36 | 0.7951 | PASS | 1.078 | 1.200 | — |

### `medium`

| Seed | Steps | Score | Pass | Mean amp (mA) | Max amp (mA) | Events fired |
|---:|---:|---:|---:|---:|---:|---|
| · | 50/60 | 0.4525 | FAIL | 1.102 | 1.460 | dyskinesia_spike@26-32 |

### `hard`

| Seed | Steps | Score | Pass | Mean amp (mA) | Max amp (mA) | Events fired |
|---:|---:|---:|---:|---:|---:|---|
| · | 30/150 | 0.5944 | FAIL | 1.095 | 1.400 | tachyphylaxis@65-83, off_med_crisis@96-106, motor_surge@50-59, motor_surge@72-77 |

