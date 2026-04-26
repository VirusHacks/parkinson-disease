# Local Inference Report

- Model: `Qwen/Qwen2.5-72B-Instruct`
- Server: `http://localhost:8000`
- Tasks: `easy`
- Seeds per task: `[0]`
- Request sleep: `0.0` s
- Inter-task sleep: `0.0` s
- Mean score (all rollouts): `0.7929`

## Task Results (aggregated across seeds)

| Task | n | Mean ± Std | Min | Max | Pass | Threshold | Grader ran? |
|---|---:|---|---:|---:|---:|---:|---:|
| `easy` | 1 | 0.7929 ± 0.0000 | 0.7929 | 0.7929 | 1/1 | 0.55 | NO (mean-reward fallback) |

## Per-seed detail

### `easy`

| Seed | Steps | Score | Pass | Mean amp (mA) | Max amp (mA) | Events fired |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 6/36 | 0.7929 | PASS | 1.075 | 1.200 | - |

