# Local Inference Report

- Model: `Qwen/Qwen2.5-72B-Instruct`
- Server: `http://localhost:8000`
- Tasks: `easy`
- Seeds per task: `[0]`
- Request sleep: `0.0` s
- Inter-task sleep: `0.0` s
- Mean score (all rollouts): `0.7936`

## Task Results (aggregated across seeds)

| Task | n | Mean ± Std | Min | Max | Pass | Threshold | Grader ran? |
|---|---:|---|---:|---:|---:|---:|---:|
| `easy` | 1 | 0.7936 ± 0.0000 | 0.7936 | 0.7936 | 1/1 | 0.55 | yes |

## Per-seed detail

### `easy`

| Seed | Steps | Score | Pass | Mean amp (mA) | Max amp (mA) | Events fired |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 36/36 | 0.7936 | PASS | 1.075 | 1.250 | — |

