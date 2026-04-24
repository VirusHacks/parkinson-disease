# Local Inference Report

- Model: `gpt-4o-mini`
- Server: `http://localhost:8000`
- Tasks: `beta_suppression, tremor_correction, full_episode`
- Request sleep: `0.5` s
- Inter-task sleep: `4.0` s
- Mean score: `0.6663`

## Task Results

| Task | Score | Success |
|---|---:|---:|
| `beta_suppression` | 0.8365 | PASS |
| `tremor_correction` | 0.5895 | PASS |
| `full_episode` | 0.5729 | FAIL |
