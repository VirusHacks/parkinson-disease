# Why MotorAssistEnv Is Benchmark-Grade

This document explains why MotorAssistEnv is not just a themed simulator or a hackathon demo, but a benchmark-grade environment for adaptive neurostimulation research and agent evaluation.

It is written as a companion to the main [README](../README.md), the calibration notes in [CALIBRATION.md](../CALIBRATION.md), and the saved benchmark artifacts in:

- [outputs/benchmark/benchmark_eval.md](../outputs/benchmark/benchmark_eval.md)
- [outputs/benchmark/benchmark_eval.json](../outputs/benchmark/benchmark_eval.json)
- [outputs/search/tremor_policy_search.md](../outputs/search/tremor_policy_search.md)
- [outputs/search/tremor_policy_search.json](../outputs/search/tremor_policy_search.json)
- [outputs/benchmark/phase6_10_summary.md](../outputs/benchmark/phase6_10_summary.md)

## What “benchmark-grade” means here

A benchmark-grade environment should do more than look realistic. It should:

- represent a meaningful real-world control problem
- have grounded and causally coherent dynamics
- expose clinically or physically interpretable observations and actions
- include objectives that cannot be trivially hacked
- separate weak, naive, unsafe, and adaptive policies cleanly
- support reproducible evaluation
- include held-out difficulty beyond the public leaderboard
- be documented clearly enough that others can understand, run, and critique it

MotorAssistEnv now checks those boxes much more convincingly than the original version.

## 1. The problem itself is benchmark-worthy

Closed-loop DBS programming for Parkinson’s disease is a genuinely difficult sequential decision problem:

- the brain state is dynamic and partially observable
- the effect of stimulation is delayed and non-linear
- over-treatment and under-treatment are both bad
- symptom suppression is not enough on its own; function must be preserved
- the controller must work over time, not just on single-step snapshots

That makes this a strong benchmark setting for RL agents, classical control policies, and LLM-based agents acting through structured actions.

This is already better than many benchmark ideas because the task is not artificial. It is rooted in a real medical-control tradeoff where safety, function, and efficiency genuinely compete.

## 2. It is grounded in published scientific data, not invented toy dynamics

The environment is anchored to outputs from Fleming et al. (2023), a peer-reviewed multivariable DBS control model. That matters because:

- the source signals are physiologically meaningful
- force, tremor, beta activity, stimulation, and side-effect traces come from the same scientific pipeline
- the benchmark inherits a real controller-derived trajectory rather than arbitrary hand-authored curves
- the DBS amplitude/pulse-width sweep is grounded in a real response surface

The calibration layer in [parkinsons_Motor/core/calibration.py](../parkinsons_Motor/core/calibration.py) aligns the source traces, normalizes them, derives clinically meaningful fields, and exposes the entrainment lookup used by the online environment.

So while this is still a benchmark and not a clinical simulator, it starts from real neuroscience outputs instead of synthetic game logic.

## 3. The online environment is action-coupled and stateful

The current environment in [parkinsons_Motor/server/parkinsons_Motor_environment.py](../parkinsons_Motor/server/parkinsons_Motor_environment.py) is benchmark-grade because it is no longer a replay disguised as control.

It now maintains and updates:

- latent beta state
- latent tremor state
- latent force state
- latent sEMG state
- entrainment state
- side-effect state
- fatigue state
- recent stimulation history

These states evolve using:

- the calibrated Fleming trajectory as anchor drift
- patient-profile-specific responsiveness and sensitivity
- action-coupled stimulation effects
- safety accumulation and recovery
- smoothness and recent-history effects
- long-horizon fatigue pressure

That is important because it means actions matter causally. The agent is not just getting credit for replayed outcomes; it is interacting with a closed-loop system whose next state depends on what it chose.

## 4. The state space is benchmark-quality, not bloated or fake

The observation model in [parkinsons_Motor/core/models.py](../parkinsons_Motor/core/models.py) is strong because every exposed field has a role:

- neural state: `beta_arv`, `tremor_arv`, `semg_arv`
- motor state: `force_preserved`, `force_amplitude`, `effective_motor_output`, `task_error`
- disease and control summaries: `disease_severity`, `beta_suppression`
- temporal features: `beta_trend`, `tremor_trend`, `side_effect_rate`
- DBS context: `dbs_amplitude_ma`, `dbs_pulse_width_ms`, `dbs_entrainment`, `recent_dbs_avg_ma`, `recent_dbs_avg_pw_ms`
- safety and policy diagnostics: `side_effect_load`, `action_smoothness_cost`, `dbs_constraint_violation`
- benchmark outputs: `grader_score`, `episode_success`

Why this is benchmark-grade:

- it is clinically interpretable
- it includes short-horizon temporal structure without leaking the full latent state
- it supports adaptive control rather than one-shot regression
- it is rich enough for learning, but still auditable by human readers

The state space is also stronger now because it explicitly separates what the agent needs for control from what exists only for evaluation or metadata.

## 5. The action space is meaningful and clinically motivated

The action space includes:

- `dbs_amplitude`
- `dbs_pulse_width`
- `motor_command`

This is benchmark-grade because:

- every dimension has functional meaning
- the environment clips actions against task envelopes
- over-cap actions are not silently rewarded
- motor performance is part of evaluation, so `motor_command` is not a fake extra dimension
- smoothness and recent-history effects make control stability matter

A weak benchmark often has actions that look realistic but do not affect the final score. That problem has been fixed here.

## 6. The reward and grader are aligned but not trivial

The benchmark now uses a strong split:

- dense per-step reward for learnability
- deterministic episode-end grading for benchmark integrity

The dense reward reflects:

- force preservation
- tracking quality
- beta suppression
- tremor suppression
- safety
- smoothness
- efficiency
- constraint compliance

The final grader in [parkinsons_Motor/graders/dbs_graders.py](../parkinsons_Motor/graders/dbs_graders.py) evaluates:

- `force_score`
- `beta_score`
- `tremor_score`
- `tracking_score`
- `safety_score`
- `smoothness_score`
- `efficiency_score`
- `terminal_stability_score`
- `recovery_score`

It also includes hard-failure logic assembled from [parkinsons_Motor/graders/components.py](../parkinsons_Motor/graders/components.py) and [parkinsons_Motor/graders/rules.py](../parkinsons_Motor/graders/rules.py) for:

- unsafe stimulation
- poor rescue behavior on the medium task
- non-treatment on rescue scenarios
- repeated task-envelope violation
- weak terminal quality on the hard task

That is benchmark-grade because the agent cannot win by optimizing only one number. It has to be good in the clinically relevant sense.

## 7. The tasks form a real curriculum rather than “same task, longer episode”

The task set in [parkinsons_Motor/tasks/scenarios.py](../parkinsons_Motor/tasks/scenarios.py) and [parkinsons_Motor/tasks/registry.py](../parkinsons_Motor/tasks/registry.py) now tests different capabilities:

- `beta_suppression`: early stabilization under a tight safety budget
- `tremor_correction`: active rescue during symptom escalation
- `full_episode`: long-horizon management of cumulative burden and terminal stability

These are not just three lengths of the same rollout. They differ in:

- horizon
- pass threshold
- safety budget
- stimulation envelope
- what the grader emphasizes
- patient-profile pool

That makes the curriculum much more meaningful for both training and evaluation.

## 8. Patient variation makes it more robust than a single canned scenario

The patient-profile system in [parkinsons_Motor/core/patient_profiles.py](../parkinsons_Motor/core/patient_profiles.py) adds:

- `balanced`
- `responsive`
- `fragile`
- `refractory`

Profiles differ in:

- beta and tremor severity
- entrainment responsiveness
- side-effect sensitivity
- recovery rate
- fatigue rate
- motor noise
- effective stimulation envelope

That is benchmark-grade because the agent is not being evaluated on one frozen patient. Even when the public seeds are small, the environment and held-out cases represent a distribution, not a single script.

## 9. The evaluation suite is reproducible and evidence-based

The benchmark harness in [parkinsons_Motor/evaluation/eval_suite.py](../parkinsons_Motor/evaluation/eval_suite.py) makes the project much stronger. It provides:

- fixed public seeds
- fixed held-out configurations
- official baseline ladder
- machine-readable JSON output
- human-readable Markdown reports

The search helper in [parkinsons_Motor/evaluation/tremor_policy_search.py](../parkinsons_Motor/evaluation/tremor_policy_search.py) strengthens this further by showing that the medium-task adaptive baseline was not chosen arbitrarily. It was calibrated from saved search results.

This matters a lot. A benchmark becomes serious when evaluation is automated, reproducible, and saved in artifacts others can inspect.

## 10. The baseline ladder now demonstrates real separation

From [outputs/benchmark/benchmark_eval.md](../outputs/benchmark/benchmark_eval.md):

### Public benchmark

- `beta_suppression`
  - all passive and constant baselines fail
  - `safety_aware` passes `4/4`
- `tremor_correction`
  - `no_dbs` fails `0/4`
  - all constant baselines fail `0/4`
  - `safety_aware` is currently marginal at `2/4`
- `full_episode`
  - all passive and constant baselines fail
  - `safety_aware` passes `4/4`

This is one of the strongest arguments that the environment is benchmark-grade. The ladder is no longer muddled:

- doing nothing is not enough
- naive constant stimulation is not enough
- brute-force stimulation is not enough
- a tuned adaptive controller can solve easy and hard cleanly while still leaving real headroom on medium rescue

That is exactly what a good benchmark should show.

The current base agent is intentionally simple. It is a hand-designed `safety_aware` controller, not a learned policy. Its performance profile is informative:

- it reliably solves the easy public task
- it is only partial on the medium rescue task, which means the rescue setting is still demanding
- it reliably solves the long-horizon hard task because its policy style is conservative and stability-oriented

That asymmetry does not mean the hard task is fake. It means this baseline is better at sustained safe control than fast reactive rescue, which is a useful benchmark signal.

## 11. Held-out scenarios prevent the benchmark from collapsing into public overfitting

The held-out set is important:

- `beta_suppression` held-out uses `fragile`
- `tremor_correction` held-out uses `fragile`
- `full_episode` held-out uses `refractory`

In those held-out cases, even the strong public baseline still fails. That is a feature, not a bug.

It shows:

- the benchmark still has headroom
- the public ladder is not the full story
- there is real generalization pressure
- “solving the leaderboard” is not the same as solving the environment family

That makes the benchmark more credible for research-minded reviewers.

## 12. The project is transparent and auditable

A benchmark is stronger when a reviewer can inspect how it works. This project is strong on transparency:

- calibration logic is readable
- task thresholds are explicit
- grader weights and hard-failure rules are explicit
- patient profiles are explicit
- evaluation seeds and held-out profiles are explicit
- outputs are stored as artifacts

That makes the benchmark easy to critique, reproduce, and improve. Hidden logic often makes benchmarks feel fragile; this repo avoids that.

## 13. It is still honest about its limits

Benchmark-grade does not mean clinically complete.

MotorAssistEnv is still:

- a semi-mechanistic benchmark, not a full patient simulator
- calibrated from a specific source-model family
- simpler than real implanted-hardware deployment
- missing many real clinical constraints such as hardware telemetry limits and full patient personalization

But being honest about those limits makes the benchmark stronger, not weaker. It shows the project understands the difference between:

- “research benchmark”
- “clinical decision system”

That honesty is part of why the benchmark is credible.

## What makes it strong for judges

For a hackathon or demo setting, this project stands out because it combines:

- a serious real-world problem
- published scientific grounding
- causal environment design
- interpretable state/action/reward spaces
- reproducible evaluation
- a clean public baseline ladder
- held-out difficulty for generalization
- strong documentation and saved evidence

That is a much stronger package than:

- a toy RL environment with a medical theme
- a static dashboard built on CSVs
- or a loosely defined “AI for healthcare” concept without measurable evaluation

## Bottom line

MotorAssistEnv is benchmark-grade because it is:

- grounded in real scientific source data
- action-coupled and stateful online
- clinically interpretable in state, action, and reward design
- difficult in the right way
- resistant to trivial reward hacking
- reproducible in evaluation
- transparent in implementation
- supported by a clean public baseline ladder and harder held-out cases

It is not just a simulator that looks impressive. It now behaves like a real benchmark: one where weak policies fail, adaptive policies earn their score, and readers can understand exactly why.
