# MotorAssistEnv: Closed-Loop Deep Brain Stimulation for Parkinson's Disease

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/openenv/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MotorAssistEnv is an OpenEnv-compatible reinforcement learning environment where an agent acts as an adaptive Deep Brain Stimulation (DBS) programmer for a patient with Parkinson's disease. At each 20 ms timestep, the agent observes pathological brain and motor signals, chooses DBS settings, and must preserve motor function while staying within a safety budget.

The environment is grounded in calibrated outputs from the peer-reviewed Fleming et al. (2023) biophysical simulation — real neural and motor dynamics, not toy physics. It features a 10-task benchmark ladder spanning easy titration through expert-level long-horizon crisis management, with deterministic-but-seeded stochastic events, patient-profile variation, and multi-objective grading aligned to clinical reality.

## Why this project matters

Parkinson's disease disrupts the basal ganglia through pathological beta-band synchrony in the subthalamic nucleus (STN). That shows up as tremor, rigidity, and degraded voluntary movement. In practice, patients lose agency over basic tasks — reaching, holding objects, writing.

DBS is one of the most effective treatments, but programming it is still largely manual. Too weak: pathological oscillations persist. Too strong: dyskinesia, discomfort, battery drain. Settings drift as disease state changes. That makes DBS programming a hard sequential decision problem:

- Each action changes the next brain state.
- Patient state is non-stationary — tremor builds, L-DOPA wears off, tolerance accumulates.
- The agent sees only partial physiological information, mirroring real closed-loop DBS hardware.
- The objective is multi-objective: restore function, suppress pathological activity, remain safe.

MotorAssistEnv frames that challenge as a rigorous benchmark for RL and LLM-driven agents.

## What the environment models

The environment is backed by outputs from:

> Fleming, J.E., Senneff, S. and Lowery, M.M. (2023), *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*, Journal of Neural Engineering, 20(5), 056029.

This model connects brain activity, DBS control, tremor, EMG, and muscle force in one pipeline with physically meaningful values. The packaged source data in `parkinsons_Motor/fleming-model-based-brain/` includes:

- A 100-step DBS-active timeline covering `t=10.02s` to `t=12.00s`
- Controller and observer CSVs for beta activity, tremor, side effects, stimulation, and force
- Large force and sEMG traces used during calibration
- A `12 × 15` DBS parameter sweep mapping amplitude and pulse width to cortical entrainment

## Core idea

At every step, the agent receives a compact summary of the patient's current neural and motor state. It selects:

- `dbs_amplitude` — stimulation current in mA
- `dbs_pulse_width` — pulse duration in ms
- `motor_command` — intended voluntary movement

Those actions feed into a calibrated entrainment lookup plus online latent dynamics. The environment tracks patient-specific beta, tremor, force, fatigue, entrainment, and side-effect states, updating them with task constraints, patient-profile coefficients, and any active stochastic events. Stronger targeted stimulation suppresses beta and reduces tremor, but excessive stimulation accumulates side effects, degrades force, and may trigger hard-failure penalties.

## Task ladder

The benchmark has 10 tasks across four difficulty tiers. All share the same action/observation schema but differ in episode length, patient type, stochastic events, biomarker targets, safety budget, and grader weights.

### Public tasks (easy / medium / hard)

| Task ID | Name | Steps | Patient | Events | Score threshold |
|---|---|---:|---|---|---:|
| `easy` | Calm Start | 36 | Responsive | None | **0.55** |
| `medium` | Rescue Phase | 60 | Balanced | Mild rescue | **0.52** |
| `hard` | Full Episode | 150 | Refractory | Heavy multi-crisis | **0.68** |

**Easy** — A responsive patient with good beta responsiveness and fast recovery. No stochastic events. The agent must titrate DBS early and hold the therapeutic window. A basic reactive controller passes.

**Medium** — A balanced (typical) patient mid-deterioration. The `rescue` event profile may fire a second deterioration wave (55 % probability) and/or mild dyskinesia pressure (30 % probability). The agent must detect the worsening trajectory and intervene — passive low-amplitude strategies fail. A good reactive controller should pass.

**Hard** — A refractory patient: weak cortical entrainment, high progression rate, slow recovery. The `long_horizon` event profile fires near-guaranteed crises across the 150-step episode:
- **Tachyphylaxis** (82 % probability, 12–20 steps): axonal tolerance builds — the same delivered current provides progressively less beta suppression. The agent must detect declining entrainment and adapt.
- **Off-medication crisis** (75 % probability, 10–15 steps): L-DOPA trough drives a sharp beta surge (+0.28 × intensity per step). The agent must increase amplitude to compensate, but the tightened safety budget (max side-effect load 0.40) punishes over-treatment.
- **Dyskinesia spike** (80 % probability, up to 2 occurrences): cumulative DBS over-treatment dramatically accelerates side-effect burden (×1.65 × intensity). Brute-force high amplitude is actively punished.
- **Motor surge** (65 % probability, up to 2 occurrences): the target motor output jumps to a high-force demand, requiring simultaneous DBS adjustment and tracking correction.

Hard targets are tighter than medium: `target_beta_arv = 0.21`, `target_tremor_arv = 0.27`, `max_side_effect_load = 0.40`. Even an excellent reactive agent must manage overlapping crises, pace its safety budget across 150 steps, and finish the episode stable.

**Difficulty is strictly ordered** — the constant 1.0 mA baseline scores easy (0.72–0.80) > medium (0.47–0.52) > hard (0.23–0.36) across all seeds. Hard never passes its threshold with any constant policy.

### Expert tasks

| Task ID | Name | Steps | Clinical scenario | Score threshold |
|---|---|---:|---|---:|
| `fragile_patient` | Fragile Window | 64 | Tight safety budget (0.26 max) — narrow therapeutic window on a sensitive patient | 0.44 |
| `refractory_patient` | Drug-Resistant | 120 | Weak entrainment, recurring tachyphylaxis, requires pulsed strategies | 0.46 |
| `personalization_generalization` | Mixed Profiles | 90 | Agent must adapt to fragile / balanced / responsive / refractory profiles in sequence | 0.50 |
| `exercise_bout` | Exercise Burst | 70 | Sustained high-motor-demand with possible post-exertion dyskinesia | 0.55 |
| `medication_interaction` | L-DOPA Interaction | 100 | Guaranteed off-med crisis followed by likely dyskinesia — phase-coupled dilemma | 0.50 |
| `nocturnal_transition` | Sleep Transition | 150 | Awake → wind-down → sleep; targets tighten progressively in sleep phase | 0.55 |
| `surgical_followup` | Post-Implant | 120 | Hard amplitude cap (0.6 mA) during microlesion window; possible impedance surges | 0.50 |

## Stochastic event system

Each task opts into a named event profile. The `EventScheduler` generates the full episode timeline at `reset()` time using a seeded RNG — a fixed `(task_id, seed)` pair always produces the same event sequence, preserving reproducibility.

Events are mechanically wired into the environment physics, not just scoring penalties:

| Event | Mechanical effect |
|---|---|
| `tachyphylaxis` | `entrainment_mult = max(0.40, 1.0 − 2.0 × intensity)` — cuts entrainment up to 60 % at max intensity |
| `off_med_crisis` | `beta_drive_add += 0.28 × intensity` per step — genuine beta-state spike the agent must compensate |
| `dyskinesia_spike` | `side_effect_burden_mult × (1.0 + 1.65 × intensity)` — accelerates side-effect accumulation, punishing high amplitude |
| `impedance_surge` | `delivered_amp_mult` drops — the same requested amplitude delivers less current to the STN |
| `motor_surge` | Target output overrides to a high-force demand — agent must track a different motor goal while maintaining DBS |
| `second_deterioration` | Adds moderate beta and tremor drive — second symptom wave during a rescue episode |

Active events are exposed in observation metadata (`active_events`) at each step so agents can observe them.

## Patient profiles

Each task is assigned a fixed patient profile tier that sets the baseline difficulty:

| Profile | Beta responsiveness | Side-effect sensitivity | Recovery rate | Used in |
|---|---|---|---|---|
| `responsive` | High | Low | Fast | easy |
| `balanced` | Medium | Medium | Normal | medium, most expert tasks |
| `refractory` | Low | Medium | Slow | hard, refractory_patient |
| `fragile` | Medium | High | Slow | fragile_patient, medication_interaction |

Patient profiles are fixed per task (not randomly drawn per episode), ensuring that score differences across seeds reflect event variation and noise, not patient-type lottery.

## Grading system

Each episode is graded by a deterministic grader producing a score in `[0.0, 1.0]`. The grader aggregates trajectory-level component scores, applies task-specific weights, then subtracts hard-failure penalties.

### Score components

| Component | What it measures |
|---|---|
| `beta_score` | Mean beta suppression + fraction of steps below `target_beta_arv` |
| `tremor_score` | Mean tremor reduction + fraction of steps below `target_tremor_arv` |
| `force_score` | Mean force preserved relative to `target_force_preserved` |
| `tracking_score` | Motor command accuracy relative to `target_tracking_error` |
| `safety_score` | Side-effect load relative to `max_side_effect_load`; penalizes both mean overload and peak overload |
| `smoothness_score` | Penalizes large parameter jumps between steps |
| `efficiency_score` | Rewards lower amplitude/pulse-width usage (weighted by therapeutic engagement) |
| `terminal_stability_score` | Force, tremor, and tracking accuracy in the final 5 steps |
| `recovery_score` | Improvement from episode start to end (force, tremor, tracking) |

### Grader weights by task

Weights reflect the primary clinical goal of each task:

| Component | easy | medium | hard |
|---|---|---|---|
| `beta_score` | 0.30 | 0.06 | **0.22** |
| `tremor_score` | 0.18 | 0.14 | **0.14** |
| `force_score` | 0.16 | 0.16 | 0.14 |
| `tracking_score` | 0.12 | 0.16 | 0.16 |
| `safety_score` | 0.14 | **0.22** | 0.18 |
| `terminal_stability_score` | 0.05 | 0.08 | **0.08** |
| `smoothness_score` | 0.05 | 0.04 | 0.04 |
| `efficiency_score` | 0.05 | 0.08 | 0.04 |

**Why the hard grader weights changed from the naive design:** An earlier version placed `safety_score` at 0.36 in the hard grader, which accidentally rewarded low-stimulation passive agents — a constant 1.0 mA policy (amp_norm = 0.42) never accumulated side effects, so safety ≈ 0.90 contributed 0.32 of the score alone. Clinically, DBS effectiveness requires *both* adequate suppression *and* safety. The current hard grader places `beta_score` at 0.22 and `tremor_score` at 0.14 — an agent that doesn't suppress pathological beta below 0.21 cannot score well regardless of how safe it was.

### Hard-failure penalties

On top of weighted component scores, each task applies a penalty block for clinically unacceptable behaviors:

| Condition | Penalty |
|---|---|
| `safety_score < 0.20` (any task) | −0.12 |
| `beta_score < 0.30` (hard) | −0.10 |
| `tremor_score < 0.25` (hard) | −0.06 |
| `terminal_stability_score < 0.25` (hard) | −0.08 |
| High mean amplitude + poor efficiency (hard) | −0.08 |
| Poor recovery from off-med crisis (medication_interaction) | −0.12 |
| Amplitude violations during microlesion window (surgical_followup) | −0.20 |
| Zero stimulation during exercise exertion (exercise_bout) | −0.16 |

The final grader score is `clamp(weighted_sum − penalty, 0, 1)`.

## Reward design

The dense per-step reward mirrors the grader weights for each task, so the training signal is aligned with evaluation. For the hard task:

```text
r_t =
    0.14 * force_preserved
  + 0.16 * tracking_accuracy
  + 0.22 * (1 − beta_arv)        ← primary DBS goal
  + 0.14 * (1 − tremor_arv)      ← co-primary goal
  + 0.18 * safety
  + 0.04 * (1 − smoothness_cost)
  + 0.04 * efficiency
  + long_horizon_shaping          ← small terminal-stability bonus in final 25 % of episode
  − 0.08 * constraint_violation
```

The easy and medium reward weights are defined separately (with safety higher in medium, beta higher in easy) to keep the training signal task-appropriate.

## Observation space

| Group | Key fields | Meaning |
|---|---|---|
| Neural state | `beta_arv`, `tremor_arv`, `semg_arv` | Pathological oscillation, tremor envelope, muscle activity proxy |
| Disease state | `disease_severity`, `beta_suppression` | Normalized severity and current suppression level |
| Motor state | `force_amplitude`, `force_preserved`, `effective_motor_output`, `task_error` | Whether the patient can produce useful movement |
| DBS state | `dbs_amplitude_ma`, `dbs_pulse_width_ms`, `dbs_entrainment`, `side_effect_load` | Applied stimulation and remaining safety budget |
| Temporal trends | `beta_trend`, `tremor_trend`, `side_effect_rate`, `recent_dbs_avg_ma` | Short-horizon trend information for closed-loop control |
| Extended state | `gamma_arv`, `medication_phase`, `stim_washout`, `adaptation_state` | Over-stimulation marker, L-DOPA cycle position, wash-in/out |
| Event metadata | `active_events` (in metadata dict) | Which stochastic events are currently active |
| Evaluation | `grader_score`, `episode_success` | Benchmark diagnostics at episode end |

Biomarker signals (`beta_arv`, `tremor_arv`, `semg_arv`, `force_preserved`) carry task-calibrated sensor noise (Gaussian, std 0.025–0.050 depending on task). The latent state used for grading is clean; only what the agent observes is perturbed — mirroring real LFP/EMG recording noise.

## Action space

| Action | Range | Purpose |
|---|---|---|
| `dbs_amplitude` | `0.0` to `5.0` mA (task-capped) | Controls stimulation strength |
| `dbs_pulse_width` | `0.06` to `0.20` ms | Controls stimulation spread and entrainment |
| `motor_command` | `−1.0` to `1.0` | Intended voluntary movement command |

The environment maps `(dbs_amplitude, dbs_pulse_width)` to the calibrated entrainment surface with a one-step lag, then distorts motor output based on current Parkinsonian state. During impedance surge events, `delivered_amp = requested_amp × delivered_amp_mult` — the agent must request higher amplitude to compensate for the hardware fault.

## System architecture

```
Fleming biophysical outputs
        ↓
core/calibration.py  →  CalibratedBrainState (100-step trajectory + DBS sweep)
        ↓
core/events.py       →  EventScheduler (seeded stochastic event timeline)
        ↓
server/parkinsons_Motor_environment.py
  reset():  sample patient profile, build event timeline, init latent state
  step():   resolve schedule overrides → poll events → clip action →
            compute entrainment → update beta/tremor/force/side-effects →
            build dense reward → record trajectory step
        ↓
episode end: graders/dbs_graders.py → deterministic score in [0, 1]
        ↓
FastAPI/OpenEnv server + optional web demo
```

## Repository structure

```text
.
├── README.md
├── run_local_inference.py
├── run_taskwise_inference.py
└── parkinsons_Motor/
    ├── openenv.yaml              # task registry (10 tasks)
    ├── core/
    │   ├── calibration.py        # Fleming data loader + entrainment table
    │   ├── events.py             # EventScheduler, event profiles, schedule_overrides
    │   ├── models.py             # ParkinsonsMotorAction / ParkinsonsMotorObservation
    │   └── patient_profiles.py   # responsive / balanced / refractory / fragile
    ├── tasks/
    │   ├── base.py               # DBSTask dataclass
    │   ├── easy.py / medium.py / hard.py
    │   ├── scenarios.py          # fragile_patient, refractory_patient, personalization_generalization
    │   ├── exercise_bout.py / medication_interaction.py
    │   ├── nocturnal_transition.py / surgical_followup.py
    │   └── registry.py
    ├── graders/
    │   ├── components.py         # beta_score, tremor_score, safety_score, …
    │   ├── rules.py              # hard-failure penalty logic
    │   ├── easy_grader.py / medium_grader.py / hard_grader.py
    │   ├── expert_grader.py / scenario_graders.py
    │   └── dbs_graders.py        # dispatcher + GRADER_REGISTRY
    ├── server/
    │   └── parkinsons_Motor_environment.py
    ├── tests/
    │   ├── smoke_test.py
    │   ├── smoke_scenarios.py
    │   └── test_remote.py
    └── fleming-model-based-brain/ # raw calibration data
```

## Quick start

### 1. Run the local OpenEnv server

```bash
uv run --project parkinsons_Motor server
```

Starts the environment at `http://localhost:8000`. Useful endpoints:

- `http://localhost:8000/docs` — FastAPI/OpenAPI interface
- `http://localhost:8000/viewer` — visual demo

### 2. Configure model credentials

Create or edit `.env` in the repository root:

```env
API_KEY="your-key-here"
API_BASE_URL="https://api.openai.com/v1"
MODEL_NAME="gpt-4o-mini"
```

### 3. Run the baseline inference loop

```bash
uv run --project parkinsons_Motor python run_local_inference.py
```

For per-task logs saved to `outputs/runs/`:

```bash
uv run --project parkinsons_Motor python run_taskwise_inference.py
```

### 4. Use the environment directly in Python

```python
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment
from parkinsons_Motor.core.models import ParkinsonsMotorAction

env = ParkinsonsMotorEnvironment()
obs = env.reset(task_id="hard", seed=42)

for step in range(obs.metadata["episode_steps"]):
    action = ParkinsonsMotorAction(
        motor_command=obs.target_output,
        dbs_amplitude=1.8,
        dbs_pulse_width=0.14,
    )
    obs = env.step(action)
    print(f"step={step} beta={obs.beta_arv:.3f} side_effects={obs.side_effect_load:.3f} reward={obs.reward:.3f}")
    if obs.done:
        print(f"grader_score={obs.grader_score:.4f} success={obs.episode_success}")
        break
```

## Calibration and scientific grounding

The calibrator loads and aligns controller outputs, force traces, and DBS sweep tables from the Fleming data into a shared 100-step timeline. It normalizes signals, derives clinically meaningful fields (`force_preserved`, `disease_severity`, `beta_suppression`), and exposes bilinear interpolation for the DBS entrainment surface.

The online environment is not a direct replay of that trajectory. Each step uses the calibrated trajectory as a *physiological anchor* — it sets the baseline beta and force targets — but the actual state evolves through action-coupled dynamics: the agent's amplitude choices drive entrainment, which suppresses beta, which controls tremor, which recovers force. Add stochastic events on top, and the state space the agent must navigate is genuinely non-trivial.

## What makes this environment interesting for RL

- **Sequential medical control**: actions compound over time; early decisions constrain late options.
- **Partial observability + sensor noise**: the agent sees noisy biomarker readings, not the true latent state.
- **Multi-objective with genuine tension**: beta suppression requires high amplitude; safety limits high amplitude; events shift the optimal operating point unpredictably.
- **Non-linear control surface**: DBS entrainment is not linear in amplitude or pulse width.
- **Temporal horizon effects**: tachyphylaxis means a strategy optimal at step 50 fails at step 100.
- **Fast training loop**: heavy neuroscience simulation is offline; the RL loop replays calibrated dynamics quickly.

## Deployment

```bash
openenv push --namespace your-namespace
```

Hosted demo: https://huggingface.co/spaces/virustechhacks/parkinsons_Motor

## Limits and scope

- This is a benchmark environment, not a clinical device or treatment system.
- The environment is mechanistically grounded but remains a semi-mechanistic simulator, not a full physiological patient model.
- Real-world deployment would require patient-specific sensing, hardware constraints, safety validation, and clinical trials.

## Extended documentation

- [PROBLEM.md](./PROBLEM.md) — clinical framing and motivation
- [ARCHITECTURE.md](./ARCHITECTURE.md) — component-level system design
- [CALIBRATION.md](./CALIBRATION.md) — how Fleming outputs become calibrated environment state
- [REWARD_DESIGN.md](./REWARD_DESIGN.md) — reward logic and anti-gaming analysis
- [STATE_ACTION_SPACE.md](./STATE_ACTION_SPACE.md) — full observation/action semantics
- [TASKS.md](./TASKS.md) — task curriculum and grading details
- [docs/BENCHMARK_GRADE.md](./docs/BENCHMARK_GRADE.md) — benchmark-grade justification
- [docs/JUDGE_PITCH.md](./docs/JUDGE_PITCH.md) — judge-facing pitch summary

## Vision

MotorAssistEnv is a benchmark for adaptive neurostimulation — a place to test whether modern agents can learn to tune a brain implant under realistic clinical trade-offs. If an agent can consistently preserve force, suppress pathological oscillation, navigate medication cycles, and remain within a safety budget across a 150-step refractory episode, it becomes a compelling prototype for future closed-loop neuromodulation systems.
