# MotorAssistEnv: Closed-Loop Deep Brain Stimulation for Parkinson's Disease

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/openenv/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MotorAssistEnv is an OpenEnv-compatible reinforcement learning environment where an agent acts like an adaptive Deep Brain Stimulation (DBS) programmer for a patient with Parkinson's disease. At each 20 ms step, the agent observes pathological brain and motor signals, chooses DBS settings, and tries to preserve motor function while staying within a side-effect budget.

This repository is built around calibrated outputs from the peer-reviewed Fleming et al. (2023) biophysical simulation, so the environment is grounded in real neural and motor dynamics rather than toy physics. The supporting docs add detail, but this README is intended to be the single document a new reader can use to understand the project end to end.

## Why this project matters

Parkinson's disease disrupts the basal ganglia through pathological beta-band synchrony in the subthalamic nucleus (STN). That disruption shows up as tremor, rigidity, and reduced ability to generate reliable voluntary movement. In practice, patients often lose agency over basic tasks such as reaching, holding objects, or writing.

DBS is one of the most effective treatments, but programming it is still largely manual. If stimulation is too weak, pathological oscillations persist and symptoms worsen. If it is too strong, the patient can experience dyskinesia, discomfort, unnecessary battery drain, and other side effects. Settings also drift over time as disease state and patient condition change. That makes DBS programming a strong sequential decision problem:

- Each action changes the next brain state.
- The patient state is non-stationary as tremor builds over the episode.
- The agent only sees partial physiological information, which mirrors real closed-loop DBS hardware.
- The objective is inherently multi-objective: restore function, suppress pathological activity, and remain safe.

MotorAssistEnv frames that challenge as a benchmark for RL agents and LLM-driven agents.

## What the environment models

The environment is backed by outputs from:

> Fleming, J.E., Senneff, S. and Lowery, M.M. (2023), *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*, Journal of Neural Engineering, 20(5), 056029.

Why this model matters:

- It connects brain activity, DBS control, tremor, EMG, and muscle force in one pipeline.
- It produces physically meaningful values rather than arbitrary simulator units.
- It includes a published closed-loop controller, which gives us a strong reference policy.

The packaged source data in `parkinsons_Motor/fleming-model-based-brain/` includes:

- A 100-step DBS-active timeline covering `t=10.02s` to `t=12.00s`
- Controller and observer CSVs for beta activity, tremor, side effects, stimulation, and force
- Large force and sEMG traces used during calibration
- A `12 x 15` DBS parameter sweep that maps amplitude and pulse width to cortical entrainment

## Core idea

At every step, the agent receives a compact summary of the patient's current neural and motor state. It then selects:

- `dbs_amplitude`
- `dbs_pulse_width`
- `motor_command`

Those actions influence the next step through a calibrated entrainment lookup plus online latent dynamics. The current environment is not a simple replay anymore: it tracks patient-specific beta, tremor, force, fatigue, entrainment, and side-effect states, then updates them with task-aware constraints and patient-profile coefficients. Stronger and better-targeted stimulation suppresses beta activity and reduces tremor, but excessive stimulation accumulates side effects, hurts force, and can trigger benchmark penalties.

In plain terms, the agent is learning when to intervene, how strongly to intervene, and how to trade symptom control against stimulation cost.

## System architecture

The project has four layers:

1. **Biophysical data layer**
   Raw simulation outputs from Fleming et al. provide the ground-truth trajectories and the DBS parameter sweep.

2. **Calibration layer**
   [`parkinsons_Motor/core/calibration.py`](./parkinsons_Motor/core/calibration.py) exposes the calibrated brain-state interface used by the environment. It loads the raw files, aligns them to a common 100-step timeline, normalizes signals, computes derived features such as `force_preserved`, and exposes fast lookup/interpolation helpers.

3. **OpenEnv environment layer**
   [`parkinsons_Motor/server/parkinsons_Motor_environment.py`](./parkinsons_Motor/server/parkinsons_Motor_environment.py) implements `reset()` and `step()`, applies the Parkinsonian distortion and DBS effects, tracks trajectory data, and computes episode-end grades.

4. **Serving and visualization layer**
   [`parkinsons_Motor/server/app.py`](./parkinsons_Motor/server/app.py) exposes the FastAPI/OpenEnv server, while `static/myosuite_demo/` provides a separate visual demo so human viewers can see tremor reduction without slowing down RL training.

### Pipeline

```text
Fleming biophysical outputs
        ->
core/calibration.py
        ->
CalibratedBrainState
        ->
OpenEnv environment (reset/step)
        ->
agent actions: amplitude, pulse width, motor command
        ->
one-step-lag DBS entrainment + motor distortion
        ->
dense reward per step + deterministic grader at episode end
        ->
optional web demo / remote inference
```

## Task design

The environment uses three clinically distinct tasks built from the same calibrated episode family. They share the same action/observation schema, but differ in horizon length, safety budget, pass threshold, and what kind of control behavior is rewarded.

| Task ID | Friendly name | Difficulty | Steps | Main goal |
|---|---|---:|---:|---|
| `beta_suppression` | Calm Start | Easy | 30 | Early stabilization under a very tight safety budget |
| `tremor_correction` | Rescue Phase | Medium | 48 | Active tremor rescue as force starts to degrade |
| `full_episode` | Full Episode | Hard | 100 | Long-horizon control with cumulative side effects and recovery pressure |

Plain-language meaning:

- `beta_suppression` / `Calm Start`: the onboarding task
- `tremor_correction` / `Rescue Phase`: the first real adaptive-control task
- `full_episode` / `Full Episode`: the long-horizon benchmark

Expected public ladder shape:

- `beta_suppression`: `no_dbs`, `const_low`, `const_mid`, and `const_high` all fail; `safety_aware` passes all public runs.
- `tremor_correction`: `no_dbs` and all constant policies fail; `safety_aware` is currently marginal, passing `2/4` public runs.
- `full_episode`: passive and constant policies fail; adaptive controllers should remain competitive but not trivial under the long-horizon safety budget.

That ladder shape is intentional. A simple hand-designed controller should solve the onboarding task, stay competitive but imperfect on the rescue task, and still need real pacing discipline on the long-horizon task.

### What the agent must learn across tasks

- Recognize the current disease phase from `beta_arv`, `tremor_arv`, and force-related signals
- Use DBS early enough to slow deterioration rather than only reacting after collapse
- Exploit the non-linear amplitude/pulse-width entrainment surface
- Balance short-term symptom suppression against long-term safety
- Compensate with motor output when the neural state is degraded

## Observation space

At each step the agent observes a clinically grounded state vector. The most important fields are:

| Group | Key fields | Meaning |
|---|---|---|
| Neural state | `beta_arv`, `tremor_arv`, `semg_arv` | Pathological oscillation, tremor envelope, and muscle activity proxy |
| Disease state | `disease_severity`, `beta_suppression` | Normalized severity and current suppression level |
| Motor state | `force_amplitude`, `force_preserved`, `effective_motor_output`, `task_error` | Whether the patient can still produce useful movement |
| DBS state | `dbs_amplitude_ma`, `dbs_pulse_width_ms`, `dbs_entrainment`, `side_effect_load` | What stimulation was applied and how much safety budget remains |
| Temporal summaries | `beta_trend`, `tremor_trend`, `side_effect_rate`, `recent_dbs_avg_ma` | Short-horizon trend information needed for closed-loop control |
| Safety and control | `side_effect_load`, `action_smoothness_cost`, `dbs_constraint_violation` | Whether the current strategy is clinically sustainable |
| Evaluation metadata | `grader_score`, `episode_success`, `sim_time_s` | Benchmark-facing diagnostics exposed at the interface |

Most continuous observation fields are normalized into `[0, 1]`, while force is also exposed in physical units.

## Action space

The action space is continuous and intentionally clinical:

| Action | Range | Purpose |
|---|---|---|
| `dbs_amplitude` | `0.0` to `5.0` mA | Controls stimulation strength |
| `dbs_pulse_width` | `0.06` to `0.20` ms | Controls stimulation spread and entrainment |
| `motor_command` | `-1.0` to `1.0` | Intended voluntary movement command |

The environment then maps `(dbs_amplitude, dbs_pulse_width)` onto the calibrated entrainment surface, applies the effect with a one-step lag, and distorts the motor command based on the current Parkinsonian state.

## Reward design

The dense per-step reward is a weighted combination of motor function, tracking quality, symptom suppression, safety, smoothness, and efficiency:

```text
r_t =
    0.28 * force_preserved
  + 0.20 * tracking_accuracy
  + 0.16 * (1 - beta_arv)
  + 0.12 * (1 - tremor_arv)
  + 0.12 * safety
  + 0.07 * (1 - smoothness_cost)
  + 0.05 * efficiency
  - 0.08 * constraint_violation
```

This is designed to reflect the real treatment objective:

- Force and tracking are rewarded directly because the benchmark is about useful function, not just pretty neural traces.
- Beta and tremor suppression matter, but they are not allowed to dominate safety or motor quality.
- Safety and smoothness make aggressive but clinically unstable controllers unattractive during training.
- Constraint violations are penalized explicitly so policies cannot exploit task clipping.~

### Episode-end grading

Each task is scored by a deterministic grader in `[0.0, 1.0]`. The grader combines:

- `force_score`
- `beta_score`
- `tremor_score`
- `tracking_score`
- `safety_score`
- `smoothness_score`
- `efficiency_score`
- `terminal_stability_score`
- `recovery_score` where relevant

Task weights shift by clinical goal:

- `beta_suppression` rewards gentle early suppression and clean tracking
- `tremor_correction` rewards active rescue instead of passive waiting
- `full_episode` rewards long-horizon stability and punishes weak terminal control

The grader also includes hard-failure logic for unsafe stimulation, repeated task-envelope violation, non-treatment on rescue tasks, and poor terminal quality on the hard task.

The dense reward tracks the same main objectives as the final grader and now adds small phase-aware shaping for recovery and late-episode stability on the longer tasks. That keeps training signal informative without collapsing the benchmark into a purely shaped objective.

## Calibration and scientific grounding

Calibration is what turns raw neuroscience output into a usable RL environment.

The calibrator:

- Loads and aligns controller outputs, force traces, and DBS sweep tables
- Converts signals to a shared 100-step timeline
- Computes normalization bounds from the actual recorded data
- Derives clinically meaningful fields such as `force_preserved`, `disease_severity`, and `beta_suppression`
- Exposes a bilinear interpolation query for DBS entrainment

Important grounding choices:

- The episode is anchored to calibrated Fleming traces, but online transitions are now action-coupled and stateful rather than direct replay
- Force is normalized against the healthy baseline from the source data
- The reward and grader are tied to clinically interpretable quantities
- Patient variation is explicit through profile-dependent responsiveness, fatigue, recovery, and side-effect sensitivity
- Difficulty comes from both the source trajectory and the calibrated closed-loop dynamics layered on top

## What makes this environment interesting for RL

- **Sequential medical control**: actions compound over time and must be planned.
- **Partial observability**: the agent does not see full neural state, only realistic clinical measurements.
- **Multi-objective optimization**: symptom suppression and side-effect management are always in tension.
- **Non-linear control surface**: the DBS entrainment table is not linear in amplitude or pulse width.
- **Fast training loop**: heavy neuroscience simulation is offline; the RL environment replays calibrated dynamics quickly.

## Repository structure

```text
.
|- README.md
|- BENCHMARK_GRADE.md
|- PROBLEM.md
|- ARCHITECTURE.md
|- CALIBRATION.md
|- REWARD_DESIGN.md
|- STATE_ACTION_SPACE.md
|- TASKS.md
|- CONTEXT.md
|- outputs/
|  |- README.md
|  |- benchmark/
|  |- search/
|  |- runs/
|  `- analysis/
|- docs/
|  |- BENCHMARK_GRADE.md
|  `- JUDGE_PITCH.md
|- run_local_inference.py
`- parkinsons_Motor/
   |- client.py
   |- inference.py
   |- core/
   |- evaluation/
   |- fleming-model-based-brain/
   |- tasks/
   |- graders/
   |- tests/
   |- server/
   `- static/myosuite_demo/
```

## Quick start

### 1. Run the local OpenEnv server

From the repository root:

```bash
uv run --project parkinsons_Motor server
```

This starts the environment on `http://localhost:8000`.

Useful endpoints:

- `http://localhost:8000/docs` for the FastAPI/OpenAPI interface
- `http://localhost:8000/viewer` for the MyoSuite-based visual demo

### 2. Configure model credentials

Create or edit `.env` in the repository root:

```env
API_KEY="your-key-here"
API_BASE_URL="https://api.openai.com/v1"
MODEL_NAME="gpt-4o-mini"
```

The inference script also accepts `OPENAI_API_KEY`, `HF_TOKEN`, `LLM_PROVIDER`, `OPENAI_MODEL`, and `HF_MODEL_NAME`. If an OpenAI key is present, it will default to the OpenAI API unless you explicitly force another provider.

### 3. Run the baseline inference loop

```bash
uv run --project parkinsons_Motor python run_local_inference.py
```

That script connects to the local server, runs the agent loop, and prints OpenEnv-style step and score logs.

If you want a lower-pressure version that runs tasks one by one with longer pauses between runs:

```bash
uv run --project parkinsons_Motor python run_taskwise_inference.py
```

That helper saves separate per-task logs and a final summary into `outputs/runs/`.

### 4. Use the environment directly in Python

```python
from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv

with ParkinsonsMotorEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id="tremor_correction")
    print(result.observation.tremor_arv)

    action = ParkinsonsMotorAction(
        motor_command=0.4,
        dbs_amplitude=1.5,
        dbs_pulse_width=0.13,
    )
    result = env.step(action)
    print(result.observation.force_preserved, result.reward)
```

## Deployment

The project includes OpenEnv and Hugging Face Space configuration under `parkinsons_Motor/`.

To push the environment with OpenEnv tooling:

```bash
openenv push --namespace your-namespace
```

The hosted demo currently lives at:

- https://huggingface.co/spaces/virustechhacks/parkinsons_Motor

## Limits and current scope

- This is a benchmark environment, not a clinical device or treatment recommendation system.
- The environment is grounded and action-coupled, but it is still a semi-mechanistic benchmark rather than a full physiological patient simulator.
- The 3D visualization is for communication and demos, not for the training loop itself.
- Real-world deployment would require patient-specific sensing, hardware constraints, safety validation, and clinical trials.

## Extended documentation

The README is the main project document. These files are extensions if you want more depth on a specific area:

- [PROBLEM.md](./PROBLEM.md): full clinical framing and motivation
- [ARCHITECTURE.md](./ARCHITECTURE.md): component-level system design
- [CALIBRATION.md](./CALIBRATION.md): how raw Fleming outputs become calibrated environment state
- [REWARD_DESIGN.md](./REWARD_DESIGN.md): detailed reward and anti-hacking logic
- [STATE_ACTION_SPACE.md](./STATE_ACTION_SPACE.md): full observation/action semantics
- [TASKS.md](./TASKS.md): task curriculum and grading thresholds
- [CONTEXT.md](./CONTEXT.md): project narrative and hackathon context
- [RESEARCH_AND_REFERENCES.md](./RESEARCH_AND_REFERENCES.md): source lineage, repo influences, and citation guidance
- [docs/BENCHMARK_GRADE.md](./docs/BENCHMARK_GRADE.md): why the environment qualifies as benchmark-grade
- [docs/JUDGE_PITCH.md](./docs/JUDGE_PITCH.md): judge-facing pitch summary
- [outputs/README.md](./outputs/README.md): guide to generated benchmark artifacts
- [parkinsons_Motor/README.md](./parkinsons_Motor/README.md): package/server quick reference

## Vision

MotorAssistEnv is a benchmark for adaptive neurostimulation: a place to test whether modern agents can learn to tune a brain implant under realistic trade-offs. If an agent can consistently preserve force, suppress pathological oscillation, and remain within a safety budget here, it becomes a compelling prototype for future closed-loop neuromodulation systems.
        
