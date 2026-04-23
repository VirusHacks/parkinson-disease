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

The source data in `fleming-model-based-brain/` includes:

- A 100-step DBS-active timeline covering `t=10.02s` to `t=12.00s`
- Controller and observer CSVs for beta activity, tremor, side effects, stimulation, and force
- Large force and sEMG traces used during calibration
- A `12 x 15` DBS parameter sweep that maps amplitude and pulse width to cortical entrainment

## Core idea

At every step, the agent receives a compact summary of the patient's current neural and motor state. It then selects:

- `dbs_amplitude`
- `dbs_pulse_width`
- `motor_command`

Those actions influence the next step through a calibrated DBS entrainment model with a one-step lag. Stronger and better-targeted stimulation suppresses beta activity and reduces tremor, but excessive stimulation consumes the safety budget and hurts long-horizon performance.

In plain terms, the agent is learning when to intervene, how strongly to intervene, and how to trade symptom control against stimulation cost.

## System architecture

The project has four layers:

1. **Biophysical data layer**
   Raw simulation outputs from Fleming et al. provide the ground-truth trajectories and the DBS parameter sweep.

2. **Calibration layer**
   [`parkinsons_Motor/brain_calibrator.py`](./parkinsons_Motor/brain_calibrator.py) loads the raw files, aligns them to a common 100-step timeline, normalizes signals, computes derived features such as `force_preserved`, and exposes fast lookup/interpolation helpers.

3. **OpenEnv environment layer**
   [`parkinsons_Motor/server/parkinsons_Motor_environment.py`](./parkinsons_Motor/server/parkinsons_Motor_environment.py) implements `reset()` and `step()`, applies the Parkinsonian distortion and DBS effects, tracks trajectory data, and computes episode-end grades.

4. **Serving and visualization layer**
   [`parkinsons_Motor/server/app.py`](./parkinsons_Motor/server/app.py) exposes the FastAPI/OpenEnv server, while `static/myosuite_demo/` provides a separate visual demo so human viewers can see tremor reduction without slowing down RL training.

### Pipeline

```text
Fleming biophysical outputs
        ->
brain_calibrator.py
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

The environment uses three tasks built from the same 100-step clinical trajectory. They share the same action/observation design and reward logic; difficulty comes from longer horizons, stricter trade-offs, and more severe tremor progression.

| Task | Difficulty | Steps | Main goal |
|---|---|---:|---|
| `beta_suppression` | Easy | 20 | Suppress early beta activity without overspending the side-effect budget |
| `tremor_correction` | Medium | 50 | Dynamically respond as tremor rises and force begins to fall |
| `full_episode` | Hard | 100 | Handle the full progression and optimize the full clinical trade-off |

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
| Transparency | `scheduler_class`, `beta_ctrl_error`, `sim_time_s` | Helpful controller and timeline context |

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

The dense per-step reward is:

```text
r_t = 0.50 * force_preserved_t
    + 0.30 * (1 - task_error_t)
    + 0.15 * dbs_entrainment_t
    - 0.005 * dbs_amplitude_t
```

This is designed to reflect the real treatment objective:

- `force_preserved` is the primary signal because restored function is the real outcome.
- `1 - task_error` rewards effective movement rather than pure stimulation.
- `dbs_entrainment` rewards meaningful suppression of the pathological circuit.
- The amplitude penalty discourages brute-force stimulation.

### Episode-end grading

Each task is scored by a deterministic grader in `[0.0, 1.0]`. The grader combines:

- `force_score`
- `beta_score`
- `side_effect_score`
- `amplitude_efficiency`
- `final_state_bonus` where relevant

Task weights shift by clinical goal:

- `beta_suppression` prioritizes keeping beta below threshold
- `tremor_correction` prioritizes preserving force while tremor ramps
- `full_episode` balances force, beta, safety, and efficiency over a long horizon

This split between dense training reward and strict final grading makes the environment trainable while keeping evaluation objective and hard to game.

## Calibration and scientific grounding

Calibration is what turns raw neuroscience output into a usable RL environment.

The calibrator:

- Loads and aligns controller outputs, force traces, and DBS sweep tables
- Converts signals to a shared 100-step timeline
- Computes normalization bounds from the actual recorded data
- Derives clinically meaningful fields such as `force_preserved`, `disease_severity`, and `beta_suppression`
- Exposes a bilinear interpolation query for DBS entrainment

Important grounding choices:

- Observation values come from calibrated simulation outputs, not invented generative dynamics
- Force is normalized against the healthy baseline from the source data
- The reward and grader are tied to clinically interpretable quantities
- Difficulty comes from the source trajectory itself, not artificial randomization

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
|- PROBLEM.md
|- ARCHITECTURE.md
|- CALIBRATION.md
|- REWARD_DESIGN.md
|- STATE_ACTION_SPACE.md
|- TASKS.md
|- CONTEXT.md
|- run_local_inference.py
|- fleming-model-based-brain/
`- parkinsons_Motor/
   |- brain_calibrator.py
   |- client.py
   |- inference.py
   |- models.py
   |- tasks/
   |- graders/
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

The inference script also accepts `HF_TOKEN` or `OPENAI_API_KEY`.

### 3. Run the baseline inference loop

```bash
uv run --project parkinsons_Motor python run_local_inference.py
```

That script connects to the local server, runs the agent loop, and prints OpenEnv-style step and score logs.

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
- The online environment replays calibrated source dynamics; it does not simulate arbitrary new patient physiology.
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
- [docs/JUDGE_PITCH.md](./docs/JUDGE_PITCH.md): judge-facing pitch summary
- [parkinsons_Motor/README.md](./parkinsons_Motor/README.md): package/server quick reference

## Vision

MotorAssistEnv is a benchmark for adaptive neurostimulation: a place to test whether modern agents can learn to tune a brain implant under realistic trade-offs. If an agent can consistently preserve force, suppress pathological oscillation, and remain within a safety budget here, it becomes a compelling prototype for future closed-loop neuromodulation systems.
