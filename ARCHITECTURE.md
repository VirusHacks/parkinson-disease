# Architecture: MotorAssistEnv (DBS Parkinson's Environment)

## 1. Overview

MotorAssistEnv is structured as a **medical BCI (Brain-Computer Interface) reinforcement learning system**. It cleanly separates:
- Real-world biophysical brain simulation (offline)
- OpenEnv RL task and state management (online)
- Neurological grading (deterministic evaluation)
- LLM/RL Agent learning (policy optimization)
- 3D Visualisation (frontend representation)

The goal is a **verifiable, scientifically grounded RL benchmark** that simulates real clinical DBS programming without subjective or "toy" scoring mechanisms.

---

## 2. High-Level Architecture

```
+-------------------------------------------------------------+
|                 OPENENV RL SYSTEM                           |
+-------------------------------------------------------------+

    +-------------------+
    | Biophysical Data  |
    | (Fleming model)   |
    +---------+---------+
              | (CSV timelines + tables)
              v
    +-------------------+
    | Brain Calibrator  |
    | (State + Lookups) |
    +---------+---------+
              |
              v
    +---------------------------+       +-------------------+
    | MotorAssist Environment   | ----> | 3D Visualisation  |
    | (OpenEnv step/reset API)  |       | (MyoSuite Demo)   |
    +---------+-----------------+       +-------------------+
              |
              ^ (brain state)
              |
              v (DBS Action)
    +---------------------------+
    | Agent (LLM / PPO / GRPO)  |
    | learns DBS tuning policy  |
    +---------+-----------------+
              |
              v
    +---------------------------+
    | Grader / Verifier System  |
    | (Deterministic 0.0 - 1.0) |
    +---------------------------+

```

---

## 3. Core Components

### 3.1 Biophysical Data Layer (Ground Truth)
The environment does not invent physics. It uses peer-reviewed biophysical simulation data from Fleming et al. (2023). 
- **What it provides:** 100-step timelines of beta oscillation, tremor amplitude, and raw muscle force.
- **Why it matters:** Agents train on real physiological dynamics, including non-stationary tremor build-up.

### 3.2 Calibration Core (`core/calibration.py`)
Loads the raw `.csv` and `.txt` files into memory during environment initialization through the calibration interface used by the environment.
- Converts raw signals to normalized `[0, 1]` ranges.
- Provides a fast bilinear interpolation function to map the agent's `(amplitude, pulse_width)` to a cortical `entrainment` fraction.

### 3.3 OpenEnv API (`server/parkinsons_Motor_environment.py`)
Provides the standard `step()` and `reset()` interface.
- **State Transition:** Applies the agent's DBS action to suppress the exact *next step's* pathological brain signals (1-step clinical lag).
- **Sub-task Management:** Uses the task registry under `tasks/` to expose three benchmark scenarios with different horizons, safety envelopes, and grading emphasis.

### 3.4 Grader System (`graders/`)
At the end of an episode, returns a fixed `[0.0, 1.0]` score.
- **No LLM-as-Judge.** The grader is 100% deterministic math.
- Split into:
  - `graders/components.py` for reusable score components
  - `graders/rules.py` for hard-failure logic
  - `graders/dbs_graders.py` for task-level deterministic score assembly

### 3.5 Evaluation and Test Harness (`evaluation/`, `tests/`)
The benchmark-facing evaluation code is separated from the environment core.
- `evaluation/eval_suite.py` generates the official public and held-out reports in `outputs/benchmark/`
- `evaluation/tremor_policy_search.py` generates the rescue-policy sweep in `outputs/search/`
- `tests/smoke_test.py` and `tests/test_remote.py` keep quick validation code out of the package root

### 3.6 3D Visualisation (`static/myosuite_demo/`)
A completely separate frontend to showcase the AI's impact visually.
- Reads `tremor_arv` from the OpenEnv backend.
- Applies proportional jitter to a 3D musculoskeletal arm.
- *Crucial detail:* The RL agent is not slowed down by rendering 3D physics during training.

---

## 4. Key Design Decisions

### 4.1 Dense vs. Sparse Reward
The environment returns a **dense per-step reward** during interaction (`reward=0.68`). However, success is judged by a **sparse grader score** at the end of the episode (`grader_score=0.92`). This allows for smooth gradient updates while preserving objective benchmark evaluation.

### 4.2 Why Not Just Use MyoSuite for RL?
MyoSuite calculates computationally heavy musculoskeletal physics. Training an RL agent inside it takes days. Since we already have the ground-truth `muscle_force` calculated by the Fleming neuroscience simulation, we use the raw math for the RL backend (instantaneous steps) and use MyoSuite solely as a visual "puppet" for human demonstrations.

### 4.3 Anti-Hacking Mechanisms
- The simulation timeline advances deterministically. An agent cannot "pause" time.
- The `side_effect_load` budget ensures the agent cannot simply max out DBS amplitude to force a perfect muscle-force score - over-stimulation depletes the safety budget and the grader penalises it heavily.

---

## 5. Metrics Tracked

- `force_preserved`: Primary proxy for patient mobility.
- `beta_arv`: The pathological driver the agent must suppress.
- `side_effect_load`: The penalty boundary.
- `grader_score`: The objective Hackathon validation proxy.

---

## 6. Codebase Layout

```text
parkinsons_Motor/
|- core/
|  |- calibration.py
|  |- models.py
|  `- patient_profiles.py
|- tasks/
|  |- base.py
|  |- scenarios.py
|  `- registry.py
|- graders/
|  |- components.py
|  |- rules.py
|  `- dbs_graders.py
|- evaluation/
|  |- eval_suite.py
|  `- tremor_policy_search.py
|- tests/
|  |- smoke_test.py
|  `- test_remote.py
|- server/
`- static/
```

This split keeps the repo presentable:
- `core/` contains benchmark logic primitives
- `tasks/` contains scenario definitions and lookup
- `graders/` contains deterministic evaluation math
- `evaluation/` contains benchmark/report generation code
- `tests/` contains validation entrypoints
