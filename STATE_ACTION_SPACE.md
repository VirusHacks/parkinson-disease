# State and Action Space — MotorAssistEnv (DBS Parkinson's Environment)

## Overview

The observation and action spaces are defined by the clinical reality of closed-loop DBS. Every field corresponds to a real physiological or device measurement — nothing is synthetic or abstract.

---

## Observation Space (What the Agent Sees)

The agent observes a 12-field state vector at each 20 ms timestep. All fields are normalised to [0, 1] unless stated otherwise.

### Neural State (Brain Signals)

| Field | Type | Range | Clinical meaning |
|---|---|---|---|
| `beta_arv` | float | [0, 1] | STN beta oscillation amplitude. 0 = fully suppressed, 1 = peak Parkinson's pathology. Pre-DBS baseline ≈ 0.78. |
| `tremor_arv` | float | [0, 1] | Tremor amplitude envelope. Grows from ~0.01 to ~0.99 across the episode. |
| `semg_arv` | float | [0, 1] | Surface EMG envelope. Mirrors the beta activity in the motor chain. |

### Disease State (Derived)

| Field | Type | Range | Clinical meaning |
|---|---|---|---|
| `disease_severity` | float | [0, 1] | Normalised tremor ARV — a direct proxy for how severe the Parkinson's state is right now. |
| `beta_suppression` | float | [0, 1] | How much DBS has reduced beta from its peak. 0 = no suppression, 1 = fully suppressed. |

### Motor Output (Muscle Function)

| Field | Type | Range | Clinical meaning |
|---|---|---|---|
| `force_amplitude` | float | [0, ∞) mN | Raw simulated muscle force. Healthy baseline = 59,752 mN. |
| `force_preserved` | float | [0, 1] | Fraction of healthy motor force the patient currently produces. 1.0 = fully healthy. |
| `effective_motor_output` | float | [-1, 1] | The agent's motor command after Parkinsonian distortion (tremor + beta interference). |
| `task_error` | float | [0, 2] | Distance between intended and actual motor output: `|target - effective|`. |

### DBS Device State

| Field | Type | Range | Clinical meaning |
|---|---|---|---|
| `dbs_amplitude_ma` | float | [0, 5] mA | DBS current delivered at the previous step. |
| `dbs_pulse_width_ms` | float | [0.06, 0.20] ms | DBS pulse width delivered at the previous step. |
| `dbs_entrainment` | float | [0, 1] | Fraction of cortical collateral axons entrained by DBS. Derived from the 12×15 parameter sweep table. |
| `side_effect_load` | float | [0, 1] | Cumulative DBS side-effect proxy. High values = patient at risk of dyskinesia or discomfort. |

### Controller Internals (Transparency)

| Field | Type | Range | Clinical meaning |
|---|---|---|---|
| `scheduler_class` | int | {0, 1} | Which sub-controller is active: 0 = tremor controller, 1 = beta controller (active 99% of the time). |
| `beta_ctrl_error` | float | (−∞, ∞) | Beta controller tracking error from the ground-truth PID. Positive = undershot. |
| `sim_time_s` | float | [10.02, 12.00] | Simulation time in seconds. |

### Task & Grader Fields

| Field | Type | Notes |
|---|---|---|
| `task_id` | str | Active task: `beta_suppression` / `tremor_correction` / `full_episode` |
| `grader_score` | float | Final grader score [0, 1]. Value is -1.0 until episode end. |
| `episode_success` | bool | True if grader_score ≥ task success_threshold. |

---

## Action Space (What the Agent Does)

The agent outputs 3 continuous values at each step:

| Field | Type | Range | Clinical meaning |
|---|---|---|---|
| `dbs_amplitude` | float | [0.0, 5.0] mA | DBS stimulation amplitude. 0 = off. Ground-truth optimal: 0.49–0.63 mA for the base simulation. Full suppression requires ≥ 2.0 mA. |
| `dbs_pulse_width` | float | [0.06, 0.20] ms | DBS pulse width. Controls spatial spread of stimulation. Wider pulse → more cortical entrainment. |
| `motor_command` | float | [-1.0, 1.0] | Intended voluntary motor output (e.g., the signal to hold a cup, reach for a door). |
| `task_id` | str | optional | Passed only on `reset()` to select which clinical task to run. Ignored during `step()`. |

---

## Physical Model — How Action Becomes Observation

### DBS Entrainment (One-Step Lag)

When the agent specifies `(dbs_amplitude, dbs_pulse_width)`, the environment bilinearly interpolates the 12×15 entrainment lookup table (derived from the Fleming et al. DBS parameter sweep):

```
entrainment = bilinear_interp(
    dbs_entrainment_table[12×15],
    amplitude_axis[12],
    pulse_width_axis[15],
    dbs_amplitude, dbs_pulse_width
)
```

This entrainment value is then applied to the **next** step's brain state (one-step clinical lag), suppressing:
- `beta_arv` by factor `(1 − entrainment)`
- `tremor_arv` by factor `(1 − 0.6 × entrainment)`
- `force_preserved` boosted by factor `(1 + 0.4 × entrainment)`

### Motor Distortion

The patient's Parkinsonian brain distorts the motor command before it reaches the muscles:

```
effective = motor_command
            × (1 − 0.55 × beta_arv_effective)
            × (1 − 0.30 × tremor_arv_effective)
            + N(0, 0.12 × tremor_arv_effective)
```

Where `beta_arv_effective` and `tremor_arv_effective` are already modulated by the current DBS entrainment from the previous step.

---

## Partial Observability

The agent does **not** observe:
- The ground-truth DBS settings used by the Fleming model (available in metadata for debugging only)
- Raw neural spike trains from individual STN neurons
- The patient's intended target movement (only the normalised `target_output` context is available via `task_error`)

This partial observability is clinically realistic: real closed-loop DBS systems measure LFP signals but not individual neuron activity.

---

## Design Rationale

| Choice | Justification |
|---|---|
| Continuous action space | Matches clinical DBS programmers who tune exact amplitude/pulse-width values |
| One-step DBS lag | Reflects real hardware response delay of DBS implants |
| Force preserved as primary signal | Clinicians care about functional motor output, not just the electrical measurement |
| Side-effect load in observation | The agent must manage the trade-off between suppression and patient safety |
| Normalised observations | Allows the agent to transfer across different disease severity levels |