# MotorAssistEnv

### Can a language model learn to program a brain implant — and give a Parkinson's patient back the ability to hold a cup?

We gave an LLM a noisy LFP electrode, three knobs on a Medtronic-class deep brain stimulator, and a patient whose basal ganglia is collapsing in real time. No medical training. No few-shot examples of "good DBS programming." Just raw biomarkers every 20 ms and a clock that does not stop.

By step 36 of an easy episode, an off-the-shelf 72B model can suppress pathological beta below the clinical target, hold a 1.0–1.2 mA dose, and pass the smoke-test grader. By step 100 of a hard episode, the same model is fighting tachyphylaxis it has never seen, an off-medication crisis it cannot predict, and a refractory patient whose brain stops responding to the moves that worked thirty seconds ago. It does not pass — but it scores **0.59** where a constant-dose policy scores **0.23**, which is the entire point of the benchmark.

**MotorAssistEnv** is an OpenEnv-compatible reinforcement-learning environment that turns adaptive Deep Brain Stimulation (aDBS) into a benchmark for sequential medical control. It is calibrated against the peer-reviewed Fleming et al. (2023) biophysical simulation of the Parkinsonian motor circuit, exposes 10 clinically grounded tasks across a strictly monotonic difficulty ladder, and is built so that the agents we train against it have to actually treat the patient — not game the metric.

> **OpenEnv Hackathon (India 2026)** — Theme #3.1 (World Modeling / Professional Tasks) primary, with Theme #2 (Long-Horizon Planning) and Theme #4 (Self-Improvement) as secondary. Built on **OpenEnv v0.2+**. Trained with **HF TRL GRPO + Unsloth 4-bit + LoRA** in Colab and Kaggle. Deployed on **Hugging Face Spaces**.

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor) [![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/meta-pytorch/OpenEnv) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Clinical Motivation

> **Full reference:** [`PROBLEM.md`](./PROBLEM.md) — full clinical framing, why DBS programming is a sequential decision problem, the case for RL over hand-tuned PID, and what specifically the Fleming biophysical model contributes to grounding this benchmark.

Parkinson's disease breaks the basal ganglia circuit. The clearest signature is pathological beta-band synchrony in the subthalamic nucleus, and that one oscillation is responsible for most of what patients actually feel — tremor, rigidity, and the slow loss of voluntary movement that turns ordinary tasks into hard ones. Holding a cup. Reaching for a door handle. Signing your name. Over 1 million patients in the US live with this; over 50,000 of them have a DBS implant. The hardware works. The settings are usually wrong, and the patient suffers in the gap between programming visits — typically 3–6 months apart, sometimes longer. Adaptive (closed-loop) DBS is where the field is going. **MotorAssistEnv frames the policy a closed-loop device would need to learn as an RL benchmark.**

---

## The story: from a 30-number observation to a 150-step crisis

### Act 1 — The cold start

Step 1 of an `easy` episode. The agent receives a `ParkinsonsMotorObservation`: 30 floats describing the patient. `beta_arv = 0.51`, `tremor_arv = 0.18`, `force_preserved = 0.62`, `medication_phase = 0.4`, `side_effect_load = 0.00`. It has never seen any of these labels before.

It chooses an action — `dbs_amplitude`, `dbs_pulse_width`, `motor_command` — and the patient's brain answers. The 12×15 entrainment table from Fleming maps `(amp, pw)` to a fraction of cortical axons recruited, with a one-step neural delay. Beta state updates. Tremor follows. Force responds. Twenty milliseconds later, the agent gets a new observation.

Untrained models flail here. They write `dbs_amplitude = 5.0`, instantly burn 30% of the safety budget, and the grader's hard-failure rules fire `−0.20`. Or they write `dbs_amplitude = 0.0`, the `therapeutic_engagement` gate collapses, and the efficiency reward they tried to claim evaporates. Either way, the score lands near the constant-dose floor.

### Act 2 — First suppression

Around step 8 the agent figures out that holding `dbs_amplitude ≈ 1.0 mA` at `pulse_width ≈ 0.13 ms` and `frequency ≈ 130 Hz` reliably drives `beta_arv` from ~0.5 down to ~0.3. The dense reward stops being negative. `force_preserved` climbs from 0.62 to 0.78. The grader's per-step `1 − beta_arv` term contributes 0.22 to every step's reward, the agent gets the gradient signal it needed, and the policy locks in.

This is where Qwen2.5-72B sits at the end of an easy episode in our most recent local-inference run — `beta_score = 0.95`, `tracking_score = 0.95`, `safety_score = 1.00`, `force_score = 0.52`, **overall = 0.7951**, threshold 0.55, **PASS**.

### Act 3 — The patient deteriorates

Step 100 of a hard episode. Same agent, same policy. But the `long_horizon` event profile fires `tachyphylaxis` (82% probability, 12–20 steps), and the entrainment lookup the agent has been relying on quietly multiplies by `max(0.40, 1.0 − 2.0 × intensity)`. The same 1.4 mA that was suppressing beta to 0.3 a minute ago is now leaving it at 0.5. Then `off_med_crisis` (75% probability) adds `+0.28 × intensity` to the underlying beta drive every step. The patient is sliding off the cliff in a way the agent's recent history does not predict.

A constant-dose policy scores **0.23–0.36** on `hard` across all seeds. A reactive 72B agent without medical priors scores **0.59**. The threshold is **0.68**. Nobody has passed `hard` reliably yet — and that's correct. Hard was calibrated to be the gap.

### Act 4 — The environment improved itself

The most useful thing that happened during this build was not the agent improving. It was the environment improving — because of what the agent failed at.

Our first `hard` grader weighted `safety_score` at **0.36**. A constant 1.0 mA policy never accumulated side effects (`amp_norm = 0.42`, well below the threshold), so safety contributed 0.32 of the score on its own. The grader was rewarding doing nothing. We rebuilt it: safety down to **0.18**, `beta_score` up to **0.22**, `tremor_score` to **0.14**, plus a hard-failure floor of `−0.10` if `beta_score < 0.30`. Constant policies dropped from 0.55 to 0.27 on hard overnight.

Same story for the dense reward. Same story for the smoothness term, which originally let an inactive agent farm `1 − smoothness_cost = 1.0` for free. Same story for the efficiency term, which now gates through `therapeutic_engagement` so an agent that produces no clinical effect cannot collect efficiency credit either.

We treat that loop — *agent failures expose environment bugs, environment patches expose new agent failures* — as a feature, not a bug log. It is the recursive-self-improvement story Theme #4 is asking for, applied to the platform itself.

---

## Problem Statements Addressed

### Primary: Theme #3.1 — World Modeling / Professional Tasks

The agent interacts with a real biophysical model — not a generative approximation, not a polynomial fit. Every observation is a transformation of calibrated outputs from a Hodgkin-Huxley simulation of ~400 neurons across cortex, STN, GPe, GPi, thalamus, spinal motoneurons, and a Hill-type muscle model with ~5M synaptic connections (Fleming et al. 2023, *J Neural Eng* 20(5):056029).

- **Real tool, real units.** `dbs_amplitude` is in mA. `force_amplitude` is in mN. `beta_arv` is normalised to a real pre-DBS baseline measured in µV. The 12×15 entrainment table the agent's actions resolve against was published in the source paper.
- **Persistent partial-observability world.** The agent reads what an implanted Medtronic RC+S, Abbott Infinity, or Boston Scientific Vercise actually exposes — LFP power bands, impedance, plus surface EMG. Spike trains, future trajectory, and per-episode noise factors are all hidden.
- **Multi-step causal workflow.** Actions at step *t* produce a brain state at step *t+1*. Adaptation accumulates. Side effects accumulate. Battery drain accumulates. There is no shortcut path from observation to grader.

### Secondary: Theme #2 — Super Long-Horizon Planning

- 150-step `hard` episodes with sparse terminal grading on top of dense per-step shaping.
- Tachyphylaxis means a strategy optimal at step 50 fails at step 100 — the agent has to detect distribution shift inside its own context window.
- Per-episode noise and per-episode random target output mean trajectory memorisation does not transfer.

### Secondary: Theme #4 — Self-Improvement

- 10-task curriculum spans easy → medium → hard → 7 expert/scenario tasks, each weighted differently by the grader.
- Constant-baseline calibration locks the difficulty ordering into the threshold itself, so the curriculum is empirically falsifiable, not aspirational.
- The grader's weight tables and hard-failure rules were rewritten by watching what the agent did — the platform co-evolved with the model under training.

### Partner sub-theme: Snorkel AI — Simulated Experts-in-the-Loop

Each task ships its own grader (`easy_grader.py`, `medium_grader.py`, `hard_grader.py`, `expert_grader.py`, `scenario_graders.py`) with weights that mirror what a different clinical specialist would score on. `surgical_followup` weights safety at **0.30** because the post-implant programming nurse cares about the microlesion window above all else. `exercise_bout` weights force and tracking at **0.22 / 0.22** because the rehab specialist cares about whether the patient could lift the weight. The agent is being scored by a different "expert" each task.

---

## The loop

```
┌──────────────────────────────────────────────────────────────────────┐
│                       MotorAssistEnv — one episode                  │
│                                                                     │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐        │
│  │  Calibrated  │───►│   Latent state  │───►│    Sensor    │        │
│  │  Fleming sim │    │  beta, tremor,  │    │   (Gaussian  │        │
│  │  (offline)   │    │  force, side-   │    │    noise)    │        │
│  └──────────────┘    │  effects, adapt │    └──────┬───────┘        │
│                      │  (hidden)       │           │ obs (30 floats)│
│                      └────────▲────────┘           ▼                │
│                               │                                      │
│  ┌──────────────┐             │              ┌──────────────┐        │
│  │  Stochastic  │   patches   │   action     │     Agent     │       │
│  │   events     ├─────────────┤◄─────────────┤  (Qwen / LLM  │       │
│  │ (tachyphyl., │             │ (amp, pw,    │   + LoRA via  │       │
│  │  off-med,    │             │  freq, motor)│     GRPO)     │       │
│  │  motor surge)│             │              └──────────────┘        │
│  └──────────────┘             ▼                                      │
│                      ┌─────────────────┐                             │
│                      │  Dense reward   │── 9-axis grader ───►  score │
│                      │  (per 20 ms)    │     (episode end)           │
│                      └─────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

### How it works, step by step

1. `reset(task_id, seed)` selects one of 10 tasks, picks the patient profile, builds a seeded event timeline, and loads the calibrated Fleming trajectory as a physiological anchor.
2. The agent receives a `ParkinsonsMotorObservation`: 30 fields covering brain biomarkers, motor function, device state, 5-step trends, medication phase, and episode metadata.
3. The agent emits a `ParkinsonsMotorAction`: `dbs_amplitude` (0–5 mA, task-capped), `dbs_pulse_width` (0.06–0.20 ms), `dbs_frequency` (60–185 Hz), `motor_command` (−1 to +1).
4. The environment resolves any active event overrides, clips the action to the task ceiling, and computes entrainment via bilinear interpolation on the 12×15 Fleming sweep (one-step lag).
5. Latent beta and tremor states evolve via `0.45·prev + 0.55·target`, where `target = baseline − 0.82·entrainment·responsiveness + event_pressures`. Tremor uses a 0.50× entrainment coefficient because real STN-DBS suppresses beta more cleanly than tremor (Tinkhauser 2017).
6. Effective motor output is `motor_command × (1 − 0.52·β)·(1 − 0.30·T)·(1 − 0.10·SE) + noise` — Parkinson's disease physically distorts what the patient is trying to do, and only DBS can clear the channel.
7. The dense reward fires (formula below). The trajectory step is recorded.
8. On termination, the deterministic 9-component grader produces a score in `[0, 1]`. `episode_success = score ≥ task.success_threshold`.

---

## What makes this different

- **Not a simulator we wrote.** The dynamics anchor is calibrated outputs from a published biophysical model. Force amplitudes are in real millinewtons (`~59,752 mN` healthy baseline). Beta values are normalised to real measured pre-DBS LFP. The 12×15 entrainment surface the agent navigates was published in *J Neural Eng* in 2023.
- **Latent vs sensed split, by construction.** The grader reads `self._beta_state` directly. The agent reads `_add_sensor_noise(self._beta_state)` via `_make_obs`. There is no API path from agent action to grader buffer. The DeepMind grasping-task class of attack — *fool the camera so the evaluator can't tell* — is structurally impossible here.
- **Multi-component reward with a therapeutic-engagement gate.** Nine clinical axes, weighted differently per task, plus a `therapeutic_engagement = 0.40·force + 0.30·beta + 0.30·tremor` multiplier on the efficiency component. An agent that produces no clinical effect collects no efficiency credit, even though it's barely using the battery.
- **Empirically calibrated difficulty.** A constant 1.0 mA / 0.13 ms / 130 Hz baseline was run 5 seeds × 3 tasks. Easy passes 5/5 (0.72–0.80). Medium fails 5/5 (0.47–0.52). Hard fails 5/5 (0.23–0.36). The thresholds (0.55 / 0.52 / 0.68) are placed exactly above what doing-nothing achieves, so passing genuinely means the agent reasoned.
- **15-attack adversarial audit, table-traceable.** Every named attack policy (do nothing, max amp forever, constant 1.0 mA, fool the sensor, farm recovery score, front-load good steps, hard-code the target, push frequency to 185 Hz for entrainment) has a documented blocker. Twelve of fifteen attacks are fully blocked; three are bounded well below the success threshold.
- **No hardcoded knowledge in the prompt.** The agent prompt does not contain "1.0 mA is therapeutic," does not contain "130 Hz peaks beta suppression," does not contain a labelled action template. It must learn the entire `(amp, pw, freq) → (β, T, force, SE)` mapping from reward, exactly as a clinician's intuition is built from outcomes.
- **The platform improved alongside the model.** Three iterations of the hard grader. Two iterations of the dense reward. One full rewrite of the efficiency term after we caught it rewarding zero-stim coasting. The reward design doc has a 15-row anti-gaming table because we kept losing rounds against our own agents.

---

## Task Suite

> **Full reference:** [`TASKS.md`](./TASKS.md) — every one of the 10 tasks in detail, with parameter tables, event schedules, success-threshold calibration, and the constant-baseline data that proves the difficulty ordering is real.

Each task uses the same `ParkinsonsMotorAction` / `ParkinsonsMotorObservation` schema but varies episode length, patient profile, event probabilities, biomarker targets, safety budget, and grader weights.

| Task | Steps | Patient | What's hard about it | Threshold |
|---|---:|---|---|---:|
| `easy` — Calm Start | 36 | Responsive | Smoke test. Almost any reasonable amp passes. | 0.55 |
| `medium` — Rescue Phase | 60 | Balanced | Mid-episode deterioration wave (55%); rescue without dyskinesia. | 0.52 |
| `hard` — Full Episode | **150** | Refractory | Tachyphylaxis (82%) + off-med crisis (75%) + 2× dyskinesia spikes (80%) + 2× motor surges (65%). 4 overlapping crises in a 150-step window. | **0.68** |
| `fragile_patient` | 64 | Fragile (1.40× side-effect sensitivity) | Therapeutic window of ~0.3–0.8 mA. Half the usable range of medium. | 0.44 |
| `refractory_patient` | 120 | Refractory | Brute-force amp doesn't scale. Pulsed strategy required. | 0.46 |
| `personalization_generalization` | 90 | All 4 mixed per episode | Read the profile from reset metadata, adapt from step 1. | 0.50 |
| `exercise_bout` | 70 | Balanced | Motor surge fires with certainty in first 30%; ramp DBS up then taper. | 0.55 |
| `medication_interaction` | 100 | Fragile | Off-med crisis at 30–45%, dyskinesia rebound at 65% probability when next dose lands. | 0.50 |
| `nocturnal_transition` | 150 | Balanced | Awake → wind-down → sleep. Targets tighten 20–35% in sleep phase. | 0.55 |
| `surgical_followup` | 120 | Balanced | 0.6 mA hard ceiling for first 25% (microlesion window). −0.20 if violated. | 0.50 |

The 10 tasks fall into three deliberate buckets:

| Bucket | Tasks | What it measures |
|---|---|---|
| **Difficulty ladder** | `easy`, `medium`, `hard` | Climb a difficulty curve under the same patient/event family |
| **Patient-class generalisation** | `fragile_patient`, `refractory_patient`, `personalization_generalization` | Transfer a control strategy across physiologically different patients |
| **Clinical scenario reasoning** | `exercise_bout`, `medication_interaction`, `nocturnal_transition`, `surgical_followup` | Recognise *what kind of situation this is* and pick a strategy fit for it |

An agent that passes all three buckets has not memorised a dose; it has internalised what DBS is *for*.

---

## State and Action Space

> **Full reference:** [`STATE_ACTION_SPACE.md`](./STATE_ACTION_SPACE.md) — every observation field with its formula, the four control knobs in real units, the four-equation dynamics block (entrainment, beta/tremor suppression, motor distortion, frequency-side-effect coupling), and the explicit list of what the agent *cannot* see and why.

### Action — 4 continuous knobs

| Field | Range | What it does |
|---|---|---|
| `dbs_amplitude` | 0.0–5.0 mA (task-capped) | Stimulation current. Primary driver of axon recruitment and beta suppression. |
| `dbs_pulse_width` | 0.06–0.20 ms | Pulse duration. Wider pulses recruit axons beyond the target nucleus volume. |
| `dbs_frequency` | 60–185 Hz | Pulse-train rate. Beta suppression peaks at ~130 Hz (Kühn 2008). |
| `motor_command` | −1.0 to +1.0 | What the patient is *trying* to do. Distorted by `(1−0.52β)(1−0.30T)(1−0.10SE)`. |

Charge per second `= amp × pw × freq` drives both therapeutic effect and battery drain. The `(amp, pw)` pair indexes the 12×15 entrainment table; frequency then scales the result via `_freq_beta_factor(freq)` peaking at 130 Hz.

### Observation — 30 fields, grouped

| Group | Sample fields | What it tells the agent |
|---|---|---|
| Brain biomarkers | `beta_arv`, `tremor_arv`, `semg_arv`, `gamma_arv` | "How bad is the Parkinson's right now? Are we over-stimulating?" |
| Motor function | `force_amplitude`, `force_preserved`, `target_output`, `effective_motor_output`, `task_error`, `tracking_accuracy` | "Can the patient actually move?" |
| Disease summary | `disease_severity`, `beta_suppression` | Convenient aggregates |
| 5-step trends | `beta_trend`, `tremor_trend`, `side_effect_rate` | "Is the situation improving or sliding?" |
| Device state | `dbs_amplitude_ma`, `dbs_pulse_width_ms`, `dbs_entrainment`, `recent_dbs_avg_ma`, `side_effect_load`, `action_smoothness_cost`, `dbs_constraint_violation`, `stim_washout`, `battery_drain_rate` | "What is the implant currently doing? How close are we to the danger zone?" |
| Patient context | `medication_phase` | Where in the L-DOPA cycle the patient is |
| Episode metadata | `sim_time_s`, `task_id`, `grader_score`, `episode_success` | Bookkeeping; grader_score is `−1.0` until terminal |

### What the agent does NOT see

| Hidden | Why |
|---|---|
| Ground-truth Fleming optimal DBS settings | Would trivialise the problem |
| Per-episode noise factors (`ep_beta_noise`, `ep_tremor_noise`, `ep_force_noise`, `ep_semg_noise`) | Would let agent pre-compute optimal compensation |
| Future event schedule | Would convert reactive control into open-loop replay |
| Underlying patient profile parameters | Would let agent hard-code a per-patient policy |
| Raw STN spike trains | Not available via chronic LFP recording in real life |

The rule: the observation is rich enough to act, too poor to plan around the grader.

---

## Reward Design

> **Full reference:** [`REWARD_DESIGN.md`](./REWARD_DESIGN.md) — full per-step reward formula, all per-task weight tables, the 9-component grader spec, every hard-failure rule per task, the 15-attack adversarial audit, and the AI-safety design principles the reward is engineered against.

### Per-step dense reward (hard task, 20 ms cadence)

```
r_t = 0.14 · force_preserved
    + 0.16 · tracking_accuracy
    + 0.22 · (1 − beta_arv)            ← primary DBS objective
    + 0.14 · (1 − tremor_arv)          ← co-primary objective
    + 0.18 · safety                    ← safety = clamp(1 − SE/SE_max)
    + 0.04 · (1 − smoothness_cost)
    + 0.04 · efficiency                ← gated by therapeutic_engagement
    + shaping_t                        ← terminal-stability bonus, final 25%
    − 0.08 · constraint_violation
```

Weights vary per task. Easy emphasises beta (the agent must *learn* DBS suppresses pathology). Medium emphasises safety + recovery (rescue without dyskinesia). Hard balances beta + tremor + safety so low-stim coasting cannot game the safety term.

| Component weight | `easy` | `medium` | `hard` |
|---|---:|---:|---:|
| beta | **0.30** | 0.06 | **0.22** |
| tremor | 0.18 | 0.14 | 0.14 |
| force | 0.16 | 0.16 | 0.14 |
| tracking | 0.12 | 0.16 | 0.16 |
| safety | 0.14 | **0.22** | 0.18 |
| smoothness | 0.05 | 0.04 | 0.04 |
| efficiency | 0.05 | 0.08 | 0.04 |

### Episode-end grader — 9 components

| Component | What it captures |
|---|---|
| `beta_score` | `0.55·weighted_mean(1−β) + 0.45·frac(β ≤ target)` — suppression depth + time in range (Tinkhauser 2017) |
| `tremor_score` | Same dual metric for tremor |
| `force_score` | `weighted_mean(force) / target_force`, early steps weighted ~1.35× (Limousin 1995) |
| `tracking_score` | `0.45·(1 − err/target_err) + 0.55·tracking_acc` |
| `safety_score` | Side-effect overload: `clamp(1 − (0.45·mean + 0.35·peak + 0.20·violation)·1.8)` (Swann 2018) |
| `efficiency_score` | `(0.65·(1−amp/max) + 0.35·(1−pw)) × therapeutic_engagement` — gated to block zero-DBS gaming |
| `smoothness_score` | `1 − mean(smoothness_cost)` (Velisar 2019) |
| `terminal_stability_score` | `0.45·force + 0.30·(1−T) + 0.25·(1−err)` on **last 5 steps only** — blocks front-loading |
| `recovery_score` | First 6 vs last 8 steps — did the agent rescue, or just survive? |

`final_score = clamp(weighted_sum − hard_failure_penalties, 0.0, 1.0)`.

### Hard-failure penalties (clinical floors)

These exist because a smooth weighted sum can hide clinically unacceptable outcomes. Each penalty fires in addition to the score it already cost the agent.

| Condition | Penalty | What it models |
|---|---:|---|
| `safety_score < 0.20` (any task) | −0.12 | Sustained dyskinesia risk |
| `beta_score < 0.30` (hard) | −0.10 | DBS providing no measurable suppression |
| `tremor_score < 0.25` (hard) | −0.06 | Active tremor uncontrolled |
| `terminal_stability_score < 0.25` (hard) | −0.08 | Patient unstable at episode end |
| `tracking_score < 0.55` (exercise_bout) | −0.14 | Patient cannot execute movement during exertion |
| Zero stim during exertion (exercise_bout) | −0.16 | Withholding therapy when the patient needs it most |
| Amp violation in microlesion window (surgical_followup) | **−0.20** | Stimulation-induced dyskinesia in a fresh implant |
| Poor recovery from off-med crisis (medication_interaction) | −0.12 | Failure to rescue during an L-DOPA trough |

---

## Anti-Hacking and Reward Integrity

> **Full reference:** [`REWARD_DESIGN.md` §6–7](./REWARD_DESIGN.md) — the complete 15-attack adversarial audit (do nothing, max amp forever, sensor fooling, recovery farming, front-loading, frequency entrainment exploit, motor-command zero, and more), each one traced through the dense reward and grader, with the specific block that defeats it.

The penalty table catches obvious specification gaming. The defenses below catch the cleverer attacks. They live in the surrounding scaffolding, not in the score formula.

| Mechanism | What it does | What it blocks |
|---|---|---|
| Latent vs sensed split | Grader reads `_beta_state` directly; agent reads it through Gaussian sensor noise | Sensor-fooling (the DeepMind grasping example) |
| Per-episode noise factors | `ep_beta_noise`, `ep_tremor_noise`, `ep_force_noise`, `ep_semg_noise` resampled every reset | Open-loop trajectory memorisation |
| Stochastic events as physics | `tachyphylaxis` modifies entrainment; `off_med_crisis` modifies beta drive — not just the score | Pre-baked schedules; ignoring events |
| FastAPI sandbox | Only entry is `step(action)` with Pydantic-validated `ParkinsonsMotorAction` | Reward tampering; reading grader weights |
| Therapeutic-engagement gate on efficiency | Efficiency credit multiplied by `0.40·force + 0.30·beta + 0.30·tremor` | "Do nothing forever for perfect efficiency" |
| Calibrated difficulty thresholds | Constant baseline scores empirically below threshold on medium and hard | Trivial constant-policy passes |
| Random `target_output` per episode | Tracking target re-rolled at reset | Hard-coding a target value |

For the full 15-attack walkthrough — including the `motor_command = 0` attack, the 185 Hz entrainment exploit, and the front-loading attack — see [REWARD_DESIGN.md §6](./REWARD_DESIGN.md).

---

## Results

### Difficulty calibration — constant 1.0 mA / 0.13 ms / 130 Hz baseline (5 seeds per task)

| Task | Min | Max | Threshold | Passes? |
|---|---:|---:|---:|:---:|
| `easy` | 0.724 | 0.804 | 0.55 | **Always** |
| `medium` | 0.470 | 0.518 | 0.52 | Never |
| `hard` | 0.231 | 0.358 | 0.68 | Never |

Strict monotonic ordering across all seeds. The thresholds are not aspirational — they sit exactly above what a passive policy can reach. *Passing means the agent did something better than nothing.*

### Off-the-shelf LLM baseline — Qwen2.5-72B-Instruct, no training, full episodes

Most recent local-inference run, single seed each task, full episode lengths:

| Task | Steps | Score | Threshold | Result | Mean amp (mA) | Notable events |
|---|---:|---:|---:|:---:|---:|---|
| `easy` | 36/36 | **0.7951** | 0.55 | **PASS** | 1.078 | none |
| `medium` | 50/60 | 0.4525 | 0.52 | FAIL | 1.102 | `dyskinesia_spike@26-32` |
| `hard` | 30/150 | 0.5944 | 0.68 | FAIL | 1.095 | `tachyphylaxis@65-83`, `off_med_crisis@96-106`, `motor_surge@50-59`, `motor_surge@72-77` |

Component breakdown for the easy pass:

| Component | Value |
|---|---:|
| `beta_score` | **0.948** |
| `tremor_score` | 0.773 |
| `force_score` | 0.523 |
| `tracking_score` | 0.949 |
| `safety_score` | 1.000 |
| `smoothness_score` | 0.959 |
| `efficiency_score` | 0.209 |
| `terminal_stability_score` | 0.339 |
| `therapeutic_engagement` | 0.725 |
| `hard_failure_penalty` | 0.040 |
| `pre_penalty_score` | 0.835 |
| **`overall_score`** | **0.7951** |

What this tells us:

1. **Beta suppression is learnable from the dense reward alone.** The 72B model with no medical priors hits `beta_score = 0.95` on easy.
2. **The hard task is genuinely hard.** A policy that scores 0.59 on hard would be considered competitive — the threshold is set above that on purpose, so an agent has to *pace its safety budget across overlapping crises* to pass.
3. **The improvement gap is real.** On hard, the LLM scores **0.59** vs the constant baseline's **0.23–0.36**. That ~0.25–0.36 absolute gain is the headroom GRPO is being asked to close.

### Trained policy (in progress)

GRPO + Unsloth 4-bit + LoRA on Qwen3-4B, training notebook at [`dbs_sft_grpo_colab.ipynb`](./dbs_sft_grpo_colab.ipynb). Training is currently running on H100; reward curves and the trained-vs-baseline comparison table will land here on completion. Live training metrics are written to `outputs/runs/`.

---

## Quick start

### 1. Run the OpenEnv server locally

```bash
uv run --project parkinsons_Motor server
```

Starts at `http://localhost:8000`. Useful endpoints:
- `http://localhost:8000/docs` — FastAPI / OpenAPI
- `http://localhost:8000/viewer` — visual demo (MyoSuite-style 3D arm that smooths as beta drops)

### 2. Talk to the environment from Python

```python
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment
from parkinsons_Motor.core.models import ParkinsonsMotorAction

env = ParkinsonsMotorEnvironment()
obs = env.reset(task_id="hard", seed=42)

for step in range(obs.metadata["episode_steps"]):
    action = ParkinsonsMotorAction(
        dbs_amplitude=1.4,
        dbs_pulse_width=0.13,
        dbs_frequency=130.0,
        motor_command=obs.target_output,
    )
    obs = env.step(action)
    print(f"step={step:3d} beta={obs.beta_arv:.3f} SE={obs.side_effect_load:.3f} reward={obs.reward:+.3f}")
    if obs.done:
        print(f"score={obs.grader_score:.4f} pass={obs.episode_success}")
        break
```

### 3. Run an LLM agent against the live server

```bash
# Configure model credentials in .env at repo root:
# API_KEY=...
# API_BASE_URL=https://api.openai.com/v1
# MODEL_NAME=gpt-4o-mini

uv run python run_local_inference.py            # easy / medium / hard
uv run python run_taskwise_inference.py         # all 10 tasks, per-task logs to outputs/runs/
```

### 4. Train with GRPO in Colab

Open [`dbs_sft_grpo_colab.ipynb`](./dbs_sft_grpo_colab.ipynb) (HF TRL `GRPOTrainer` + Unsloth 4-bit + LoRA on Qwen3-4B). The notebook connects to the deployed HF Space, collects rollouts via `trl.experimental.openenv.generate_rollout_completions`, and pushes checkpoints to the HF Hub.

### 5. Deploy to Hugging Face Spaces

```bash
openenv push --namespace your-namespace
```

Live demo: **https://huggingface.co/spaces/virustechhacks/parkinsons_Motor**

---

## Architecture

```
═══════════════════════════════════════════════════════════════════════════
                  DATA LAYER  (offline, fixed, peer-reviewed)
═══════════════════════════════════════════════════════════════════════════
  parkinsons_Motor/fleming-model-based-brain/
  ├── Model_Results/                ← 34 controller/observer CSVs
  │   ├── beta_ARV_Observer_values.csv
  │   ├── tremor_ARV_Observer_values.csv
  │   ├── Force_amplitude_values.csv          (6.7M samples)
  │   └── ...
  ├── Collaterals_Entrained_values.txt        ← 12 × 15 DBS sweep
  ├── DBS_Amplitude_Interpolation_values.txt
  └── DBS_Pulse_Width_Interpolation_values.txt

═══════════════════════════════════════════════════════════════════════════
                  CALIBRATION  (runs once, cached)
═══════════════════════════════════════════════════════════════════════════
  parkinsons_Motor/core/calibration.py
  └── calibrate() → CalibratedBrainState
      ├── 100-step physiological anchor (t = 10.02–12.00 s)
      ├── Normalisation bounds from data maxima
      ├── Pre-DBS baselines (β, tremor, force)
      └── Bilinear DBS entrainment surface

═══════════════════════════════════════════════════════════════════════════
                  ENVIRONMENT  (online, per-episode)
═══════════════════════════════════════════════════════════════════════════
  parkinsons_Motor/
  ├── openenv.yaml              ← spec_version 1, 10-task registry
  ├── core/
  │   ├── models.py             ← Pydantic Action / Observation
  │   ├── events.py             ← seeded EventScheduler + 6 event types
  │   └── patient_profiles.py   ← responsive / balanced / fragile / refractory
  ├── tasks/                    ← 10 task specs (frozen dataclasses)
  ├── graders/                  ← 9-component dispatcher + per-task graders
  └── server/
      ├── parkinsons_Motor_environment.py    ← reset / step / state
      └── app.py                              ← FastAPI + WebSocket

═══════════════════════════════════════════════════════════════════════════
                  AGENT + TRAINING
═══════════════════════════════════════════════════════════════════════════
  dbs_sft_grpo_colab.ipynb      ← TRL GRPO + Unsloth 4-bit + LoRA on Qwen3-4B
  run_local_inference.py         ← LLM-vs-environment loop (single seed)
  run_taskwise_inference.py      ← LLM × all 10 tasks, per-task logging
```

---

## Repository structure

```
.
├── README.md                                ← you are here
├── PROBLEM.md                               ← full clinical framing
├── STATE_ACTION_SPACE.md                    ← every observation field, every action
├── REWARD_DESIGN.md                         ← reward formulas + 15-attack audit
├── TASKS.md                                 ← all 10 tasks with parameters
├── RESEARCH_AND_REFERENCES.md               ← scientific lineage + 25-citation bibliography
├── dbs_sft_grpo_colab.ipynb                 ← Colab GRPO training notebook
├── run_local_inference.py
├── run_taskwise_inference.py
├── outputs/runs/                            ← per-run reports (json + md)
└── parkinsons_Motor/
    ├── openenv.yaml
    ├── client.py                            ← OpenEnv WebSocket client
    ├── core/
    │   ├── calibration.py
    │   ├── events.py
    │   ├── models.py
    │   └── patient_profiles.py
    ├── tasks/
    │   ├── easy.py / medium.py / hard.py
    │   ├── scenarios.py
    │   ├── exercise_bout.py / medication_interaction.py
    │   ├── nocturnal_transition.py / surgical_followup.py
    │   └── registry.py
    ├── graders/
    │   ├── components.py
    │   ├── rules.py
    │   ├── easy_grader.py / medium_grader.py / hard_grader.py
    │   ├── expert_grader.py / scenario_graders.py
    │   └── dbs_graders.py
    ├── server/
    │   ├── parkinsons_Motor_environment.py
    │   └── app.py
    ├── tests/
    │   ├── smoke_test.py
    │   ├── smoke_scenarios.py
    │   └── test_remote.py
    └── fleming-model-based-brain/           ← raw calibration data
```

---

## Key design decisions

1. **Real biophysical anchor over a generative model.** Calibrating from Fleming et al. (2023) gives every observation a defensible physical unit and every reward weight a defensible clinical citation. A reviewer can audit the chain `parameter → simulation output → environment field → reward term → published paper`.

2. **Latent vs sensed split.** Only path the agent has to influence the grader is to actually treat the patient. Sensor-fooling is structurally impossible. This single design choice does more anti-hacking work than every penalty rule combined.

3. **Empirically calibrated difficulty thresholds.** Thresholds were placed exactly above the constant 1.0 mA baseline's measured score, not pulled from intuition. Difficulty ordering is a *measurement*, not a label.

4. **Nine-component grader with task-varying weights.** Lets `surgical_followup` weight safety at 0.30 and `exercise_bout` weight force at 0.22 without rewriting the grader. Each task is graded by a different "specialist."

5. **Therapeutic-engagement gate on efficiency.** Closes the most attractive shortcut in the original design — *do nothing for perfect efficiency*. Efficiency credit now requires actually producing clinical effect.

6. **Patient-profile variation as a first-class axis.** A policy that hard-codes for one patient type fails `personalization_generalization`, which alone passes the threshold only for agents that read the profile from reset metadata.

7. **GRPO over PPO.** Sparse terminal grader on top of dense per-step shaping is exactly the regime GRPO's group-relative advantages are best at. No value function to mis-train, multi-rollout variance produces stable gradients.

8. **The platform co-evolved with the model.** Three iterations of the hard grader. Two of the dense reward. One full rewrite of the efficiency term. The hard-failure penalty table grew row by row each time we caught an agent gaming a previous version. We treat that loop as the *Theme #4 self-improvement story applied at the platform level*.

---

## Documentation Map

This README is meant to be read end-to-end without opening anything else. Each section above links inline to the deep-dive document it summarises. Collected here for convenience:

| Document | What's in it | Section it expands |
|---|---|---|
| [`PROBLEM.md`](./PROBLEM.md) | Full clinical framing, RL problem structure, why not PID, Fleming model deep-dive, real-world deployment trajectory | *Clinical Motivation* |
| [`STATE_ACTION_SPACE.md`](./STATE_ACTION_SPACE.md) | All 30 observation fields with formulas, 4 control knobs in real units, the 4-equation dynamics block, latent-vs-sensed split, patient profiles | *State and Action Space* |
| [`REWARD_DESIGN.md`](./REWARD_DESIGN.md) | Per-step reward formula, per-task weight tables, 9-component grader spec, hard-failure rules per task, 15-attack adversarial audit, integrity defenses | *Reward Design*, *Anti-Hacking and Reward Integrity* |
| [`TASKS.md`](./TASKS.md) | All 10 tasks with parameters, event schedules, success-threshold calibration, the constant-baseline difficulty proof | *Task Suite* |
| [`RESEARCH_AND_REFERENCES.md`](./RESEARCH_AND_REFERENCES.md) | 25-source annotated bibliography, scientific lineage, AI-safety influences, canonical citation block, per-source repo map | *Scientific Grounding and Citations* |
| [`dbs_sft_grpo_colab.ipynb`](./dbs_sft_grpo_colab.ipynb) | Runnable training notebook — HF TRL `GRPOTrainer` + Unsloth 4-bit + LoRA on Qwen3-4B, connects to the live HF Space | *Quick start §4*, *Results* |
| [`outputs/runs/`](./outputs/runs/) | Per-run JSON + Markdown reports from `run_local_inference.py` and `run_taskwise_inference.py` | *Results* |

---

## Scientific Grounding and Citations

> **Full reference:** [`RESEARCH_AND_REFERENCES.md`](./RESEARCH_AND_REFERENCES.md) — annotated bibliography of all 25 sources, the Fleming model lineage explained in depth, prior art on RL-for-DBS, MyoSuite framing, AI-safety influences on the reward design, and a per-source map of *which file in this repo each citation grounds*.

**Computational modeling and biophysical source**
- Fleming, J. E., Senneff, S., & Lowery, M. M. (2023). *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*. **J Neural Eng** 20(5), 056029. https://doi.org/10.1088/1741-2552/acfbfa
- Fleming, J. E., Dunn, E., & Lowery, M. M. (2020). *Simulation of closed-loop DBS control schemes for suppression of pathological beta oscillations*. **Front Neurosci** 14, 166. https://doi.org/10.3389/fnins.2020.00166

**Clinical literature behind reward terms**
- Limousin, P. et al. (1995). *Lancet* 345(8942):91–95.
- Deuschl, G. et al. (2006). *NEJM* 355(9):896–908.
- Kühn, A. A. et al. (2008). *NeuroImage* 36(2):379–387.
- Little, S. et al. (2013). *Ann Neurol* 74(3):449–457.
- Priori, A. et al. (2013). *Exp Neurol* 245:77–86.
- Rosa, M. et al. (2015). *Mov Disord* 30(7):1003–1005.
- Little, S. et al. (2016). *Mov Disord* 31(8):1336–1341.
- Tinkhauser, G. et al. (2017). *Brain* 140(11):2968–2981.
- Swann, N. C. et al. (2018). *J Neural Eng* 15(4):046006.
- Velisar, A. et al. (2019). *Brain Stimul* 12(4):868–876.

**Prior art — RL for DBS**
- Krylov, D. et al. (2020). *Reinforcement Learning Framework for Deep Brain Stimulation Study*. **IJCAI-20**. https://doi.org/10.24963/ijcai.2020/394

**Methodology — reward design and AI safety**
- Krakovna, V. et al. (2020). *Specification gaming: the flip side of AI ingenuity*. **DeepMind blog**. https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/

**Frameworks**
- meta-pytorch / **OpenEnv**. https://github.com/meta-pytorch/OpenEnv
- MyoHub / **MyoSuite** (visualisation framing). https://github.com/MyoHub/myosuite

A complete annotated bibliography (25 sources) lives in [`RESEARCH_AND_REFERENCES.md`](./RESEARCH_AND_REFERENCES.md).

---

## Limits and scope

- This is a **benchmark environment**, not a clinical device or a treatment system.
- The dynamics are mechanistically grounded but remain a semi-mechanistic simulator, not a full physiological patient model.
- Real-world deployment to an implanted device would require patient-specific sensing calibration, hardware-in-the-loop validation, and clinical trials governed by FDA/CE pathways.

What it *is*: the most clinically grounded RL-for-DBS environment in the OpenEnv ecosystem, calibrated against peer-reviewed biophysics, with a 10-task curriculum, a 9-component grader, and an empirically falsifiable difficulty ordering — built so that a policy that scores well on it would be a credible prototype for the inference firmware on a next-generation adaptive DBS implant.

---

## License

MIT. See [`LICENSE`](./LICENSE).
