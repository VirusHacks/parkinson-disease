# Tasks Design — MotorAssistEnv (DBS Parkinson's Environment)

## Overview

The three tasks are slices of the same real 100-step biophysical simulation, ordered by the clinical difficulty of the DBS programming challenge they present. Difficulty increases because the simulation's tremor dynamics are naturally progressive — early steps are calm, late steps are near-catastrophic.

All tasks share the same observation space, action space, and reward architecture. Only the episode window and grading thresholds change.

---

## Task 1 — `beta_suppression` (Easy, 20 steps)

**Clinical scenario:**  
The patient has just been connected to their DBS device. Beta oscillation is near its pre-DBS peak (β_arv ≈ 0.74) but tremor has not yet built up significantly (tremor_arv ≈ 0.01–0.17). Muscle force is still at ~93% of healthy. The window of the simulation covered is t=10.02–10.40 s.

**Agent's job:**  
Act as a DBS programmer who must find a stimulation setting that suppresses the beta oscillation below the clinical threshold (β_arv < 0.20) before tremor builds. This is the most forgiving window — a well-chosen single amplitude setting is sufficient. The agent must keep DBS ≤ 1.0 mA to avoid immediately exhausting the side-effect budget.

**Grader weights:**
| Component | Weight | Meaning |
|---|---|---|
| Beta suppression | 0.50 | Primary: did beta stay below threshold? |
| Force preserved | 0.25 | Secondary: motor function not sacrificed |
| Side-effect penalty | 0.15 | Safety: stayed within budget |
| Amplitude efficiency | 0.10 | Bonus: minimum effective amplitude |

**Success threshold:** grader_score ≥ 0.60  
**Episode length:** 20 steps (400 ms of simulated brain time)

---

## Task 2 — `tremor_correction` (Medium, 50 steps)

**Clinical scenario:**  
The episode covers the period when tremor is actively building (t=10.02–11.00 s). Tremor amplitude rises from 0.01 to 0.80 normalised. Muscle force declines from ~93% to ~30% of healthy. The DBS controller must react dynamically: when beta spikes, amplitude must increase; when side-effect load climbs, amplitude must be reduced.

**Agent's job:**  
Dynamically balance DBS amplitude (0–2.0 mA) and pulse width across 50 steps. The agent must prevent force from dropping below 35% of healthy while managing the cumulative side-effect budget (≤ 0.50). Unlike Task 1, a fixed setting will fail because the brain state is non-stationary.

**Grader weights:**
| Component | Weight | Meaning |
|---|---|---|
| Force preserved | 0.50 | Primary: keep motor function above threshold |
| Beta suppression | 0.25 | Secondary: target the oscillation driver |
| Side-effect penalty | 0.15 | Safety: avoid cumulative stimulation overload |
| Final state bonus | 0.10 | Reward for not collapsing at episode end |

**Success threshold:** grader_score ≥ 0.55  
**Episode length:** 50 steps (1 second of simulated brain time)

---

## Task 3 — `full_episode` (Hard, 100 steps)

**Clinical scenario:**  
The full 100-step simulation (t=10.02–12.00 s, 2 seconds of brain time). By step 80+, tremor is near-maximum (0.99 normalised) and force has collapsed to ~4% of healthy. The DBS controller must find a policy that uses aggressive stimulation early to slow tremor progression, then sustains force through the severe late-episode phase — all while keeping the cumulative side-effect load below 0.70.

**Agent's job:**  
This is the clinical optimisation problem in its full form. The agent must learn a dynamic policy across 100 steps that maximises cumulative muscle force preservation. Fixed settings fail — the agent must adapt as the brain state changes dramatically across the episode. This mirrors the real challenge of programming a Parkinson's patient's DBS device for day-to-day motor function.

**Grader weights:**
| Component | Weight | Meaning |
|---|---|---|
| Force preserved | 0.40 | Primary: sustained motor function |
| Beta suppression | 0.20 | Oscillation suppression |
| Side-effect penalty | 0.20 | Safety over a long horizon |
| Amplitude efficiency | 0.10 | Appropriate amplitude use |
| Final state bonus | 0.10 | Episode didn't collapse at end |

**Success threshold:** grader_score ≥ 0.50  
**Episode length:** 100 steps (2 seconds of simulated brain time)

---

## Task Comparison

| Property | beta_suppression | tremor_correction | full_episode |
|---|---|---|---|
| Difficulty | Easy | Medium | Hard |
| Steps | 20 | 50 | 100 |
| Brain time covered | 400 ms | 1000 ms | 2000 ms |
| Max DBS allowed | 1.0 mA | 2.0 mA | 3.0 mA |
| Key challenge | Find correct amplitude | Dynamic response | Long-horizon policy |
| Naive agent score | ~0.43 | ~0.88 | ~0.76 |
| Success threshold | 0.60 | 0.55 | 0.50 |
| Naive agent succeeds? | ❌ | ✅ | ✅ |

> The naive agent (constant 1.0 mA / 0.13 ms) fails Easy but passes Medium/Hard — demonstrating that Task 1 genuinely requires the agent to _not_ overpower the stimulation in a short window.

---

## Curriculum Strategy

Tasks are designed to be run in order during training:

1. **Phase 1 (bootstrap):** `beta_suppression` — teaches the agent that beta suppression matters and that low amplitude can be effective
2. **Phase 2 (dynamic):** `tremor_correction` — teaches the agent to react to changing brain state
3. **Phase 3 (full problem):** `full_episode` — the agent must apply everything it learned to the full clinical scenario

During inference, all 3 tasks are run independently and scored. The mean score across tasks represents the agent's overall clinical competence.

---

## Evaluation Metrics per Task

Each episode reports:
- `grader_score` — deterministic float in [0.0, 1.0]
- `episode_success` — boolean, True if score ≥ task threshold
- `force_preserved` at episode end — direct motor function measure
- `beta_arv` trajectory — how well the oscillation was suppressed
- `side_effect_load` at episode end — cumulative stimulation safety
- Per-step `reward` — dense signal for RL training