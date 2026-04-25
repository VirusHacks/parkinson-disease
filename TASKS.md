# Task Specifications — MotorAssistEnv

## Overview

MotorAssistEnv defines 10 tasks across four difficulty tiers. All share the same observation space, action space, and environment dynamics. What changes per task is the clinical scenario:

- **Episode length** — from 36 steps (titration) to 150 steps (full-session management)
- **Patient profile** — fixed per task; determines baseline entrainment, side-effect sensitivity, recovery rate
- **Event profile** — seeded stochastic events wired into the physics (tachyphylaxis, off-med crisis, dyskinesia, etc.)
- **Biomarker targets** — how tightly beta, tremor, and tracking must be controlled
- **Safety budget** — maximum cumulative side-effect load before safety score collapses
- **Grader weights** — which clinical objectives matter most for this scenario
- **Success threshold** — minimum grader score to declare the episode a success

**Difficulty ordering guarantee:** The constant 1.0 mA baseline scores easy (0.72–0.80) > medium (0.47–0.52) > hard (0.23–0.36) across all seeds. Hard never passes its threshold with any constant policy.

---

## Public Tasks

### `easy` — Calm Start

**Clinical scenario:** Early DBS titration in a newly programmed patient. Beta activity is rising but tremor is not yet severe. The agent's job is to establish a therapeutic amplitude early and hold it cleanly — the onboarding clinical task.

**Patient:** `responsive` — high beta responsiveness (1.08×), fast recovery rate (0.08), low side-effect sensitivity (0.92×). The most amenable patient type, chosen deliberately so the agent can learn clean control without fighting a difficult physiology.

**Events:** None. The environment is deterministic across seeds (only per-episode signal noise varies).

**Parameters:**

| Parameter | Value | Clinical rationale |
|---|---|---|
| `n_steps` | 36 | ~720 ms — one programming window |
| `max_dbs_amplitude` | 1.5 mA | Moderate ceiling; leaves clear headroom |
| `max_side_effect_load` | 0.55 | Forgiving budget; 1.0 mA sustained stays well within limits |
| `target_beta_arv` | 0.30 | Achievable suppression target for a responsive patient |
| `target_tremor_arv` | 0.24 | Low tremor appropriate for early phase |
| `target_force_preserved` | 0.78 | 78% of healthy force — realistic early-phase goal |
| `sensor_noise_std` | 0.025 | Mild sensor noise |
| `success_threshold` | **0.55** | Achievable by any policy that stimulates at all |

**Grader emphasis:** Beta suppression (0.30 weight) and tremor (0.18). This task is about establishing control, not managing crises.

**What fails this task:**
- Zero stimulation → hard penalty (−0.20 for no-DBS with poor suppression)
- Constant maximum amplitude → efficiency collapses, mild safety penalty
- Abrupt amplitude steps → smoothness penalty

**Expected scores — constant 1.0 mA policy:** 0.72–0.80 (passes). A good reactive agent: 0.78–0.88.

---

### `medium` — Rescue Phase

**Clinical scenario:** The patient is mid-deterioration. The episode starts during active symptom escalation — beta is rising, force is falling. This mirrors an adaptive DBS system activating during an "off" phase or after medication wears off.

**Patient:** `balanced` — standard entrainment scale (1.0), moderate side-effect sensitivity (1.0), normal recovery rate (0.06). Represents the typical clinical population.

**Events:** `rescue` profile:
- **Second deterioration wave** (55% probability, intensity 0.10–0.18): additive beta and tremor drive in the second half of the episode — the patient worsens again just as the agent thinks it has stabilised
- **Mild dyskinesia pressure** (30% probability, intensity 0.08–0.16): modest increase in side-effect burden if the agent over-stimulates

These events are mild — they require the agent to notice and respond, not survive a catastrophe.

**Parameters:**

| Parameter | Value | Clinical rationale |
|---|---|---|
| `n_steps` | 60 | ~1.2 s — long enough to observe a full rescue arc |
| `max_dbs_amplitude` | 1.8 mA | Higher ceiling for aggressive initial rescue |
| `max_side_effect_load` | 0.60 | Moderate budget — 1.0 mA is safe, max amplitude is not |
| `target_beta_arv` | 0.28 | Clinically meaningful suppression |
| `target_tremor_arv` | 0.32 | Active tremor reduction target |
| `target_tracking_error` | 0.28 | Moderate motor accuracy requirement |
| `sensor_noise_std` | 0.040 | Moderate sensor noise |
| `success_threshold` | **0.52** | Requires measurable rescue — constant passive policy is marginal |

**Grader emphasis:** Safety (0.22) and tracking (0.16) and force (0.16). Recovery is measured explicitly — the grader compares the patient state at the start vs. the end of the episode.

**What fails this task:**
- Passive low-amplitude strategy → tremor uncontrolled, recovery_score low
- Constant max amplitude → side effects accumulate by step ~35, safety collapses
- Missing the second deterioration wave → score drops in final third of episode

**Expected scores — constant 1.0 mA policy:** 0.47–0.52 (fails on most seeds). A good reactive agent: 0.58–0.70.

---

### `hard` — Full Episode

**Clinical scenario:** End-to-end 150-step closed-loop DBS management of a drug-resistant patient through a full session containing multiple overlapping crises. The agent must pace itself across onset, escalation, crisis events, and late-episode stability while managing a tighter safety budget on a patient who responds poorly to stimulation.

**Patient:** `refractory` — weak cortical entrainment (0.88×), high progression rate (1.10×), slow recovery (0.04), elevated adaptation gain (1.25×). This patient type requires more amplitude for the same effect and builds tolerance faster.

**Events:** `long_horizon` profile with near-guaranteed crises:

| Event | Probability | Duration | Intensity | Mechanical effect |
|---|---|---|---|---|
| `tachyphylaxis` | 82% | 12–20 steps | 0.20–0.30 | `entrainment_mult = max(0.40, 1 − 2.0 × intensity)` — up to 60% entrainment loss |
| `off_med_crisis` | 75% | 10–15 steps | 0.25–0.45 | `beta_drive_add += 0.28 × intensity` per step — genuine beta spike |
| `dyskinesia_spike` | 80%, up to 2× | 6–11 steps | 0.22–0.40 | `side_effect_burden_mult × (1.0 + 1.65 × intensity)` — accelerates overload |
| `motor_surge` | 65%, up to 2× | 5–9 steps | 0.60–0.90 | Target output overrides to high-force demand; force floor raised |

**Parameters:**

| Parameter | Value | Clinical rationale |
|---|---|---|
| `n_steps` | 150 | ~3 s simulated — full extended DBS session |
| `max_dbs_amplitude` | 2.4 mA | Highest available ceiling |
| `max_side_effect_load` | 0.40 | Tight budget — refractory patient, long session |
| `target_beta_arv` | 0.21 | Tight suppression target (below therapeutic threshold) |
| `target_tremor_arv` | 0.27 | Near-full tremor control required |
| `target_tracking_error` | 0.22 | Precise motor accuracy needed |
| `sensor_noise_std` | 0.050 | High sensor noise — real LFP/EMG quality |
| `success_threshold` | **0.68** | Requires multi-crisis management with clean terminal stability |

**Grader emphasis:** Beta (0.22) and tremor (0.14) are now the primary weights — DBS that doesn't suppress pathological beta is not doing its job, regardless of how safe it was. Safety (0.18) still matters but cannot be gamed by low-amplitude coasting.

**Why the hard grader changed from the original design:** An earlier version placed `safety_score` at 0.36 weight. This accidentally rewarded passive low-stimulation agents: a constant 1.0 mA policy (amp_norm = 0.42) never accumulated side effects, so safety ≈ 0.90 contributed 0.32 of the grader score alone — enough to reach 0.55 without ever suppressing beta. That is clinically backwards. Good DBS requires *both* adequate symptom suppression *and* safety. The current weights reflect that.

**The genuine control dilemmas this task creates:**

1. **Tachyphylaxis trap:** The agent increases amplitude to suppress beta → adaptation builds → after 15+ steps of high amplitude, entrainment drops to 40% of its prior value. The same setting that was working now provides 60% less suppression. The agent must detect this (via rising beta despite constant amplitude) and either back off to allow recovery or switch to a pulsed strategy.

2. **Off-med crisis vs. safety budget:** L-DOPA trough fires a beta spike that demands higher amplitude to compensate. But the safety budget is already partially depleted from the episode's first half. The agent must decide: increase amplitude (suppress crisis, risk dyskinesia) or hold steady (protect safety, accept beta excursion). There is no free solution.

3. **Motor surge + DBS state:** The target motor output jumps to a high-force demand during a surge. The agent must simultaneously track the new target AND maintain DBS settings appropriate for the underlying neural state — the motor task and the stimulation task compete for the agent's attention.

4. **Refractory physiology:** More amplitude produces less effect than on a responsive patient. Brute-force strategies that work on easy fail here because the same dose causes more side effects and less suppression.

**Expected scores — constant 1.0 mA policy:** 0.23–0.36 (never passes). A good reactive agent: 0.48–0.60. Passing (0.68+) requires adaptive phase-aware control.

---

## Expert Tasks

### `fragile_patient` — Tight Safety Budget

**Clinical scenario:** A safety-constrained patient with elevated side-effect sensitivity (1.40×) — clinically corresponding to patients with lower dyskinesia thresholds due to prior levodopa exposure or neural sensitisation. The therapeutic window is approximately 0.3–0.8 mA.

**Patient:** `fragile` — side_effect_sensitivity = 1.40, recovery_rate = 0.05.

**Key parameters:** `max_side_effect_load = 0.26`, `max_dbs_amplitude = 1.4 mA`, 64 steps.

**Core challenge:** The usable amplitude range is half that of the medium task. Jitter causes safety violations; timidity leaves symptoms uncontrolled. The agent must find and hold a precise therapeutic window.

**Success threshold: 0.44** — achievable only by a policy that has found and held the narrow window.

---

### `refractory_patient` — Drug-Resistant

**Clinical scenario:** A patient whose DBS response is blunted — common after years of stimulation or advanced neurodegeneration. Entrainment scale = 0.88, progression_scale = 1.10. The `long_horizon` event profile fires (same as hard) with recurring tachyphylaxis.

**Core challenge:** "More DBS" is not the answer. Brute-force amplitude produces similar entrainment to a moderate policy on a responsive patient, but with 1.25× higher adaptation gain and more side effects. The agent must discover pulsed stimulation — higher amplitude during escalation, genuine rest periods during stability.

**Success threshold: 0.46** over 120 steps.

---

### `personalization_generalization` — Mixed Profiles

**Clinical scenario:** Evaluates whether a policy generalises across patient phenotypes without per-patient prior history. All four profiles (balanced, responsive, fragile, refractory) appear across episodes. The profile ID is visible in reset metadata but the agent has no prior episode history for that patient.

**Core challenge:** A policy specialised for responsive patients will over-stimulate fragile patients and under-treat refractory ones. Success requires a meta-strategy: read the profile, infer the therapeutic window, and apply profile-appropriate settings from step one.

**Success threshold: 0.50** over 90 steps.

---

### `exercise_bout` — Exercise Burst

**Clinical scenario:** A patient performing sustained physical exercise. A `motor_surge` event fires with certainty (100% probability) in the first 30% of the episode — the patient is suddenly demanding high-force output during intense voluntary activity. A post-exertion dyskinesia spike may follow (40% probability) as accumulated DBS + exertion interact.

**Key challenge:** The agent must ramp DBS to support the high-force demand during exercise, then rapidly taper when the surge ends before dyskinesia risk accumulates. Zero-stim during exertion is a hard failure (−0.16 penalty).

**Grader emphasis:** Force (0.22) and tracking (0.22) — motor performance during the bout is the primary clinical goal.

**Success threshold: 0.55** over 70 steps.

---

### `medication_interaction` — L-DOPA Interaction

**Clinical scenario:** Phase-coupled medication dynamics. A guaranteed off-med crisis fires in the middle of the episode (steps 30–45%) as L-DOPA wears off — beta surges sharply. If the agent over-responds with high amplitude, a dyskinesia spike follows in the second half (65% probability) as the next dose kicks in and DBS interaction accumulates.

**The clinical dilemma:** The correct response to an off-med crisis is to increase DBS. But over-increasing DBS when the next L-DOPA dose is approaching creates dyskinesia. The agent must time its response and taper before the medication rebound.

**Grader emphasis:** Safety (0.22) and recovery (0.10) — the agent must handle the crisis without over-treating.

**Success threshold: 0.50** over 100 steps.

---

### `nocturnal_transition` — Sleep Transition

**Clinical scenario:** The patient transitions from waking activity through wind-down into sleep. The `nocturnal` schedule progressively tightens beta and tremor targets from step 40% onward, peaking in the sleep phase (step 65%+). Targets are tightest during sleep because patient movement is minimal and any residual beta oscillation disrupts sleep quality.

**Time-varying schedule:**
- 0–40%: full waking demands
- 40–65%: wind-down (targets tighten by 10–15%)
- 65–100%: sleep phase (targets tighten by 20–35% from baseline)

**Core challenge:** The agent must progressively reduce stimulation as motor demands drop but maintain biomarker control during sleep — a different operating point than the waking phase.

**Success threshold: 0.55** over 150 steps.

---

### `surgical_followup` — Post-Implant Programming

**Clinical scenario:** First-week post-implant DBS programming during the microlesion window — post-surgical swelling temporarily improves symptoms, meaning the same amplitude delivers more effect and risks over-treatment. An amplitude ceiling of 0.6 mA is enforced for the first 25% of the episode (the `surgical_microlesion` schedule). Impedance surges (70% probability) may fire as the electrode settles, reducing delivered current without warning.

**Hard constraint:** Any amplitude violations during the microlesion window trigger a −0.20 penalty — the clinical equivalent of causing stimulation-induced dyskinesia in a just-implanted patient.

**Grader emphasis:** Safety (0.30) — this task is about disciplined programming under hardware and physiological constraints.

**Success threshold: 0.50** over 120 steps.

---

## Task Parameter Summary

| Task | Difficulty | Steps | Patient | Events | SE Budget | Threshold |
|---|---|---:|---|---|---:|---:|
| `easy` | Easy | 36 | responsive | none | 0.55 | **0.55** |
| `medium` | Medium | 60 | balanced | rescue (mild) | 0.60 | **0.52** |
| `hard` | Hard | 150 | refractory | long_horizon (heavy) | 0.40 | **0.68** |
| `fragile_patient` | Expert | 64 | fragile | none | 0.26 | 0.44 |
| `refractory_patient` | Expert | 120 | refractory | long_horizon | 0.48 | 0.46 |
| `personalization_generalization` | Expert | 90 | all four | long_horizon | 0.40 | 0.50 |
| `exercise_bout` | Expert | 70 | balanced | exercise | 0.55 | 0.55 |
| `medication_interaction` | Expert | 100 | fragile | medication | 0.52 | 0.50 |
| `nocturnal_transition` | Expert | 150 | balanced | nocturnal | 0.55 | 0.55 |
| `surgical_followup` | Expert | 120 | balanced | surgical | 0.55 | 0.50 |

---

## Difficulty Calibration Validation

Scores for a constant 1.0 mA / 0.13 ms / 130 Hz / motor_command=target policy across 5 seeds:

```
easy    0.724  0.731  0.804  0.739  0.744   min=0.72   threshold=0.55  → always passes
medium  0.485  0.470  0.518  0.476  0.509   min=0.47   threshold=0.52  → never passes
hard    0.358  0.231  0.338  0.252  0.329   min=0.23   threshold=0.68  → never passes
```

Strict ordering holds every seed: `easy > medium > hard`. The hard task's minimum (0.23) is well below medium's minimum (0.47). A good reactive LLM agent is expected to score roughly:
- **easy:** 0.78–0.88 (reliable pass)
- **medium:** 0.58–0.70 (passes with correct rescue response)
- **hard:** 0.48–0.62 (struggles; needs phase-aware strategy to reach 0.68)

---

## References

- Castrioto A et al. (2011). "Ten-year outcome of subthalamic stimulation in Parkinson disease." *Arch Neurol* 68(12):1550–1556.
- Fleming JE et al. (2023). "Multivariable closed-loop control of deep brain stimulation for Parkinson's disease." *J Neural Eng* 20(5):056029.
- Little S et al. (2016). "Closed-loop deep brain stimulation: An evolving technology." *Mov Disord* 31(8):1336–1341.
- Olanow CW et al. (2013). "Levodopa in the treatment of Parkinson's disease: current controversies." *Mov Disord* 19(9):997–1005.
- Tinkhauser G et al. (2017). "Beta burst dynamics in Parkinson's disease OFF and ON dopaminergic medication." *Brain* 140(11):2968–2981.
