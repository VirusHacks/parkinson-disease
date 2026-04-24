# Task Specifications — MotorAssistEnv

## Overview

MotorAssistEnv defines six tasks across two tiers. The three core public tasks form a
curriculum ladder from introductory to full clinical benchmark. The three advanced
extension tasks add scenario complexity for safety sensitivity, weak-response patients,
and cross-patient generalisation.

All tasks share the same 27-field observation space, 4-field action space, and
environment dynamics. What changes per task is the clinical scenario:

- **Episode window**: which phase of the Fleming 100-step trajectory is active
- **Episode length**: how many control steps the agent takes
- **DBS ceiling**: the maximum allowable amplitude and pulse-width
- **Safety budget**: the maximum cumulative side-effect load before safety score collapses
- **Clinical targets**: beta, tremor, force, and tracking thresholds used by the grader
- **Patient profile pool**: which patient types may appear
- **Success threshold**: minimum grader score to declare the episode a success

**Curriculum design principle (from hackathon guide):** success probability must be
greater than zero for learning to occur. Easy tasks should be solvable by a thoughtful
rule-based policy. Hard tasks should require temporal reasoning about cumulative effects.
The boundary is not "hard enough to look impressive" — it is "hard enough that a
naive agent fails while a smart agent can succeed."

---

## Core Public Tasks

### Task 1: `beta_suppression` — Easy

**Clinical scenario:** Pre-emptive DBS calibration during the early-phase window,
before tremor escalation. This mirrors the clinical practice of programming a newly
implanted IPG in a responsive patient who is still in good baseline motor condition.

**Patient population:** Responsive profile only (entrainment_scale=1.08,
side_effect_sensitivity=0.92, recovery_rate=0.08). The most amenable patient type —
chosen deliberately so the agent can establish clean control patterns without fighting
a difficult physiology.

**Episode parameters:**

| Parameter | Value | Justification |
|---|---|---|
| `start_step` | 5 | Before the first significant beta escalation in the Fleming trajectory |
| `n_steps` | 30 | Short horizon: 600 ms of closed-loop control (clinically: one programming window) |
| `max_dbs_amplitude` | 1.5 mA | Below typical therapeutic ceiling; leaves clear headroom without trivialising |
| `max_dbs_pulse_width` | 0.15 ms | Moderate spatial spread |
| `max_side_effect_load` | 0.55 | Forgiving budget; 1.0 mA sustained stays well within limits |
| `target_force_preserved` | 0.78 | 78% of healthy force — realistic early-phase preservation target |
| `target_beta_arv` | 0.26 | Below the clinical suppression threshold (Tinkhauser et al.: <0.30 indicates good suppression) |
| `target_tremor_arv` | 0.20 | Low tremor target appropriate for early phase |
| `success_threshold` | **0.50** | Half the maximum score — achievable by a coherent low-amplitude policy |

**What must happen for success:**
- Apply DBS (zero-stim fails via hard-failure penalty)
- Keep amplitude below 1.5 mA ceiling (no constraint violations)
- Maintain `side_effect_load < 0.40` across 30 steps
- Keep `beta_arv < 0.26` on most steps

**What fails this task:**
- `amp = 0` throughout (hard penalty: −0.20)
- Constant max amplitude (efficiency_score collapses, mild safety penalty)
- Abrupt amplitude changes without beta feedback (smoothness + safety cost)

**Expected score distribution:**

| Policy | Score range | Pass? |
|---|---|---|
| zero_stim | 0.18–0.28 | No |
| constant 1.0 mA, 130 Hz | 0.44–0.56 | Marginal |
| adaptive rule-based | 0.52–0.64 | Yes |

---

### Task 2: `tremor_correction` — Medium

**Clinical scenario:** Acute tremor rescue. The episode begins at step 16 — the point
in the Fleming trajectory where tremor is actively escalating and force is falling.
This mirrors an aDBS system activating during a patient's "off" phase or after a
medication dose has worn off.

**Patient population:** Balanced or responsive (sampled randomly per episode).
Balanced patients have standard DBS response; responsive patients respond somewhat
faster. The randomisation forces the agent to track the observation rather than
memorise a fixed response pattern.

**Episode parameters:**

| Parameter | Value | Justification |
|---|---|---|
| `start_step` | 16 | Peak tremor escalation window in the Fleming trajectory |
| `n_steps` | 48 | 960 ms — long enough for a full rescue arc to be observed |
| `max_dbs_amplitude` | 1.8 mA | Higher ceiling to allow aggressive initial rescue |
| `max_dbs_pulse_width` | 0.18 ms | Near-maximum spatial spread available |
| `max_side_effect_load` | 0.60 | Moderate budget — 1.0 mA is safe, 1.8 mA (max) unsustainable |
| `target_force_preserved` | 0.64 | 64% of healthy force — the rescue target after active tremor |
| `target_tremor_arv` | 0.32 | Meaningful tremor reduction but not full suppression |
| `success_threshold` | **0.40** | Requires measurable rescue — constant safe-zero policy fails |

**The core control challenge:** A constant high-amplitude policy reaches the safety
budget ceiling around step 30–35, at which point safety_score starts collapsing.
The agent must increase amplitude rapidly in steps 1–15 to rescue the escalating
tremor, then pull back to a maintenance level. This temporal modulation — push then
sustain — is the key behaviour.

**What the `recovery_score` measures:** The grader explicitly compares the first-6-step
window (during escalation) to the last-8-step window. An agent that arrives with the
patient in bad shape and leaves them in good shape scores well on recovery. An agent
that just sustains a mediocre state throughout does not.

**Hard-failure rules:**
- `tremor_score < 0.22`: tremor not meaningfully reduced → −0.10
- `mean_amp < 0.10` + `tremor_score < 0.28` or `recovery_score < 0.22`: no rescue attempted → −0.16

**Expected score distribution:**

| Policy | Score range | Pass? |
|---|---|---|
| zero_stim | 0.10–0.20 | No |
| constant 1.0 mA, 130 Hz | 0.35–0.48 | Marginal |
| constant 1.8 mA (max), 130 Hz | 0.28–0.38 | No (side effects) |
| adaptive: push then sustain | 0.42–0.58 | Yes |

---

### Task 3: `full_episode` — Hard

**Clinical scenario:** End-to-end closed-loop DBS management over a complete 100-step
clinical episode (2000 ms of simulated time). The agent must handle the full arc:
early-phase pre-emption (steps 0–15), peak escalation rescue (steps 16–40),
mid-episode sustained maintenance (steps 40–70), and terminal stability (steps 70–100).

**Patient population:** Balanced, responsive, or refractory (sampled randomly).
The refractory patient has reduced entrainment response (0.88×) and faster disease
progression — appearing in ~1/3 of episodes forces the agent to detect and adapt
to a patient who simply responds less to the same DBS settings.

**Episode parameters:**

| Parameter | Value | Justification |
|---|---|---|
| `start_step` | 0 | Full trajectory from onset |
| `n_steps` | 100 | Complete clinical episode |
| `max_dbs_amplitude` | 2.4 mA | Highest available ceiling for the hardest scenario |
| `max_dbs_pulse_width` | 0.20 ms | Full pulse-width range |
| `max_side_effect_load` | 0.55 | Stricter long-horizon safety budget; sustained high amplitude now exhausts the episode budget faster |
| `target_force_preserved` | 0.60 | Sustained function across the full episode |
| `success_threshold` | **0.66** | Stricter high threshold — requires multi-phase adaptive control with cleaner budget management |

**Why safety dominates (36% grader weight):** At 100 steps, an agent that runs
maximum amplitude for the first 60 steps will exhaust the side-effect budget and
be unable to deliver any safe DBS for the remaining 40 steps. This is the most
dangerous clinical outcome — the patient is left unprotected during the terminal
phase when cumulative fatigue and disease pressure are highest.

**The two failure modes this task is designed to expose:**

1. **Greedy front-loading:** Max amplitude in steps 1–40 → safety budget depleted
   by step 50 → `side_effect_load > 0.55` → safety_score collapses → episode fails.

2. **Conservative under-treatment:** Low constant amplitude to preserve safety budget
   → tremor escalates unchecked in steps 16–40 → force collapses → terminal state
   is bad → `terminal_stability_score` collapses → hard penalty.

**The optimal strategy** uses a three-phase approach: moderate pre-emption (0.8–1.2 mA),
aggressive rescue (1.5–2.0 mA, reducing when side_effect_rate turns positive),
sustained maintenance at minimum effective amplitude (0.6–1.0 mA).

**Per-episode stochasticity:** Each reset samples independent noise (std=0.08) on
beta and tremor baselines. Policies that memorise the exact Fleming trajectory degrade
across episodes — the agent must react to observed biomarkers, not play back a cached
control sequence.

**Expected score distribution:**

| Policy | Score range | Pass? |
|---|---|---|
| zero_stim | 0.12–0.22 | No |
| constant 1.0 mA, 130 Hz | 0.30–0.42 | No |
| constant 2.4 mA (max), 130 Hz | 0.20–0.30 | No (side effects exhaust budget) |
| safety_aware rule-based | 0.45–0.58 | No (misses threshold) |
| adaptive multi-phase | 0.66–0.78 | Yes |

---

## Advanced Extension Tasks

These tasks are intentionally harder and are presented as research benchmarks,
not the primary public ladder. They add scenario difficulty without changing the
evaluation framework.

---

### Task 4: `fragile_patient` — Expert

**Clinical scenario:** Safety-constrained programming for a patient with elevated
side-effect sensitivity (1.40×) — corresponding clinically to patients with lower
dyskinesia thresholds, possibly due to prior levodopa exposure or neural sensitisation
(Olanow et al. 2013). The tolerable stimulation range is narrow.

**Key parameters:**
- `max_side_effect_load = 0.26` (vs 0.40–0.55 in core tasks)
- `max_dbs_amplitude = 1.4 mA`
- `patient_profile_ids = ("fragile",)` — side_effect_sensitivity = 1.40, recovery_rate = 0.05

**Core challenge:** The therapeutic window (amplitude range that suppresses symptoms
without violating side-effect budget) is approximately 0.4–0.9 mA — half the range
available in the medium task. The agent must find and hold this narrow window with
precise, smooth adjustments. Jitter causes safety violations; timidity leaves symptoms
uncontrolled.

**Success threshold: 0.44** — achievable only by a policy that has found the therapeutic
window and holds it steadily. Success probability for a zero-shot LLM agent: ~20–35%.

---

### Task 5: `refractory_patient` — Expert

**Clinical scenario:** A patient whose DBS response is blunted — common in patients
who have had DBS for several years and developed stimulation tolerance, or in patients
with more advanced neurodegeneration (Castrioto et al. 2011). Entrainment scale = 0.88,
tremor_responsiveness = 0.88, progression_scale = 1.10.

**Key challenge:** Naive brute-force (max amplitude) produces similar entrainment to
a moderate-amplitude policy in a responsive patient, but with 1.12× higher side effects
and faster adaptation (the adaptation_gain coefficient is 1.25× for refractory profiles).
The agent must discover that "more DBS" is not the right answer and instead use pulsed
stimulation — higher amplitude during escalation phases, genuine rest periods during
stable phases — to extract therapeutic value while limiting adaptation.

**Success threshold: 0.42** over 100 steps with a refractory patient is harder than
0.66 on a mixed-profile full_episode.

---

### Task 6: `personalization_generalization` — Expert

**Clinical scenario:** Evaluating whether a policy generalises across patient phenotypes
without per-patient adaptation. The profile is revealed in the reset metadata
(`patient_profile_id`) but the agent has no prior episode history for that patient.
This mirrors the clinical challenge of programming a DBS system for a new patient
during the first programming visit.

**Patient pool:** All four profiles (balanced, responsive, fragile, refractory) —
each appears with equal probability.

**Why this is the hardest benchmark:** A policy that specialises for responsive
patients will over-stimulate fragile patients (side-effect violations) and
under-treat refractory patients (insufficient suppression). Success requires a
meta-strategy: read the profile from metadata, infer the therapeutic window,
and apply profile-appropriate DBS settings from the first step.

**Success threshold: 0.45** — the only way to reach this across all profiles is
a policy that meaningfully reads and uses the patient profile context.

---

## Task Parameter Summary

| Task | Start | Steps | Max Amp | SE Budget | Profiles | Threshold | Tier |
|---|---|---|---|---|---|---|---|
| `beta_suppression` | 5 | 30 | 1.5 mA | 0.55 | responsive | **0.50** | Core/Easy |
| `tremor_correction` | 16 | 48 | 1.8 mA | 0.60 | balanced, responsive | **0.36** | Core/Medium |
| `full_episode` | 0 | 100 | 2.4 mA | 0.55 | balanced, responsive, refractory | **0.66** | Core/Hard |
| `fragile_patient` | 12 | 64 | 1.4 mA | 0.26 | fragile | **0.44** | Expert |
| `refractory_patient` | 0 | 100 | 2.2 mA | 0.48 | refractory | **0.42** | Expert |
| `personalization_generalization` | 10 | 72 | 1.9 mA | 0.40 | all four | **0.45** | Expert |

---

## Difficulty Calibration Rationale

The hackathon guide (helpguide.txt, section 1) states:

> "The task is hard enough to be interesting, but not so hard that the model never succeeds.
> RL only works if the probability of getting a good answer is greater than zero."

These thresholds are calibrated so that:

- **Easy (0.50):** A thoughtful constant policy (1.0 mA, 130 Hz, motor_command = target_output)
  scores 0.44–0.56. A zero-shot LLM agent that reads the action description should reliably pass.

- **Medium (0.40):** The same constant policy scores 0.35–0.48 — marginal. The agent must
  detect the tremor escalation signal and increase amplitude during steps 1–15. One-step
  reactive control is sufficient. Zero-shot LLM success probability: ~40–60%.

- **Hard (0.66):** No constant policy reaches this. Requires multi-phase temporal reasoning
  across 100 steps. Zero-shot LLM success probability: ~5–20%. A policy trained with even
  a few episodes of RL should improve measurably.

- **Expert (0.42–0.45):** Designed for RL-trained policies. Zero-shot success probability:
  ~5–15% depending on the task. Provides headroom for demonstrating learning gains.

---

## References

- Castrioto A et al. (2011). "Ten-year outcome of subthalamic stimulation in Parkinson disease." *Arch Neurol* 68(12):1550–1556.
- Fleming JE et al. (2020). "Simulation of closed-loop deep brain stimulation control schemes." *PLOS Comput Biol* 16(8):e1008165.
- Little S et al. (2016). "Closed-loop deep brain stimulation: An evolving technology." *Mov Disord* 31(8):1336–1341.
- Olanow CW et al. (2013). "Levodopa in the treatment of Parkinson's disease: current controversies." *Mov Disord* 19(9):997–1005.
- Tinkhauser G et al. (2017). "Beta burst dynamics in Parkinson's disease OFF and ON dopaminergic medication." *Brain* 140(11):2968–2981.
