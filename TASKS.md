# Task Specifications — MotorAssistEnv

## 1. What does the agent actually have to do?

Every 20 milliseconds — 50 times a second — the agent acts as a closed-loop adaptive Deep Brain Stimulation (aDBS) controller for a Parkinson's patient with an implanted brain stimulator. It reads the patient's neural and motor state, then turns four continuous dials: stimulation amplitude `dbs_amplitude` (mA), pulse width `dbs_pulse_width` (ms), pulse rate `dbs_frequency` (Hz), and the voluntary motor command the patient is attempting. Under-stimulate and tremor and rigidity break through; over-stimulate and the patient develops dyskinesia while the device's battery drains and tissue impedance drifts.

The patient is not scripted. Their basal ganglia, motoneuron pool, and motor output are simulated by a biophysically grounded model derived from Fleming et al. (2023), so every action has compounding physiological consequences. The episode is graded on four axes — pathological activity suppressed, voluntary motor function preserved, cumulative safety budget respected, and a clinically stable end state — and there is no single correct dose. The right setting depends on the patient profile, the disease state, and the last few seconds of history. The ten tasks below stress different parts of that decision.

---

## 2. What are the 10 tasks at a glance?

| Task | What's actually happening | Patient | What's hard about it |
|---|---|---|---|
| `easy` | Brand-new patient, calm conditions, just hold a sensible dose for ~0.7 sec | `responsive` (easy to treat) | Almost nothing — it's the smoke test |
| `medium` | Symptoms flare up mid-episode; agent must rescue without overdoing it | `balanced` (typical) | Reacting to a deterioration wave without causing dyskinesia |
| `hard` | Long, drug-resistant patient through multiple overlapping crises | `refractory` (difficult) | Pacing through 4 different crisis types over 150 steps |
| `fragile_patient` | Patient who can't tolerate much stimulation — narrow safe range | Sensitive to side effects | Finding and holding the small therapeutic window |
| `refractory_patient` | DBS works less well — needs creative pulsed strategy | Drug-resistant | "More DBS" stops working; brute force fails |
| `personalization_generalization` | A different patient type each episode | All four mixed | Reading the profile and adapting from step 1 |
| `exercise_bout` | Patient suddenly starts intense exercise mid-episode | Average | Ramping up for the bout, then tapering before dyskinesia |
| `medication_interaction` | L-DOPA wears off → beta surges, then next dose kicks in | Sensitive | Timing the response so DBS + next dose don't combine into dyskinesia |
| `nocturnal_transition` | Patient transitions from waking → wind-down → sleep | Average | Different operating point in sleep; tighter targets, less motor demand |
| `surgical_followup` | First week post-implant — swelling temporarily boosts DBS effect | Average | Hard amplitude ceiling for 25% of episode; impedance can surge |

---

## 3. What does each task look like in detail?

### `easy` — Calm Start

A patient just started DBS. The agent has to find a sensible stimulation level early and hold it cleanly — no events, no surprises, just early titration in a `responsive` patient with rising beta and mild tremor.

| Param | Value | Why |
|---|---|---|
| `n_steps` | 36 | ~720 ms — one programming window |
| `max_dbs_amplitude` | 1.5 mA | Moderate ceiling |
| `max_side_effect_load` | 0.55 | Forgiving; constant 1.0 mA stays well within |
| `target_beta_arv` | 0.30 | Achievable for a responsive patient |
| `success_threshold` | **0.55** | Anyone who stimulates at all should pass |

Fails if the agent stimulates not at all (−0.20 penalty), pins amplitude at the maximum (−0.14 efficiency penalty), or swings amplitude jaggedly (smoothness penalty).

A constant 1.0 mA policy scores 0.72–0.80 here. A good reactive agent reaches 0.78–0.88.

---

### `medium` — Rescue Phase

The patient is mid-deterioration. The agent walks in while symptoms are getting worse and has to bring things back without going so aggressive it triggers dyskinesia. Active symptom escalation in a `balanced` patient with the `rescue` event profile.

Two events fire stochastically: a **second deterioration wave** (55% probability) — additive beta and tremor drive in the second half of the episode, where the patient worsens just as the agent thinks it has stabilised — and **mild dyskinesia pressure** (30% probability) where the side-effect burden creeps up if the agent over-stimulates.

| Param | Value | Why |
|---|---|---|
| `n_steps` | 60 | ~1.2 s — long enough for a full rescue arc |
| `max_dbs_amplitude` | 1.8 mA | Higher ceiling for aggressive initial rescue |
| `max_side_effect_load` | 0.60 | 1.0 mA is safe; max amplitude is not |
| `success_threshold` | **0.52** | Requires measurable rescue; passive policy fails |

Fails if the agent plays it safe and lets tremor stay uncontrolled, pins the amplitude high until safety collapses around step 35, or misses the second deterioration wave entirely.

A constant 1.0 mA policy scores 0.47–0.52 (fails most seeds). A good reactive agent reaches 0.58–0.70.

---

### `hard` — Full Episode

A long session managing a difficult patient through several different crises that overlap — like running a 5-kilometre race across uneven terrain. The agent has to pace itself, avoid overreacting, and finish in a stable state. End-to-end 150-step closed-loop DBS on a `refractory` patient (weak entrainment, slow recovery, elevated adaptation gain) with the `long_horizon` event profile.

The events are near-guaranteed:

| Event | Prob | What it does mechanically |
|---|---|---|
| `tachyphylaxis` | 82% | Up to 60% entrainment loss for 12–20 steps |
| `off_med_crisis` | 75% | Genuine beta spike for 10–15 steps |
| `dyskinesia_spike` | 80% (×2) | Side-effect accumulation accelerates 1.65× |
| `motor_surge` | 65% (×2) | Target output jumps to high-force demand |

| Param | Value | Why |
|---|---|---|
| `n_steps` | 150 | Full extended DBS session |
| `max_dbs_amplitude` | 2.4 mA | Highest ceiling |
| `max_side_effect_load` | 0.40 | Tight — refractory patient, long session |
| `success_threshold` | **0.68** | Multi-crisis management with clean terminal stability |

The four real control dilemmas this task creates:

1. **Tachyphylaxis trap** — the same setting that worked at step 30 produces 60% less suppression at step 100. The agent must detect this from rising beta despite constant amplitude, then either back off to allow recovery, or switch to a pulsed strategy.
2. **Off-med crisis vs safety budget** — an L-DOPA trough demands more amplitude, but the safety budget is already half-spent. There's no free solution; the agent has to trade.
3. **Motor surge with neural state** — target output jumps to a high-force demand during a surge, and the agent must satisfy the motor task while still maintaining DBS appropriate for the underlying brain state.
4. **Refractory physiology** — more amplitude produces less effect than on a responsive patient, so brute force that worked on easy fails here.

A constant 1.0 mA policy scores 0.23–0.36 (never passes). A good reactive agent reaches 0.48–0.60. Reliable passing requires phase-aware adaptive control.

---

### `fragile_patient` — Tight Safety Budget

A patient with a small "safe zone" — turn it up too far and they get bad side effects fast; turn it down too far and symptoms break through. The agent has to find that narrow window and stay in it. Clinically, this represents elevated dyskinesia sensitivity (1.40×) often seen after long levodopa exposure, with a therapeutic window of roughly 0.3–0.8 mA.

| Param | Value |
|---|---|
| Patient profile | `fragile` (side_effect_sensitivity = 1.40, recovery_rate = 0.05) |
| `n_steps` | 64 |
| `max_dbs_amplitude` | 1.4 mA |
| `max_side_effect_load` | 0.26 |
| `success_threshold` | **0.44** |

The usable amplitude range is half that of the medium task. Jitter causes safety violations; timidity leaves symptoms uncontrolled.

---

### `refractory_patient` — Drug-Resistant

A patient whose brain has stopped responding well to stimulation — common after years on DBS. Cranking the dose doesn't help; the agent has to be smarter and pulse the stimulation instead. Blunted DBS response (entrainment scale = 0.88) with the same `long_horizon` event profile as `hard`, including recurring tachyphylaxis.

| Param | Value |
|---|---|
| Patient profile | `refractory` |
| `n_steps` | 120 |
| `success_threshold` | **0.46** |

Brute-force amplitude here produces similar entrainment to a moderate policy on a responsive patient — but with 1.25× higher adaptation gain and more side effects. The winning move is *pulsed* stimulation: higher amplitude during escalation, genuine rest periods during stability.

---

### `personalization_generalization` — Mixed Profiles

Each episode the patient is different. The agent has to read who's in front of it and adjust its strategy from the very first step. All four profiles (`balanced`, `responsive`, `fragile`, `refractory`) appear across episodes. The profile ID is in reset metadata; there's no per-patient prior history to lean on.

| Param | Value |
|---|---|
| `n_steps` | 90 |
| Events | `long_horizon` |
| `success_threshold` | **0.50** |

A policy specialised for responsive patients will over-stimulate fragile ones and under-treat refractory ones. Success requires reading the profile and applying profile-appropriate settings from step 1.

---

### `exercise_bout` — Exercise Burst

The patient suddenly starts exercising hard. The agent has to ramp DBS up to support the activity, then quickly dial it back down before side effects kick in. A `motor_surge` event fires with certainty in the first 30% of the episode; a post-exertion dyskinesia spike may follow with 40% probability.

| Param | Value |
|---|---|
| `n_steps` | 70 |
| Grader emphasis | force 0.22, tracking 0.22 |
| `success_threshold` | **0.55** |

Zero stimulation during the exertion period is a hard failure (−0.16 penalty).

---

### `medication_interaction` — L-DOPA Interaction

The patient's medication wears off mid-episode and symptoms surge. The agent has to push DBS up — but not so much that when the next dose kicks in, the combination causes dyskinesia. An off-med crisis fires guaranteed at steps 30–45%. If the agent over-responds, a dyskinesia spike follows in the second half (65% probability) as the next L-DOPA dose accumulates and DBS interaction builds.

| Param | Value |
|---|---|
| Patient profile | `fragile` |
| `n_steps` | 100 |
| Grader emphasis | safety 0.22, recovery 0.10 |
| `success_threshold` | **0.50** |

The dilemma is real: the correct response to an off-med crisis is *more* DBS, but over-increasing while the next dose approaches creates dyskinesia. The agent has to time its response and taper before the medication rebound.

---

### `nocturnal_transition` — Sleep Transition

The patient is going to bed. The agent has to gradually reduce stimulation as the patient stops moving, but tighten control of the underlying brain rhythms because residual oscillations disrupt sleep. The `nocturnal` schedule progressively tightens beta and tremor targets from step 40% onward, peaking in the sleep phase.

| Phase | Step range | What happens |
|---|---|---|
| Waking | 0–40% | Full motor demands |
| Wind-down | 40–65% | Targets tighten 10–15% |
| Sleep | 65–100% | Targets tighten 20–35% from baseline |

| Param | Value |
|---|---|
| `n_steps` | 150 |
| `success_threshold` | **0.55** |

A different operating point than waking — less stimulation, but tighter biomarker control.

---

### `surgical_followup` — Post-Implant Programming

The patient just had the device implanted last week. Post-surgical swelling is making the same dose feel stronger than it normally would, so the agent must stay below a strict ceiling for the first quarter of the session — like the speed limits in a school zone. A hard amplitude ceiling of 0.6 mA is enforced for the first 25% of the episode (the `surgical_microlesion` schedule), and impedance surges (70% probability) reduce delivered current as the electrode settles.

| Param | Value |
|---|---|
| `n_steps` | 120 |
| Grader emphasis | safety 0.30 |
| `success_threshold` | **0.50** |

Any amplitude violation in the microlesion window triggers a −0.20 penalty — the clinical equivalent of causing stimulation-induced dyskinesia in a just-implanted patient.

---

## 4. Why ten tasks and not one?

A benchmark with one task teaches an agent to be good at that one task. We want to test whether the agent has learned a *strategy* it can adapt to new situations, so we put it in lots of different rooms with the same toolkit. The 10 tasks fall into three deliberate buckets, each measuring a different skill.

| Bucket | Tasks | Skill being measured | What a constant-dose policy does |
|---|---|---|---|
| **Difficulty ladder** | `easy`, `medium`, `hard` | Climb a difficulty curve | Passes easy, borderline medium, fails hard |
| **Patient-class generalisation** | `fragile`, `refractory`, `personalization` | Transfer across patients | Hard-coded amplitude breaks when profile changes |
| **Clinical scenarios** | `exercise`, `medication`, `nocturnal`, `surgical` | Recognise a *situation* and adjust | "Same as on hard" is the wrong answer for every one |

| What the agent passes | What it has learned |
|---|---|
| Easy / medium / hard only | Generic patient control |
| Above + patient-class scenarios | Generalisation across patients |
| Above + clinical scenarios | Has internalised what DBS is *for* — situational reasoning |

---

## 5. How were the success thresholds set so they actually mean something?

We didn't pick numbers that "feel right." We ran the simplest possible policy — constant dose, never adjust — and put each threshold just above what that policy could achieve. Passing therefore genuinely means the agent did something better than nothing. The thresholds are anchored to the constant 1.0 mA baseline, not to any particular LLM, so they don't move when LLMs get better.

| Tier | Threshold | Constant baseline | Why this gap |
|---|---|---|---|
| Easy | 0.55 | ≈ 0.72–0.80 | Smoke test — anyone with a basic policy clears it |
| Medium | 0.52 | ≈ 0.47–0.52 | Forces *something* mid-episode; constant fails most seeds |
| Hard | 0.68 | ≈ 0.23–0.36 | Requires titration + reaction + recovery; no shortcut |
| Expert / scenario | 0.44–0.55 | Varies | Recognising the scenario should dominate the signal |

---

## 6. Where did the per-task numbers come from?

| Parameter | Source |
|---|---|
| `target_beta_arv` | Clinical literature: well-treated PD resting beta = 6.5–8.0 µV ARV (Tinkhauser 2017) |
| `target_tremor_arv` | Resting-tremor amplitude in mild-to-moderate PD (Deuschl 2006) |
| Patient-profile thresholds | Severity multipliers in `patient_profiles.py` × baseline target |
| `max_side_effect_load` | Tuned per-task: ran the constant baseline, chose the budget that produces the calibration table below |
| `success_threshold` | Set last, after watching the baseline across many seeds — "just above what a non-strategy passes" |

---

## 7. Does the difficulty ordering actually hold?

Yes. The simple constant-dose policy was run 5 times on each task. Easy always beats medium, medium always beats hard, and hard never passes — so the difficulty labels are real, not aspirational.

Constant 1.0 mA / 0.13 ms / 130 Hz / `motor_command = target` across 5 seeds:

```
easy    0.724  0.731  0.804  0.739  0.744   min=0.72   threshold=0.55  → always passes
medium  0.485  0.470  0.518  0.476  0.509   min=0.47   threshold=0.52  → never passes
hard    0.358  0.231  0.338  0.252  0.329   min=0.23   threshold=0.68  → never passes
```

| Task | Expected for a good reactive agent | Verdict |
|---|---|---|
| easy | 0.78–0.88 | Reliable pass |
| medium | 0.58–0.70 | Passes with correct rescue |
| hard | 0.48–0.62 | Marginal — needs phase-aware crisis management |

---

## 8. References

- Castrioto A et al. (2011). *Arch Neurol* 68(12):1550–1556.
- Fleming JE et al. (2023). *J Neural Eng* 20(5):056029.
- Little S et al. (2016). *Mov Disord* 31(8):1336–1341.
- Olanow CW et al. (2013). *Mov Disord* 19(9):997–1005.
- Tinkhauser G et al. (2017). *Brain* 140(11):2968–2981.
- Deuschl G et al. (2006). *NEJM* 355(9):896–908.
