# Task Specifications - MotorAssistEnv

## 1. What does the agent actually have to do?

Every 20 milliseconds - 50 times a second - the agent acts as a closed-loop adaptive Deep Brain Stimulation (aDBS) controller for a Parkinson's patient with an implanted brain stimulator. It reads the patient's neural and motor state, then turns four continuous dials: stimulation amplitude `dbs_amplitude` (mA), pulse width `dbs_pulse_width` (ms), pulse rate `dbs_frequency` (Hz), and the voluntary motor command the patient is attempting. Under-stimulate and tremor and rigidity break through; over-stimulate and the patient develops dyskinesia while the device's battery drains and tissue impedance drifts.

The patient is not scripted. Their basal ganglia, motoneuron pool, and motor output are simulated by a biophysically grounded model derived from Fleming et al. (2023), so every action has compounding physiological consequences. The episode is graded on four axes - pathological activity suppressed, voluntary motor function preserved, cumulative safety budget respected, and a clinically stable end state - and there is no single correct dose. The right setting depends on the patient profile, the disease state, and the last few seconds of history. The ten tasks below stress different parts of that decision.

---

## 2. What are the 10 tasks at a glance?

### Difficulty Ladder

These three tasks use the same patient family and grader logic but ramp up crisis load, episode length, and physiological complexity. They are the core benchmark - the agent must climb all three to demonstrate it has actually learned to control DBS.

| Task | Scenario name | Steps | Patient | What's hard | Threshold |
|---|---|---:|---|---|---:|
| `easy` | **Calm Start** | 36 | `responsive` | Brand-new patient, calm conditions. Hold a sensible dose for 36 steps - the smoke test. | 0.55 |
| `medium` | **Rescue Phase** | 60 | `balanced` | Symptoms flare mid-episode; rescue without overdoing it and triggering dyskinesia. | 0.52 |
| `hard` | **Full Episode** | **150** | `refractory` | Four overlapping crises - tachyphylaxis (82%), off-med crisis (75%), dyskinesia spikes (80%), motor surges (65%). | **0.68** |

A constant 1.0 mA policy passes easy every time, fails medium on most seeds, and never passes hard. The thresholds are set exactly above that baseline - passing means the agent did something better than nothing.

### Expert Tasks

These seven tasks do not simply add more crises - each one tests a qualitatively different clinical skill. An agent that aces all three difficulty-ladder tasks but ignores patient profiles or situational cues will fail here.

**Patient-class generalisation** - same problem, different physiology

| Task | Steps | What it tests | Threshold |
|---|---:|---|---:|
| `fragile_patient` | 64 | Patient with 1.40× side-effect sensitivity. Usable amplitude range is roughly 0.3–0.8 mA - half that of medium. Jitter causes violations; timidity leaves symptoms uncontrolled. | 0.44 |
| `refractory_patient` | 120 | Brain barely responds to stimulation. Cranking the dose doesn't help; the agent needs a pulsed strategy with genuine rest periods. | 0.46 |
| `personalization_generalization` | 90 | A different patient profile every episode (all four types). The agent must read the reset metadata and adapt from step 1 - no per-patient history. | 0.50 |

**Clinical scenario reasoning** - same physiology, different situation

| Task | Steps | What it tests | Threshold |
|---|---:|---|---:|
| `exercise_bout` | 70 | Patient suddenly exercises hard. Motor surge fires with certainty in the first 30%; post-exertion dyskinesia follows with 40% probability. Zero stim during exertion is a hard failure (−0.16). | 0.55 |
| `medication_interaction` | 100 | L-DOPA wears off mid-episode → beta surges. The agent must push DBS up, but not so aggressively that the rebound dose causes dyskinesia. Timing is everything. | 0.50 |
| `nocturnal_transition` | 150 | Awake → wind-down → sleep. Beta and tremor targets tighten 20–35% in the sleep phase. Less stimulation is needed, but tighter biomarker control is required. | 0.55 |
| `surgical_followup` | 120 | First week post-implant. Hard amplitude ceiling of 0.6 mA for the first 25% of the episode (microlesion window). Any violation triggers a −0.20 penalty. | 0.50 |

---

## 3. What does each task look like in detail?

### `easy` - Calm Start

A patient just started DBS. The agent has to find a sensible stimulation level early and hold it cleanly - no events, no surprises, just early titration in a `responsive` patient with rising beta and mild tremor.

| Param | Value | Why |
|---|---|---|
| `n_steps` | 36 | ~720 ms - one programming window |
| `max_dbs_amplitude` | 1.5 mA | Moderate ceiling |
| `max_side_effect_load` | 0.55 | Forgiving; constant 1.0 mA stays well within |
| `target_beta_arv` | 0.30 | Achievable for a responsive patient |
| `success_threshold` | **0.55** | Anyone who stimulates at all should pass |

Fails if the agent stimulates not at all (−0.20 penalty), pins amplitude at the maximum (−0.14 efficiency penalty), or swings amplitude jaggedly (smoothness penalty).

A constant 1.0 mA policy scores 0.72–0.80 here. A good reactive agent reaches 0.78–0.88.

---

### `medium` - Rescue Phase

The patient is mid-deterioration. The agent walks in while symptoms are getting worse and has to bring things back without going so aggressive it triggers dyskinesia. Active symptom escalation in a `balanced` patient with the `rescue` event profile.

Two events fire stochastically: a **second deterioration wave** (55% probability) - additive beta and tremor drive in the second half of the episode, where the patient worsens just as the agent thinks it has stabilised - and **mild dyskinesia pressure** (30% probability) where the side-effect burden creeps up if the agent over-stimulates.

| Param | Value | Why |
|---|---|---|
| `n_steps` | 60 | ~1.2 s - long enough for a full rescue arc |
| `max_dbs_amplitude` | 1.8 mA | Higher ceiling for aggressive initial rescue |
| `max_side_effect_load` | 0.60 | 1.0 mA is safe; max amplitude is not |
| `success_threshold` | **0.52** | Requires measurable rescue; passive policy fails |

Fails if the agent plays it safe and lets tremor stay uncontrolled, pins the amplitude high until safety collapses around step 35, or misses the second deterioration wave entirely.

A constant 1.0 mA policy scores 0.47–0.52 (fails most seeds). A good reactive agent reaches 0.58–0.70.

---

### `hard` - Full Episode

A long session managing a difficult patient through several different crises that overlap - like running a 5-kilometre race across uneven terrain. The agent has to pace itself, avoid overreacting, and finish in a stable state. End-to-end 150-step closed-loop DBS on a `refractory` patient (weak entrainment, slow recovery, elevated adaptation gain) with the `long_horizon` event profile.

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
| `max_side_effect_load` | 0.40 | Tight - refractory patient, long session |
| `success_threshold` | **0.68** | Multi-crisis management with clean terminal stability |

The four real control dilemmas this task creates:

1. **Tachyphylaxis trap** - the same setting that worked at step 30 produces 60% less suppression at step 100. The agent must detect this from rising beta despite constant amplitude, then either back off to allow recovery, or switch to a pulsed strategy.
2. **Off-med crisis vs safety budget** - an L-DOPA trough demands more amplitude, but the safety budget is already half-spent. There's no free solution; the agent has to trade.
3. **Motor surge with neural state** - target output jumps to a high-force demand during a surge, and the agent must satisfy the motor task while still maintaining DBS appropriate for the underlying brain state.
4. **Refractory physiology** - more amplitude produces less effect than on a responsive patient, so brute force that worked on easy fails here.

A constant 1.0 mA policy scores 0.23–0.36 (never passes). A good reactive agent reaches 0.48–0.60. Reliable passing requires phase-aware adaptive control.

---

### `fragile_patient` - Tight Safety Budget

A patient with a small "safe zone" - turn it up too far and they get bad side effects fast; turn it down too far and symptoms break through. The agent has to find that narrow window and stay in it. Clinically, this represents elevated dyskinesia sensitivity (1.40×) often seen after long levodopa exposure, with a therapeutic window of roughly 0.3–0.8 mA.

| Param | Value |
|---|---|
| Patient profile | `fragile` (side_effect_sensitivity = 1.40, recovery_rate = 0.05) |
| `n_steps` | 64 |
| `max_dbs_amplitude` | 1.4 mA |
| `max_side_effect_load` | 0.26 |
| `success_threshold` | **0.44** |

The usable amplitude range is half that of the medium task. Jitter causes safety violations; timidity leaves symptoms uncontrolled.

---

### `refractory_patient` - Drug-Resistant

A patient whose brain has stopped responding well to stimulation - common after years on DBS. Cranking the dose doesn't help; the agent has to be smarter and pulse the stimulation instead. Blunted DBS response (entrainment scale = 0.88) with the same `long_horizon` event profile as `hard`, including recurring tachyphylaxis.

| Param | Value |
|---|---|
| Patient profile | `refractory` |
| `n_steps` | 120 |
| `success_threshold` | **0.46** |

Brute-force amplitude here produces similar entrainment to a moderate policy on a responsive patient - but with 1.25× higher adaptation gain and more side effects. The winning move is *pulsed* stimulation: higher amplitude during escalation, genuine rest periods during stability.

---

### `personalization_generalization` - Mixed Profiles

Each episode the patient is different. The agent has to read who's in front of it and adjust its strategy from the very first step. All four profiles (`balanced`, `responsive`, `fragile`, `refractory`) appear across episodes. The profile ID is in reset metadata; there's no per-patient prior history to lean on.

| Param | Value |
|---|---|
| `n_steps` | 90 |
| Events | `long_horizon` |
| `success_threshold` | **0.50** |

A policy specialised for responsive patients will over-stimulate fragile ones and under-treat refractory ones. Success requires reading the profile and applying profile-appropriate settings from step 1.

---

### `exercise_bout` - Exercise Burst

The patient suddenly starts exercising hard. The agent has to ramp DBS up to support the activity, then quickly dial it back down before side effects kick in. A `motor_surge` event fires with certainty in the first 30% of the episode; a post-exertion dyskinesia spike may follow with 40% probability.

| Param | Value |
|---|---|
| `n_steps` | 70 |
| Grader emphasis | force 0.22, tracking 0.22 |
| `success_threshold` | **0.55** |

Zero stimulation during the exertion period is a hard failure (−0.16 penalty).

---

### `medication_interaction` - L-DOPA Interaction

The patient's medication wears off mid-episode and symptoms surge. The agent has to push DBS up - but not so much that when the next dose kicks in, the combination causes dyskinesia. An off-med crisis fires guaranteed at steps 30–45%. If the agent over-responds, a dyskinesia spike follows in the second half (65% probability) as the next L-DOPA dose accumulates and DBS interaction builds.

| Param | Value |
|---|---|
| Patient profile | `fragile` |
| `n_steps` | 100 |
| Grader emphasis | safety 0.22, recovery 0.10 |
| `success_threshold` | **0.50** |

The dilemma is real: the correct response to an off-med crisis is *more* DBS, but over-increasing while the next dose approaches creates dyskinesia. The agent has to time its response and taper before the medication rebound.

---

### `nocturnal_transition` - Sleep Transition

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

A different operating point than waking - less stimulation, but tighter biomarker control.

---

### `surgical_followup` - Post-Implant Programming

The patient just had the device implanted last week. Post-surgical swelling is making the same dose feel stronger than it normally would, so the agent must stay below a strict ceiling for the first quarter of the session - like the speed limits in a school zone. A hard amplitude ceiling of 0.6 mA is enforced for the first 25% of the episode (the `surgical_microlesion` schedule), and impedance surges (70% probability) reduce delivered current as the electrode settles.

| Param | Value |
|---|---|
| `n_steps` | 120 |
| Grader emphasis | safety 0.30 |
| `success_threshold` | **0.50** |

Any amplitude violation in the microlesion window triggers a −0.20 penalty - the clinical equivalent of causing stimulation-induced dyskinesia in a just-implanted patient.

---

## 4. What do ten tasks tell us that one cannot?

A benchmark with a single task trains an agent to be good at exactly that task. We want to know whether the agent has learned a transferable strategy - not a memorised dose. Ten tasks across three buckets answer three progressively harder questions about what the policy actually knows.

| Bucket | Tasks | Question being asked |
|---|---|---|
| Difficulty ladder | `easy`, `medium`, `hard` | Can the agent handle increasing crisis load with the same patient family? |
| Patient generalisation | `fragile_patient`, `refractory_patient`, `personalization_generalization` | Does the strategy transfer when the patient's physiology changes? |
| Clinical scenarios | `exercise_bout`, `medication_interaction`, `nocturnal_transition`, `surgical_followup` | Can the agent recognise *what kind of situation this is* and adapt - not just react? |

Passing only the difficulty ladder means the agent learned generic control. Passing the patient tasks means it generalises. Passing the scenario tasks means it has internalised what DBS is actually for.

---

## 5. How were the success thresholds set so they actually mean something?

We didn't pick numbers that "feel right." We ran the simplest possible policy - constant dose, never adjust - and put each threshold just above what that policy could achieve. Passing therefore genuinely means the agent did something better than nothing. The thresholds are anchored to the constant 1.0 mA baseline, not to any particular LLM, so they don't move when LLMs get better.

| Tier | Threshold | Constant baseline | Why this gap |
|---|---|---|---|
| Easy | 0.55 | ≈ 0.72–0.80 | Smoke test - anyone with a basic policy clears it |
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
| `success_threshold` | Set last, after watching the baseline across many seeds - "just above what a non-strategy passes" |

---

## 7. Does the difficulty ordering actually hold?

Yes. The simple constant-dose policy was run 5 times on each task. Easy always beats medium, medium always beats hard, and hard never passes - so the difficulty labels are real, not aspirational.

Constant 1.0 mA / 0.13 ms / 130 Hz / `motor_command = target` across 5 seeds:

| Task | s1 | s2 | s3 | s4 | s5 | Min | Threshold | Passes? |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `easy` | 0.724 | 0.731 | 0.804 | 0.739 | 0.744 | **0.72** | 0.55 | Always |
| `medium` | 0.485 | 0.470 | 0.518 | 0.476 | 0.509 | **0.47** | 0.52 | Never |
| `hard` | 0.358 | 0.231 | 0.338 | 0.252 | 0.329 | **0.23** | 0.68 | Never |

A reactive LLM agent (Qwen2.5-72B, no training) does measurably better - but the hard task is still out of reach by design.

| Task | Constant baseline | Reactive LLM (72B) | Threshold | Gap to close |
|---|---:|---:|---:|---:|
| `easy` | 0.72–0.80 | 0.78–0.88 | 0.55 | Already passing |
| `medium` | 0.47–0.52 | 0.58–0.70 | 0.52 | LLM clears with correct rescue |
| `hard` | 0.23–0.36 | 0.48–0.62 | 0.68 | **~0.06–0.20 remaining** |

---

## 8. References

- Castrioto A et al. (2011). *Arch Neurol* 68(12):1550–1556.
- Fleming JE et al. (2023). *J Neural Eng* 20(5):056029.
- Little S et al. (2016). *Mov Disord* 31(8):1336–1341.
- Olanow CW et al. (2013). *Mov Disord* 19(9):997–1005.
- Tinkhauser G et al. (2017). *Brain* 140(11):2968–2981.
- Deuschl G et al. (2006). *NEJM* 355(9):896–908.
