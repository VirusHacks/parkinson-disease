# State and Action Space — MotorAssistEnv

## 1. What is the agent looking at, and what can it touch?

At every 20 ms timestep, the agent reads a 30-dimensional observation of the patient and writes back a 4-dimensional action. The four actions shape the brain stimulator — amplitude, pulse width, and pulse frequency — plus the voluntary movement the patient is attempting. The 30 observation fields cover what the patient's brain is doing, what their body is doing, what the device is doing, and where in the episode they are.

Every observation field corresponds to a real signal a closed-loop DBS system would actually have access to — LFP power bands and impedance from the implanted device (Medtronic RC+S, Abbott Infinity, and Boston Scientific Vercise Genus all expose these), augmented with surface EMG from a wearable patch (Rosa 2015; Swann 2018). Nothing is invented for the benchmark. The sections below walk through the four controls, the thirty observations, the biophysics that maps one to the other, and — critically — what the agent is *not* allowed to see and why.

---

## 2. What four things can the agent control?

Three of the four control the brain stimulator itself — how strong the pulse is, how long each one lasts, and how fast they come — and the fourth represents what the patient is *trying* to do with their body.

### The DBS triad — the three knobs on a real implant

Real implantable pulse generators expose exactly these three parameters to their closed-loop algorithms:

| Action field | Range | What it does | Clinical meaning |
|---|---|---|---|
| `dbs_amplitude` | 0–5 mA | How strong the zap is | Stimulation current — primary driver of axon recruitment and beta suppression. Task ceiling enforced at runtime. |
| `dbs_pulse_width` | 0.06–0.20 ms | How long each zap lasts | Pulse duration — wider pulses recruit axons beyond the target nucleus volume |
| `dbs_frequency` | 60–185 Hz | How fast the zaps come | Pulse-train frequency — beta suppression peaks at ~130 Hz; lower preferentially helps tremor; higher just adds side effects |

The three combine via charge:
```
charge_per_pulse  (nC)   = amplitude × pulse_width
charge_per_second (mC/s) = charge_per_pulse × frequency
```
Charge per second drives both therapeutic effect (axon entrainment) and battery drain. The entrainment lookup table is indexed by `(amplitude, pulse_width)`; frequency then scales the result via `_freq_beta_factor(freq)`, peaking at ~130 Hz (Kühn 2008).

There's a one-step clinical lag: parameters chosen at step *t* affect the brain state observed at step *t+1*, which reflects the real 15–25 ms neural response delay observed in LFP recordings.

### The fourth control — voluntary motor intent

| Action field | Range | What it does | Clinical meaning |
|---|---|---|---|
| `motor_command` | -1 to +1 | What the patient is *trying* to do with their hand | Voluntary motor intent — the environment distorts this proportionally to `beta_arv` and `tremor_arv`. Better DBS → less distortion → lower task error. |

A trivially correct strategy is `motor_command = target_output` at every step (the target is in the observation). The tracking problem then reduces to the DBS control problem.

---

## 3. What does the agent see at every timestep?

A dashboard of about thirty numbers covering: how the patient's brain is oscillating, how strong their muscles are firing, what the device is currently delivering, whether things are getting better or worse over the last five steps, and a few pieces of context like the medication phase. All continuous signals are normalised to [0, 1] or [-1, 1] unless stated.

### A. Brain biomarkers (what the implanted electrode reads)

| Field | What it tells the agent | Clinical meaning |
|---|---|---|
| `beta_arv` | "How bad is the Parkinson's right now?" 0 = quiet, 1 = peak pathology | STN beta-band (13–30 Hz) oscillation amplitude, normalised to pre-DBS baseline. Primary aDBS feedback signal (Little 2013). |
| `tremor_arv` | "How much is the patient shaking?" | Tremor envelope (3–8 Hz). Grows from ~0.01 to ~0.99 if untreated. |
| `semg_arv` | "How tense are the muscles?" | Surface EMG envelope — downstream motor consequence of STN pathology |
| `gamma_arv` | "Are we overdoing the stimulation?" | High-gamma (60–90 Hz) LFP power. Elevated → over-stimulation. Early warning *before* `side_effect_load` builds (Kühn 2008). |

### B. Motor function (what the patient's body is doing)

| Field | What it tells the agent | Clinical meaning |
|---|---|---|
| `force_amplitude` | Raw muscle force | mN — healthy baseline ~59,752 mN (calibrated from Fleming) |
| `force_preserved` | "How much of the patient's normal strength is left?" 1.0 = healthy, 0.0 = lost | Fraction of healthy force currently produced. **Primary outcome variable.** |
| `target_output` | "What movement does the patient want to make?" | Task-defined motor target. Set `motor_command` to this. |
| `effective_motor_output` | "What actually happened, after Parkinson's interfered" | Agent's command after Parkinsonian distortion |
| `task_error` | "How far off was the movement?" | `|target − effective|` |
| `tracking_accuracy` | "How well did the patient hit the target?" 0–1 | `1 − task_error / 2` |

### C. Disease summary (convenient aggregates)

| Field | Formula | Use |
|---|---|---|
| `disease_severity` | `0.55·tremor + 0.45·beta` | Combined disease burden — high → patient needs more DBS |
| `beta_suppression` | `1 − beta_arv` | Convenience inverse |

### D. Trends (5-step deltas — is the situation improving?)

| Field | What it tells the agent | Clinical use |
|---|---|---|
| `beta_trend` | "Is the bad oscillation rising or falling?" | Anticipate trajectory before it appears in force |
| `tremor_trend` | "Is the shaking getting worse?" | Positive + high `tremor_arv` → increase amplitude now |
| `side_effect_rate` | "Are side effects building up?" | Positive → reduce. Negative → safe to push. |

### E. Device state (what the implant is doing)

| Field | What it tells the agent | Clinical meaning |
|---|---|---|
| `dbs_amplitude_ma` | Last delivered current | After task ceiling clip |
| `dbs_pulse_width_ms` | Last delivered pulse width | — |
| `dbs_entrainment` | "How much of the brain is the device actually capturing?" | Fraction of cortical collateral axons entrained. From 12×15 parameter sweep table (one-step lag). |
| `recent_dbs_avg_ma` | "How aggressive has the dose been recently?" | 5-step rolling mean — predicts upcoming side-effect accumulation |
| `recent_dbs_avg_pw_ms` | Same, for pulse width | — |
| `side_effect_load` | "How close are we to the danger zone?" | Cumulative side-effect proxy. Must stay below `max_side_effect_load`. Recovers at ~0.07/step during low-amp periods. |
| `action_smoothness_cost` | "Was that last change too jerky?" | Cost for abrupt changes. High → slow down sweeps. |
| `dbs_constraint_violation` | "Did we cross a hard limit?" | Fraction by which action exceeded amp/pw ceiling. Direct reward penalty + grader hard-failure rule. |
| `stim_washout` | "How much of the stim is currently 'soaked in'?" | Neural wash-in state. Ramps over 3–5 steps when DBS on, decays over 5–10 steps off. Predicts entrainment **before** it shows. |
| `battery_drain_rate` | "How fast is the device's battery being burned?" | `∝ amp × pw × freq`, normalised. Long-term device-management objective. |

### F. Patient context

| Field | What it tells the agent | Clinical meaning |
|---|---|---|
| `medication_phase` | "Where are we in the medication cycle?" 0 = trough (worst), 1 = peak (best) | Normalised L-DOPA cycle (4–6 hour on/off, Nutt & Holford 1996). Phase offset randomised per episode. |

### G. Episode metadata

| Field | Type | Meaning |
|---|---|---|
| `sim_time_s` | float | Simulation time (10.02–12.00 s, from Fleming) |
| `task_id` | str | Active task (`easy`, `hard`, `exercise_bout`, etc.) |
| `grader_score` | float | Final score in [0, 1]. `-1.0` until episode ends. |
| `episode_success` | bool | True if `grader_score >= task.success_threshold` |

---

## 4. How does the action become the observation?

When the agent turns up the stimulation, it doesn't immediately change the brain. The dose takes a few steps to soak in, then it suppresses the bad oscillation, which lets the muscles work more normally. But if the agent overdoes it or holds high amplitude too long, the brain adapts and the same dose stops working as well. The four equations below capture each of those four pieces of physics.

### a. DBS entrainment (one-step lag)
```
raw_entrainment    = bilinear_interp(entrainment_table[12×15], amp, pw)
freq_factor        = freq_beta_factor(freq)        # peaks at ~130 Hz
adaptation_factor  = 1 − 0.45 × adaptation_state   # diminishing returns
entrainment_t      = 0.35 × entrainment_{t−1}
                   + 0.65 × raw × profile.entrainment_scale × freq_factor × adaptation_factor
```
The 0.35/0.65 smoothing reflects the ~3-step neural response time constant observed in real STN LFP.

### b. Beta and tremor suppression
```
base_beta_t+1   = fleming_beta[t+1] × profile.beta_scale × episode_noise_beta
target_beta     = base_beta + progressions + pressures − 0.82 × entrainment × profile.beta_responsiveness
beta_state_t+1  = 0.45 × beta_state_t + 0.55 × target_beta
```
Tremor follows the same form with a 0.50× entrainment coefficient — beta is more robustly suppressed by 130 Hz DBS than tremor (Tinkhauser 2017).

### c. Motor distortion
```
effective_motor = motor_command
                × (1 − 0.52 × beta_state)
                × (1 − 0.30 × tremor_state)
                × (1 − 0.10 × side_effect_state)
                + noise
```
The coefficients are calibrated so that at the Fleming baseline, a patient with moderate disease retains ~65% of voluntary motor ability — consistent with UPDRS-III mild-to-moderate (Deuschl 2006).

### d. Frequency effect on side effects
```
freq_side_factor = clamp(0.82 + 0.36 × (freq − 60) / 125)   # 0.82 at 60 Hz → 1.18 at 185 Hz
stimulation_burden *= freq_side_factor
```
Higher charge-per-second drives faster axonal fatigue and dyskinesia risk (Priori 2013).

---

## 5. What does the agent NOT see, and why?

The agent sees what a real doctor with a brain implant readout and a wearable patch would see — but not the answer key. It can't peek at what the simulation will do next, can't see the random noise applied this episode, and can't see exactly what makes this patient different from the average. This forces it to react in real time, the way a clinician actually has to.

Some things are hidden because they would let the agent peek at the answer:

| Hidden | Why hidden | Clinical analogue |
|---|---|---|
| Ground-truth Fleming DBS settings | The "optimal answer" is in metadata for debugging only; would trivialise the problem | Real clinicians can't know the correct amplitude in advance |
| Raw STN spike trains | Available in simulation but not via LFP recording | Spiking activity can't be recorded chronically |
| Patient's true voluntary intent (only normalised `target_output` shown) | Clinicians can't directly observe intent | Inferred from task performance |
| Future disease trajectory | No lookahead | Clinicians observe current state only |
| Per-episode noise factors | Hidden per-episode scaling | Real patients vary across sessions |

Other things are hidden specifically to block reward hacking:

| Hidden | If exposed, the attack would be |
|---|---|
| Per-episode noise factors (`ep_beta_noise`, `ep_tremor_noise`, `ep_force_noise`, `ep_semg_noise`) | Pre-compute optimal compensation per episode |
| Future event schedule | Pre-baked schedules instead of reactive control |
| Next episode's `target_output` | Hard-code a tracking target |
| True L-DOPA pharmacokinetic phase | Bypass the medication-interaction reasoning |
| Underlying patient profile parameters | Hard-code per-patient amplitude policies |

The rule the design follows: the observation is rich enough to act (30 fields covering brain, body, device, trends, context) but too poor to plan around the grader. Anything a sufficiently powerful agent could use to precompute the optimal trajectory is hidden; everything a real clinician would have access to is exposed.

---

## 6. How does the grader see the patient differently from the agent?

The agent sees noisy readings — like reading a scale that wobbles. The grader sees the actual, true patient state. So an agent can't get a high score by tricking its own sensors; it has to actually treat the patient. This is the single design choice that does the most anti-hacking work in the entire environment, and it blocks the DeepMind grasping-task class of attack — where an agent learns to occlude the camera so the evaluator can't tell whether the grasp succeeded — by construction.

| Aspect | What the agent reads | What the grader reads |
|---|---|---|
| Source variable | `_add_sensor_noise(self._beta_state)` in `_make_obs` | `self._beta_state` directly |
| Noise applied | Gaussian, σ scaled by `obs_noise_scale × ep_beta_noise` | None |
| Mutation path from agent action | Read-only after `_make_obs` returns | Read-only at episode end |
| What it represents | LFP/sEMG recording from the implanted device | Underlying neurophysiology and motor function |

The split is also clinically realistic. Real DBS devices read noisy LFP off the same electrode they stimulate from, while clinical outcomes are measured separately by a physician with a dynamometer or UPDRS rater. The signal you control with and the outcome you're judged on are physically distinct.

---

## 7. How do different patient types vary?

Four patient templates — easy to treat, average, fragile, and resistant to treatment. Each has different scaling factors that change how the body responds to stimulation.

| Profile | What it represents | beta_scale | tremor_scale | entrainment_scale | side_effect_sensitivity | recovery_rate |
|---|---|---|---|---|---|---|
| `responsive` | Easy patient — DBS works well, tolerates side effects | 0.98 | 0.96 | 1.08 | 0.92 | 0.08 |
| `balanced` | Average patient — the typical clinical population | 1.00 | 1.00 | 1.00 | 1.00 | 0.07 |
| `fragile` | Sensitive patient — small therapeutic window | 1.05 | 1.10 | 0.95 | 1.40 | 0.05 |
| `refractory` | Drug-resistant — DBS works less well, needs creative strategy | 1.08 | 1.15 | 0.88 | 1.12 | 0.06 |

The profile is revealed in reset metadata. The `personalization_generalization` task draws all four uniformly across episodes, so a policy hard-coded for one profile fails.

---

## 8. References

- Deuschl G et al. (2006). *NEJM* 355(9):896–908.
- Fleming JE et al. (2020). *PLOS Comput Biol* 16(8):e1008165.
- Kühn AA et al. (2008). *NeuroImage* 36(2):379–387.
- Little S et al. (2013). *Ann Neurol* 74(3):449–457.
- Nutt JG & Holford NH (1996). *Ann Neurol* 39(5):561–573.
- Priori A et al. (2013). *Exp Neurol* 245:77–86.
- Rosa M et al. (2015). *Mov Disord* 30(7):1003–1005.
- Swann NC et al. (2018). *J Neural Eng* 15(4):046006.
- Tinkhauser G et al. (2017). *Brain* 140(11):2968–2981.
