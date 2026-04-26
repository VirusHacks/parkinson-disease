# State and Action Space - MotorAssistEnv

## 1. What is the agent looking at, and what can it touch?

At every 20 ms timestep, the agent reads a 30-dimensional observation of the patient and writes back a 4-dimensional action. The four actions shape the brain stimulator - amplitude, pulse width, and pulse frequency - plus the voluntary movement the patient is attempting. The 30 observation fields cover what the patient's brain is doing, what their body is doing, what the device is doing, and where in the episode they are.

Every observation field corresponds to a real signal a closed-loop DBS system would actually have access to - LFP power bands and impedance from the implanted device (Medtronic RC+S, Abbott Infinity, and Boston Scientific Vercise Genus all expose these), augmented with surface EMG from a wearable patch (Rosa 2015; Swann 2018). Nothing is invented for the benchmark.

## 2. What four things can the agent control?

Three of the four control the brain stimulator itself - how strong the pulse is, how long each one lasts, and how fast they come - and the fourth represents what the patient is trying to do with their body.

**The DBS triad - three knobs on a real implant**

| Action field | Range | Role | Clinical meaning |
|---|---|---|---|
| `dbs_amplitude` | 0–5 mA | Stimulation strength | Primary driver of axon recruitment and beta suppression. Task ceiling enforced at runtime. |
| `dbs_pulse_width` | 0.06–0.20 ms | Pulse duration | Wider pulses recruit axons beyond the target nucleus volume. |
| `dbs_frequency` | 60–185 Hz | Pulse-train rate | Beta suppression peaks at ~130 Hz; lower preferentially helps tremor; higher adds side effects. |

The three combine into charge per second, which drives both therapeutic effect and battery drain:

```
charge_per_pulse  (nC)   = amplitude × pulse_width
charge_per_second (mC/s) = charge_per_pulse × frequency
```

The entrainment lookup table is indexed by `(amplitude, pulse_width)`; frequency then scales the result via `_freq_beta_factor(freq)`, peaking at ~130 Hz (Kühn 2008). Parameters chosen at step *t* affect the brain state at step *t+1*, reflecting the real 15–25 ms neural response delay in LFP recordings.

**The fourth control - voluntary motor intent**

| Action field | Range | Role | Clinical meaning |
|---|---|---|---|
| `motor_command` | −1 to +1 | Voluntary motor intent | Distorted by `beta_arv` and `tremor_arv`. Better DBS → less distortion → lower task error. |

A trivially correct strategy is `motor_command = target_output` at every step. The tracking problem then reduces entirely to the DBS control problem.

## 3. What does the agent see at every timestep?

Thirty numbers across seven groups covering brain oscillation, muscle output, device delivery, 5-step trends, and episode context. All continuous signals are normalised to `[0, 1]` or `[−1, 1]` unless stated.

**A. Brain biomarkers** - what the implanted electrode reads

| Field | Signal | Clinical meaning |
|---|---|---|
| `beta_arv` | Pathological beta level | STN beta-band (13–30 Hz) amplitude, normalised to pre-DBS baseline. Primary aDBS feedback signal (Little 2013). 0 = quiet, 1 = peak pathology. |
| `tremor_arv` | Tremor intensity | Tremor envelope (3–8 Hz). Grows from ~0.01 to ~0.99 if untreated. |
| `semg_arv` | Muscle tension level | Surface EMG envelope - downstream motor consequence of STN pathology. |
| `gamma_arv` | Over-stimulation warning | High-gamma (60–90 Hz) LFP power. Elevated before `side_effect_load` builds - early warning signal (Kühn 2008). |

**B. Motor function** - what the patient's body is doing

| Field | Signal | Clinical meaning |
|---|---|---|
| `force_amplitude` | Raw muscle force | mN - healthy baseline ~59,752 mN (calibrated from Fleming). |
| `force_preserved` | Fraction of normal strength | Primary outcome variable. 1.0 = fully healthy, 0.0 = ability lost. |
| `target_output` | Intended movement | Task-defined motor target. Set `motor_command` to match this. |
| `effective_motor_output` | Actual movement achieved | Agent's command after Parkinsonian distortion is applied. |
| `task_error` | Movement error | `|target − effective|` - how far off the movement was. |
| `tracking_accuracy` | Movement accuracy | `1 − task_error / 2`, normalised to [0, 1]. |

**C. Disease summary** - convenient aggregates

| Field | Signal | Clinical meaning |
|---|---|---|
| `disease_severity` | Combined disease burden | `0.55·tremor + 0.45·beta` - high value means the patient needs more DBS. |
| `beta_suppression` | DBS effectiveness proxy | `1 − beta_arv` - convenience inverse of pathological beta. |

**D. Trends** - 5-step deltas

| Field | Signal | Clinical meaning |
|---|---|---|
| `beta_trend` | Beta trajectory direction | Rising → anticipate worsening before it shows in force. |
| `tremor_trend` | Tremor trajectory direction | Positive + high `tremor_arv` → increase amplitude now. |
| `side_effect_rate` | Side-effect accumulation rate | Positive → reduce dose. Negative → safe to push harder. |

**E. Device state** - what the implant is doing

| Field | Signal | Clinical meaning |
|---|---|---|
| `dbs_amplitude_ma` | Last delivered current | After task ceiling clip is applied. |
| `dbs_pulse_width_ms` | Last delivered pulse width | After task ceiling clip is applied. |
| `dbs_entrainment` | Brain capture fraction | Fraction of cortical collateral axons entrained. From 12×15 parameter sweep table (one-step lag). |
| `recent_dbs_avg_ma` | 5-step amplitude average | Predicts upcoming side-effect accumulation. |
| `recent_dbs_avg_pw_ms` | 5-step pulse-width average | Paired with amplitude for charge-per-second estimate. |
| `side_effect_load` | Cumulative side-effect burden | Must stay below `max_side_effect_load`. Recovers ~0.07/step at low amplitude. |
| `action_smoothness_cost` | Last-step jerk cost | Cost for abrupt amplitude/pw changes. High → slow down sweeps. |
| `dbs_constraint_violation` | Hard-limit breach fraction | How far the action exceeded the ceiling. Direct reward penalty + grader hard-failure rule. |
| `stim_washout` | Neural wash-in state | Ramps over 3–5 steps when DBS is on, decays over 5–10 steps off. Predicts entrainment ahead of time. |
| `battery_drain_rate` | Battery consumption rate | `∝ amp × pw × freq`, normalised. Long-term device-management objective. |

**F. Patient context**

| Field | Signal | Clinical meaning |
|---|---|---|
| `medication_phase` | L-DOPA cycle position | Normalised 4–6 hour on/off cycle (Nutt & Holford 1996). 0 = trough (worst), 1 = peak (best). Phase offset randomised per episode. |

**G. Episode metadata**

| Field | Signal | Clinical meaning |
|---|---|---|
| `sim_time_s` | Simulation clock | 10.02–12.00 s window, from Fleming biophysical data. |
| `task_id` | Active task name | e.g. `easy`, `hard`, `exercise_bout`. |
| `grader_score` | Final episode score | Returns `−1.0` until the episode terminates, then [0, 1]. |
| `episode_success` | Pass/fail flag | `True` if `grader_score >= task.success_threshold`. |

## 4. How does the action become the observation?

When the agent increases stimulation, the dose takes a few steps to soak in, suppresses the pathological oscillation, and lets the muscles work more normally. If the agent overdoes it or holds high amplitude too long, the brain adapts and the same dose becomes less effective. Four equations capture each piece of that physics.

**a. DBS entrainment (one-step lag)**
```
raw_entrainment    = bilinear_interp(entrainment_table[12×15], amp, pw)
freq_factor        = freq_beta_factor(freq)         # peaks at ~130 Hz
adaptation_factor  = 1 − 0.45 × adaptation_state   # diminishing returns
entrainment_t      = 0.35 × entrainment_{t−1}
                   + 0.65 × raw × profile.entrainment_scale × freq_factor × adaptation_factor
```
The 0.35/0.65 smoothing reflects the ~3-step neural response time constant observed in real STN LFP.

**b. Beta and tremor suppression**
```
base_beta_t+1   = fleming_beta[t+1] × profile.beta_scale × episode_noise_beta
target_beta     = base_beta + progressions + pressures − 0.82 × entrainment × profile.beta_responsiveness
beta_state_t+1  = 0.45 × beta_state_t + 0.55 × target_beta
```
Tremor follows the same form with a 0.50× entrainment coefficient - beta is more robustly suppressed by 130 Hz DBS than tremor (Tinkhauser 2017).

**c. Motor distortion**
```
effective_motor = motor_command
                × (1 − 0.52 × beta_state)
                × (1 − 0.30 × tremor_state)
                × (1 − 0.10 × side_effect_state)
                + noise
```
The coefficients are calibrated so that at the Fleming baseline, a patient with moderate disease retains ~65% of voluntary motor ability - consistent with UPDRS-III mild-to-moderate (Deuschl 2006).

**d. Frequency effect on side effects**
```
freq_side_factor = clamp(0.82 + 0.36 × (freq − 60) / 125)   # 0.82 at 60 Hz → 1.18 at 185 Hz
stimulation_burden *= freq_side_factor
```
Higher charge-per-second drives faster axonal fatigue and dyskinesia risk (Priori 2013).

## 5. How does the grader see the patient differently from the agent?

The agent sees noisy readings. The grader sees the true patient state. An agent cannot score well by tricking its own sensors - it has to actually treat the patient. This single design choice blocks the DeepMind grasping-task class of attack by construction.

| Aspect | Agent reads | Grader reads |
|---|---|---|
| Source variable | `_add_sensor_noise(self._beta_state)` via `_make_obs` | `self._beta_state` directly |
| Noise applied | Gaussian, σ scaled by `obs_noise_scale × ep_beta_noise` | None |
| Mutation path | Read-only after `_make_obs` returns | Read-only at episode end |
| Represents | LFP/sEMG from the implanted device | Underlying neurophysiology and motor function |

The split is also clinically realistic. Real DBS devices read noisy LFP off the same electrode they stimulate from, while clinical outcomes are measured separately by a physician with a dynamometer or UPDRS rater.

## 6. How do different patient types vary?

Four patient templates - easy to treat, average, fragile, and resistant to treatment. Each profile changes how strongly the brain responds to stimulation and how quickly side effects accumulate.

| Profile | Represents | beta_scale | tremor_scale | entrainment_scale | se_sensitivity | recovery_rate |
|---|---|---:|---:|---:|---:|---:|
| `responsive` | DBS works well, tolerates side effects | 0.98 | 0.96 | 1.08 | 0.92 | 0.08 |
| `balanced` | Typical clinical population | 1.00 | 1.00 | 1.00 | 1.00 | 0.07 |
| `fragile` | Small therapeutic window, dyskinesia-prone | 1.05 | 1.10 | 0.95 | 1.40 | 0.05 |
| `refractory` | Drug-resistant, DBS works less well | 1.08 | 1.15 | 0.88 | 1.12 | 0.06 |

The profile is revealed in reset metadata. The `personalization_generalization` task draws all four uniformly across episodes, so a policy hard-coded for one profile fails.

## 7. References

- Deuschl G et al. (2006). *NEJM* 355(9):896–908.
- Fleming JE et al. (2020). *PLOS Comput Biol* 16(8):e1008165.
- Kühn AA et al. (2008). *NeuroImage* 36(2):379–387.
- Little S et al. (2013). *Ann Neurol* 74(3):449–457.
- Nutt JG & Holford NH (1996). *Ann Neurol* 39(5):561–573.
- Priori A et al. (2013). *Exp Neurol* 245:77–86.
- Rosa M et al. (2015). *Mov Disord* 30(7):1003–1005.
- Swann NC et al. (2018). *J Neural Eng* 15(4):046006.
- Tinkhauser G et al. (2017). *Brain* 140(11):2968–2981.
