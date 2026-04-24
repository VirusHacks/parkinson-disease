# State and Action Space — MotorAssistEnv

## Design Philosophy

Every field in the observation and action space corresponds to a real physiological
signal, device parameter, or clinical diagnostic available in a closed-loop DBS
implant system. Nothing is synthetic or abstract.

The observation space mirrors what the Medtronic RC+S, Abbott Infinity, and
Boston Scientific Vercise Genus systems actually record and expose to their
adaptive DBS algorithms — LFP power bands, stimulation parameters, and impedance
measurements — augmented with the muscle-level signals that would be available
via a surface EMG telemetry patch (Rosa et al. 2015; Swann et al. 2018).

---

## Action Space — The DBS Programming Triad + Motor Command

The agent outputs **4 continuous values** at each 20 ms timestep.

### DBS Parameters (the clinical triad)

Real implantable pulse generators (Medtronic Activa PC+S, Abbott Infinity G4,
Boston Scientific Vercise Genus) expose exactly these three parameters to their
closed-loop algorithms:

| Field | Type | Range | Unit | Clinical meaning |
|---|---|---|---|---|
| `dbs_amplitude` | float | [0.0, 5.0] | mA | Stimulation current. Primary driver of axon recruitment and beta suppression. Task ceiling is enforced at runtime. |
| `dbs_pulse_width` | float | [0.06, 0.20] | ms (60–200 μs) | Pulse duration. Determines spatial spread — wider pulses recruit axons beyond the target nucleus volume. |
| `dbs_frequency` | float | [60, 185] | Hz | Pulse train frequency. Beta suppression peaks at ~130 Hz; lower frequencies preferentially suppress tremor; higher frequencies increase side-effect burden without proportional benefit. |

**How the three parameters interact:**

```
charge_per_pulse (nC) = amplitude (mA) × pulse_width (ms)
charge_per_second (mC/s) = charge_per_pulse × frequency (Hz)
```

Charge per second determines both the therapeutic effect (through axon entrainment)
and the battery drain rate. The entrainment lookup table is indexed by
`(amplitude, pulse_width)` — frequency then scales the result via
`_freq_beta_factor(freq)` which peaks at ~130 Hz (Kühn et al. 2008).

**One-step clinical lag:** The DBS parameters chosen at step t affect the brain
state observed at step t+1, not t. This reflects the real 15–25 ms neural response
delay observed in LFP recordings after DBS parameter changes.

### Motor Command

| Field | Type | Range | Clinical meaning |
|---|---|---|---|
| `motor_command` | float | [-1.0, 1.0] | Voluntary motor intent. The environment distorts this proportionally to `beta_arv` and `tremor_arv`. Better DBS → less distortion → lower `task_error`. |

**Strategy:** set `motor_command = target_output` every step (visible in the
observation). The tracking problem reduces to the DBS control problem: better
symptom suppression automatically reduces motor distortion.

---

## Observation Space — 27 Fields

The agent observes a structured state vector at each 20 ms timestep.
All continuous signals are normalised to [0, 1] or [-1, 1] unless otherwise noted.

### A. Primary Neural Biomarkers

These are the signals a real aDBS system reads from the implanted electrode:

| Field | Range | Clinical meaning |
|---|---|---|
| `beta_arv` | [0, 1] | STN beta-band (13–30 Hz) oscillation amplitude, normalised to the pre-DBS baseline. 0 = fully suppressed, 1 = peak Parkinson's pathology. The primary aDBS feedback signal (Little et al. 2013). |
| `tremor_arv` | [0, 1] | Tremor envelope (3–8 Hz band), normalised. Grows from ~0.01 to ~0.99 across the episode as disease state worsens without treatment. |
| `semg_arv` | [0, 1] | Surface EMG envelope, reflecting the downstream motor consequence of STN pathology. Correlated with beta + tremor state. |
| `gamma_arv` | [0, 1] | High-gamma (60–90 Hz) LFP power. Elevated gamma indicates over-stimulation — the brain's response to excessive DBS drive (Kühn et al. 2008). Use as an early warning before `side_effect_load` builds. |

### B. Motor Function Signals

These would be measured via dynamometry and surface EMG in a clinical system:

| Field | Range | Clinical meaning |
|---|---|---|
| `force_amplitude` | [0, ∞) mN | Raw simulated muscle force. Healthy baseline = 59,752 mN (calibrated from Fleming et al. simulation). |
| `force_preserved` | [0, 1] | Fraction of healthy force currently produced. 1.0 = fully healthy, 0.0 = complete motor loss. **Primary outcome variable.** |
| `target_output` | [-1, 1] | Task-defined motor target for this episode (e.g., "hold a cup at 40% effort"). Set `motor_command` to this value every step. |
| `effective_motor_output` | [-1, 1] | Agent's motor command after Parkinsonian distortion. Approaches `target_output` as DBS improves. |
| `task_error` | [0, 2] | `|target_output - effective_motor_output|`. Minimise by: (1) setting `motor_command = target_output`, (2) suppressing disease state via DBS. |
| `tracking_accuracy` | [0, 1] | `1 - task_error / 2`. Normalised tracking quality. |

### C. Derived Disease Summary (convenience aggregates)

These are computed from primary signals — they are not independent measurements:

| Field | Formula | Use |
|---|---|---|
| `disease_severity` | `0.55 * tremor_arv + 0.45 * beta_arv` | Combined disease burden proxy. High values indicate the patient needs more DBS. |
| `beta_suppression` | `1 - beta_arv` | Convenience inverse. 1.0 = fully suppressed. |

### D. Temporal Trend Signals

5-step deltas — tell the agent whether the situation is improving or worsening:

| Field | Range | Clinical meaning |
|---|---|---|
| `beta_trend` | [-1, 1] | `beta_arv[t] - beta_arv[t-1]`. Negative = beta falling (improving). Use to anticipate trajectory before it appears in force. |
| `tremor_trend` | [-1, 1] | Tremor direction. Negative = tremor falling. Positive with high `tremor_arv` → increase amplitude now. |
| `side_effect_rate` | [-1, 1] | `side_effect_load` direction. Positive = load building → reduce amplitude or frequency. Negative = recovering → safe to push harder. |

### E. DBS Device State

What a real IPG telemetry readout would show:

| Field | Range | Clinical meaning |
|---|---|---|
| `dbs_amplitude_ma` | [0, 5] mA | Delivered amplitude last step (after task ceiling clip). |
| `dbs_pulse_width_ms` | [0.06, 0.20] ms | Delivered pulse width last step. |
| `dbs_entrainment` | [0, 1] | Fraction of cortical collateral axons entrained by current DBS settings. Derived from the 12×15 parameter sweep table (one-step lag). |
| `recent_dbs_avg_ma` | [0, 5] mA | 5-step rolling mean of amplitude. Predicts upcoming side-effect accumulation. |
| `recent_dbs_avg_pw_ms` | [0.06, 0.20] ms | 5-step rolling mean of pulse width. |
| `side_effect_load` | [0, 1] | Cumulative side-effect proxy. Must remain below task `max_side_effect_load`. Increases with charge delivery, decreases during low-amplitude periods (`recovery_rate ≈ 0.07/step`). |
| `action_smoothness_cost` | [0, 1] | Cost for abrupt parameter changes this step. High values indicate jagged control — a signal to slow down parameter sweeps. |
| `dbs_constraint_violation` | [0, 1] | Fraction by which the action exceeded the task amplitude or pulse-width ceiling. Incurs a direct reward penalty and contributes to grader hard-failure rules. |
| `stim_washout` | [0, 1] | Neural wash-in state. Ramps up over 3–5 steps when DBS is on, decays over 5–10 steps when reduced. Predicts upcoming entrainment before it appears in `dbs_entrainment`. Use for proactive control. |
| `battery_drain_rate` | [0, 1] | Instantaneous IPG battery consumption: `∝ amplitude × pulse_width × frequency`. Normalised to max possible. Long-term device management objective. |

### F. Patient Physiology Signals

Context about the patient state that a wearable patch and medication log would provide:

| Field | Range | Clinical meaning |
|---|---|---|
| `medication_phase` | [0, 1] | Normalised L-DOPA cycle position. Parkinson's patients on oral levodopa experience 4–6 hour on/off cycles (Nutt & Holford 1996). 0.0 = medication trough (worst state, need more DBS). 1.0 = medication peak (best state, reduce DBS to avoid dyskinesia). Phase offset is randomised per episode. |

### G. Task and Episode Metadata

| Field | Type | Meaning |
|---|---|---|
| `sim_time_s` | float | Simulation time in seconds (10.02–12.00 s, from the Fleming trajectory). |
| `task_id` | str | Active task: `beta_suppression`, `tremor_correction`, `full_episode`, etc. |
| `grader_score` | float | Final deterministic score in [0, 1]. Value is -1.0 until episode end. |
| `episode_success` | bool | True if `grader_score >= task.success_threshold`. |

---

## Physical Model — How Action Becomes Observation

### 1. DBS Entrainment (One-Step Lag)

```
raw_entrainment = bilinear_interp(entrainment_table[12×15], amp, pw)
freq_factor      = freq_beta_factor(freq)          # peaks at ~130 Hz
adaptation_factor = 1 - 0.45 * adaptation_state   # diminishing returns
entrainment_t   = 0.35 * entrainment_{t-1}
                + 0.65 * raw * profile.entrainment_scale * freq_factor * adaptation_factor
```

The 0.35/0.65 exponential smoothing reflects the ~3-step neural response time constant
observed in STN LFP recordings after DBS parameter changes.

### 2. Beta and Tremor Suppression

```
base_beta_t+1  = fleming_beta[t+1] * profile.beta_scale * episode_noise_beta
target_beta    = base_beta + progressions + pressures - 0.82 * entrainment * profile.beta_responsiveness
beta_state_t+1 = 0.45 * beta_state_t + 0.55 * target_beta
```

Tremor follows analogously with a 0.50× entrainment suppression coefficient
(beta is more robustly suppressed by 130 Hz DBS than tremor — Tinkhauser et al. 2017).

### 3. Motor Distortion

```
noise = U(-1, 1) * 0.10 * motor_noise_scale * (0.35 + tremor_state)
effective_motor = motor_command
                * (1 - 0.52 * beta_state)
                * (1 - 0.30 * tremor_state)
                * (1 - 0.10 * side_effect_state)
                + noise
```

The 0.52 beta coefficient and 0.30 tremor coefficient are calibrated so that at
the Fleming trajectory baseline (beta≈0.40, tremor≈0.10), a patient with moderate
disease retains ~65% of voluntary motor ability — consistent with UPDRS-III motor
scores in mild-to-moderate Parkinson's (Deuschl et al. 2006).

### 4. Frequency Effect on Side Effects

```
freq_side_factor = clamp(0.82 + 0.36 * (freq - 60) / 125)
# 0.82 at 60 Hz → 1.18 at 185 Hz
stimulation_burden *= freq_side_factor
```

This reflects the higher charge-per-second at elevated frequency driving faster
axonal fatigue and increased dyskinesia risk (Priori et al. 2013).

---

## Partial Observability

The agent does **not** observe:

| Hidden information | Why hidden | Clinical analogue |
|---|---|---|
| Ground-truth Fleming DBS settings | Would trivialise the problem (optimal answer is in metadata for debugging only) | Real clinicians cannot know the "correct" amplitude a priori |
| Raw STN spike trains from individual neurons | Available in the simulation but not via LFP recording | Spiking activity cannot be recorded chronically with current implants |
| The patient's intended voluntary target (only the normalised `target_output` context is given) | Clinicians cannot directly observe patient intent | Inferred from motor task performance |
| Future disease trajectory | No lookahead into the Fleming timeline | Clinicians observe current state only |
| Episode noise factors | Hidden per-episode scaling to prevent memorisation | Real patients vary across sessions |

---

## Patient Profile Variation

Four patient profiles scale disease dynamics and DBS response:

| Profile | beta_scale | tremor_scale | entrainment_scale | side_effect_sensitivity | recovery_rate |
|---|---|---|---|---|---|
| responsive | 0.98 | 0.96 | 1.08 | 0.92 | 0.08 |
| balanced | 1.00 | 1.00 | 1.00 | 1.00 | 0.07 |
| fragile | 1.05 | 1.10 | 0.95 | 1.40 | 0.05 |
| refractory | 1.08 | 1.15 | 0.88 | 1.12 | 0.06 |

Profile is revealed in the reset metadata. The `personalization_generalization` task
requires the agent to adapt its strategy based on the profile without prior experience
on that specific patient — testing generalisation rather than specialisation.

---

## References

- Deuschl G et al. (2006). *NEJM* 355(9):896–908.
- Fleming JE et al. (2020). *PLOS Comput Biol* 16(8):e1008165.
- Kühn AA et al. (2008). *NeuroImage* 36(2):379–387.
- Little S et al. (2013). *Ann Neurol* 74(3):449–457.
- Nutt JG & Holford NH (1996). "The response to levodopa in Parkinson's disease." *Ann Neurol* 39(5):561–573.
- Priori A et al. (2013). *Exp Neurol* 245:77–86.
- Rosa M et al. (2015). *Mov Disord* 30(7):1003–1005.
- Swann NC et al. (2018). *J Neural Eng* 15(4):046006.
- Tinkhauser G et al. (2017). *Brain* 140(11):2968–2981.
