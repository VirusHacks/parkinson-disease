# Reward Design — MotorAssistEnv

## 1. Clinical Grounding

Every reward term maps to a published clinical endpoint or documented DBS failure mode. The reward is not heuristic — it is a computational translation of what neurologists optimise during programming sessions.

| Reward component | Clinical basis |
|---|---|
| `force_preserved` | Primary functional outcome in DBS trials (Limousin et al. 1995; Deuschl et al. 2006) |
| `beta_arv` suppression | Beta-band power as closed-loop DBS feedback target (Little et al. 2013, 2016) |
| `tremor_arv` suppression | Tremor ARV as secondary biomarker (Tinkhauser et al. 2017) |
| `side_effect_load` budget | Stimulation-induced dyskinesia threshold (Rosa et al. 2015; Swann et al. 2018) |
| `efficiency` (minimum effective amplitude) | Battery longevity in implanted pulse generators (Priori et al. 2013); adaptive DBS reduces stimulation time by 56% while matching outcomes (Little et al. 2016) |
| `smoothness_cost` | Abrupt parameter changes cause transient dyskinesia (Velisar et al. 2019) |
| `terminal_stability` | Sustained benefit at episode end — prevents early-episode gaming |

---

## 2. Per-Step Dense Reward

At every 20 ms timestep the agent receives:

```
r_t = w_force  * force_preserved_t
    + w_track  * tracking_accuracy_t
    + w_beta   * (1 − beta_arv_t)
    + w_tremor * (1 − tremor_arv_t)
    + w_safety * safety_t
    + w_smooth * (1 − smoothness_cost_t)
    + w_eff    * efficiency_t
    + shaping_t                         ← small phase-aware bonus on hard/medium
    − 0.08 * constraint_violation_t
```

where:
- `safety_t = clamp(1 − side_effect_load_t / max_side_effect_load)`
- `efficiency_t = 0.65 * (1 − amp/amp_max) + 0.35 * (1 − pw_norm)` — gated by therapeutic engagement
- `shaping_t` — late-episode terminal stability proxy (hard: final 25% only; medium: recovery proxy from step ~50%)

**Weights are task-specific** so the training signal is aligned with the evaluation grader:

| Weight | `easy` | `medium` | `hard` |
|---|---|---|---|
| `w_beta` | 0.30 | 0.06 | **0.22** |
| `w_tremor` | 0.18 | 0.14 | **0.14** |
| `w_force` | 0.16 | 0.16 | 0.14 |
| `w_track` | 0.12 | 0.16 | 0.16 |
| `w_safety` | 0.14 | **0.22** | 0.18 |
| `w_smooth` | 0.05 | 0.04 | 0.04 |
| `w_eff` | 0.05 | 0.08 | 0.04 |

**Why hard's weights changed from the naive design:** The original hard reward had `w_safety = 0.36`. This created a training incentive to stimulate conservatively — a constant 1.0 mA policy (amp_norm = 0.42) accumulated almost no side effects, so `safety_t ≈ 0.90` contributed 0.32 per step. An agent could reach average reward ~0.55 by doing very little, which is clinically meaningless. The current weights make beta and tremor suppression the primary signal, matching the actual goal of DBS.

---

## 3. Episode-End Grader Score (Deterministic 0.0–1.0)

The grader is the authoritative benchmark signal — computed once at episode end from the full trajectory. It is deterministic given the same trajectory.

### Score Components

**`beta_score`**
```
beta_score = 0.55 * weighted_mean(1 − beta_arv)
           + 0.45 * fraction_of_steps(beta_arv ≤ target_beta_arv)
```
Combines mean suppression depth with time-in-therapeutic-range (TTR), mirroring the dual metric used in aDBS trials (Tinkhauser et al. 2017). The TTR term (fraction of steps below target) is critical — an agent that briefly dips below the target but averages poorly still scores low.

**`tremor_score`**
```
tremor_score = 0.60 * weighted_mean(1 − tremor_arv)
             + 0.40 * fraction_of_steps(tremor_arv ≤ target_tremor_arv)
```

**`force_score`**
```
force_score = weighted_mean(force_preserved) / target_force_preserved, clamped [0, 1]
```
Early steps are weighted ~1.35× relative to terminal steps — mirrors clinical priority of maintaining immediate motor function.

**`tracking_score`**
```
tracking_score = 0.45 * weighted_mean(1 − task_error / target_tracking_error)
               + 0.55 * weighted_mean(tracking_accuracy)
```
Voluntary motor task performance — the patient's ability to execute intended movements is the primary reason for DBS implantation (Limousin et al. 1995).

**`safety_score`**
```
overload_per_step = max(0, (side_effect_load − budget) / (1 − budget))
safety_score = clamp(1 − (0.45 * mean_overload + 0.35 * peak_overload + 0.20 * mean_violation) * 1.8)
```
Peak overload is weighted separately from mean — stimulation-induced dyskinesia emerges as a threshold effect where a single brief overload can be as harmful as sustained moderate overload (Swann et al. 2018).

**`efficiency_score`**
```
efficiency_score = (0.65*(1 − mean_amp/max_amp) + 0.35*(1 − mean_pw_norm))
                 * therapeutic_engagement
```
where `therapeutic_engagement = 0.40*force_score + 0.30*beta_score + 0.30*tremor_score`.

Gated by therapeutic engagement to prevent reward for doing nothing — zero-DBS has perfect amplitude efficiency but scores zero because force/beta/tremor all collapse.

**`smoothness_score`**
```
smoothness_score = 1 − mean(smoothness_cost_per_step)
```
Abrupt amplitude changes cause transient dyskinesia even within safe total dose (Velisar et al. 2019).

**`terminal_stability_score`**
```
Computed on the final 5 steps only.
terminal_stability = 0.45*(last_force/target) + 0.30*(1 − last_tremor/target) + 0.25*(1 − last_error/target)
```
Prevents strategies that front-load good performance then degrade.

**`recovery_score`**
```
Compares first-6-step window vs last-8-step window for force, tremor, and tracking.
recovery_score = 0.40*force_recovery + 0.40*tremor_recovery + 0.20*tracking_recovery
```
Key for the medium (rescue) task: did the agent actually rescue the patient from escalation, or just survive?

---

## 4. Task-Specific Grader Weights

### easy — Calm Start
```
0.30 * beta_score + 0.18 * tremor_score + 0.16 * force_score
+ 0.12 * tracking_score + 0.14 * safety_score
+ 0.05 * smoothness_score + 0.05 * efficiency_score
```
Beta suppression dominates because the easy task's primary clinical goal is establishing initial control — the agent must learn that DBS actually suppresses pathological activity.

### medium — Rescue Phase
```
0.22 * safety_score + 0.16 * force_score + 0.16 * tracking_score
+ 0.14 * tremor_score + 0.08 * terminal_stability_score + 0.08 * efficiency_score
+ 0.06 * beta_score + 0.06 * recovery_score + 0.04 * smoothness_score
```
Safety is primary because rescue without managing the dyskinesia risk is clinically unacceptable. Recovery is scored explicitly — the agent must improve the patient state, not just maintain mediocrity.

### hard — Full Episode
```
0.22 * beta_score + 0.18 * safety_score + 0.16 * tracking_score
+ 0.14 * tremor_score + 0.14 * force_score + 0.08 * terminal_stability_score
+ 0.04 * smoothness_score + 0.04 * efficiency_score
```
Beta and tremor together carry 0.36 weight — this is the task where the agent must demonstrate that DBS is actually working therapeutically. Low-stim coasting can no longer game the safety term into a passing grade.

### expert tasks — scenario graders

| Task | Primary weights |
|---|---|
| `fragile_patient` | safety 0.28, tracking 0.18, force 0.18 — therapeutic window precision |
| `refractory_patient` | force 0.18, tracking 0.14, tremor 0.12 — functional outcome despite weak response |
| `personalization_generalization` | force 0.18, tracking 0.18, safety 0.18 — balanced across all profiles |
| `exercise_bout` | force 0.22, tracking 0.22 — motor performance during high-demand bout |
| `medication_interaction` | safety 0.22, recovery 0.10 — crisis management without over-treatment |
| `nocturnal_transition` | safety 0.22, terminal_stability 0.12, efficiency 0.12 — sleep-phase stability |
| `surgical_followup` | safety 0.30 — microlesion window constraint dominates |

---

## 5. Hard-Failure Penalties

Applied on top of the weighted score. These model clinically unacceptable outcomes that a smooth component score might otherwise undervalue.

### Universal (all tasks)

| Condition | Penalty | Clinical basis |
|---|---|---|
| `safety_score < 0.20` | −0.12 | Sustained dyskinesia risk; unsafe stimulation |
| `tracking_score < 0.20` | −0.08 | Patient cannot execute voluntary movement |
| `beta_score < 0.40` | −0.06 | DBS providing no measurable beta suppression |
| `tremor_score < 0.22` | −0.05 | Active tremor uncontrolled |
| `force_score < 0.55` | −0.04 | Severe motor function loss |
| `safety_score == 0.0` (medium/hard) | −0.10 | Complete safety budget exhaustion |
| `mean_constraint_violation > 0.20` (medium/hard) | −0.08 | Repeated device limit violations |

### Task-specific

| Task | Condition | Penalty |
|---|---|---|
| easy | zero-stim + poor suppression | −0.20 |
| easy | constant max-amp + low efficiency | −0.14 |
| medium | `tremor_score < 0.20` | −0.10 |
| medium | zero-stim + poor tremor/recovery | −0.14 |
| **hard** | `beta_score < 0.30` | **−0.10** |
| **hard** | `tremor_score < 0.25` | **−0.06** |
| **hard** | `terminal_stability_score < 0.25` | **−0.08** |
| hard | high mean amp + poor efficiency | −0.08 |
| exercise_bout | `tracking_score < 0.55` | −0.14 |
| exercise_bout | `force_score < 0.60` | −0.10 |
| exercise_bout | zero-stim during exertion | −0.16 |
| medication_interaction | `recovery_score < 0.40` | −0.12 |
| medication_interaction | high trailing amp + poor safety | −0.10 |
| nocturnal_transition | `terminal_stability_score < 0.45` | −0.12 |
| nocturnal_transition | low efficiency + poor safety | −0.08 |
| surgical_followup | amplitude violations in first 25% | **−0.20** |
| surgical_followup | `safety_score < 0.55` | −0.08 |
| surgical_followup | `recovery_score < 0.30` | −0.06 |

The final grader score is `clamp(weighted_sum − total_penalty, 0.0, 1.0)`.

---

## 6. Anti-Hacking Analysis

The environment is explicitly designed to block RL shortcuts:

| Exploit strategy | Blocking mechanism |
|---|---|
| Max amplitude every step | `efficiency_score` penalises mean amp; side-effect budget depletes `safety_score`; hard-specific over-treatment penalty |
| Zero stimulation | `beta_score`, `tremor_score` collapse; hard-failure penalty −0.20 for no-DBS + symptoms |
| Front-load good steps, degrade at end | `terminal_stability_score` grades only the final 5 steps; −0.08 penalty if below 0.25 on hard |
| Safety-coast (low amp, never accumulate side effects) | Hard grader: `beta_score` at 0.22 weight — low amp = low entrainment = high beta = low score |
| Memorise the fixed Fleming trajectory | Per-episode signal noise (std 0.025–0.050) + seeded stochastic events prevent trajectory replay |
| Set `motor_command = 0` | `tracking_score` measures `|target − effective|` — zero command scores poorly at any nonzero target |
| High frequency for marginal entrainment gain | `_freq_side_effect_factor` scales burden faster at >140 Hz; safety budget depletes |
| React only to events (ignore baseline) | Events are stochastic and may not fire every episode; agent must maintain baseline control between events |

---

## 7. Validation: Expected Score Ranges

Baseline constant policy (1.0 mA, 0.13 ms, 130 Hz, motor_command = target_output):

| Task | Min score | Max score | Threshold | Constant passes? |
|---|---|---|---|---|
| easy | 0.72 | 0.80 | 0.55 | Always |
| medium | 0.47 | 0.52 | 0.52 | Never |
| hard | 0.23 | 0.36 | 0.68 | Never |

Expected ranges for a good reactive LLM agent (adjusts amplitude based on beta_trend and side_effect_rate):

| Task | Expected score | Passes? |
|---|---|---|
| easy | 0.78–0.88 | Yes |
| medium | 0.58–0.70 | Yes |
| hard | 0.48–0.62 | Marginal — needs phase-aware crisis management to reliably pass |

---

## 8. References

- Deuschl G et al. (2006). "A randomized trial of deep-brain stimulation for Parkinson's disease." *NEJM* 355(9):896–908.
- Kühn AA et al. (2008). "High-frequency stimulation of the subthalamic nucleus suppresses oscillatory beta activity in patients with Parkinson's disease." *NeuroImage* 36(2):379–387.
- Limousin P et al. (1995). "Effect of parkinsonian signs and symptoms of bilateral subthalamic nucleus stimulation." *Lancet* 345(8942):91–95.
- Little S et al. (2013). "Adaptive deep brain stimulation in advanced Parkinson disease." *Ann Neurol* 74(3):449–457.
- Little S et al. (2016). "Closed-loop deep brain stimulation: An evolving technology." *Mov Disord* 31(8):1336–1341.
- Priori A et al. (2013). "Adaptive deep brain stimulation (aDBS) controlled by local field potential oscillations." *Exp Neurol* 245:77–86.
- Rosa M et al. (2015). "Adaptive deep brain stimulation in a freely moving Parkinsonian patient." *Mov Disord* 30(7):1003–1005.
- Swann NC et al. (2018). "Adaptive deep brain stimulation for Parkinson's disease using motor cortex sensing." *J Neural Eng* 15(4):046006.
- Tinkhauser G et al. (2017). "Beta burst dynamics in Parkinson's disease OFF and ON dopaminergic medication." *Brain* 140(11):2968–2981.
- Velisar A et al. (2019). "Dual threshold neural closed loop deep brain stimulation in Parkinson disease patients." *Brain Stimul* 12(4):868–876.
- Fleming JE et al. (2023). "Multivariable closed-loop control of deep brain stimulation for Parkinson's disease." *J Neural Eng* 20(5):056029.
