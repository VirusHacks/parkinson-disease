# Reward Design — MotorAssistEnv

## 1. Clinical Grounding

The reward function is a computational translation of the clinical objectives that
neurologists optimise during DBS programming sessions. It is not heuristic — every
term maps to a published clinical endpoint or a documented failure mode.

**Primary clinical references:**

| Reward component | Clinical basis |
|---|---|
| `force_preserved` | Primary functional outcome in DBS trials (Limousin et al. 1995; Deuschl et al. 2006) |
| `beta_arv` suppression | Beta-band power as closed-loop DBS feedback target (Little et al. 2013, 2016) |
| `tremor_arv` suppression | Tremor ARV as secondary biomarker (Tinkhauser et al. 2017) |
| `side_effect_load` budget | Stimulation-induced dyskinesia threshold (Rosa et al. 2015; Swann et al. 2018) |
| `efficiency` (min effective amp) | Battery longevity in implanted pulse generators (Priori et al. 2013) |
| `smoothness_cost` penalty | Abrupt parameter changes cause transient dyskinesia (Velisar et al. 2019) |
| `terminal_stability` | Sustained benefit at episode end — prevents early-episode gaming |

**Core insight from Little et al. (2016):** Adaptive DBS using beta power as feedback
reduced stimulation time by 56% while matching or exceeding continuous DBS outcomes.
This motivates our efficiency term — maximum amplitude is never the right answer.

---

## 2. Per-Step Dense Reward

At every 20 ms timestep the agent receives a dense reward signal:

```
r_t = w_force  * force_preserved_t
    + w_track  * tracking_accuracy_t
    + w_beta   * (1 - beta_arv_t)
    + w_tremor * (1 - tremor_arv_t)
    + w_safety * safety_t
    + w_smooth * (1 - smoothness_cost_t)
    + w_eff    * efficiency_t
    - 0.08     * constraint_violation_t
```

where `safety_t = clamp(1 - side_effect_load_t / budget)` and
`efficiency_t = 1 - (amp / amp_max)`.

**Weights are task-specific** — the training signal mirrors the evaluation grader
so the agent cannot exploit misalignment between the two:

| Weight | beta_suppression | tremor_correction | full_episode |
|---|---|---|---|
| `w_force` | 0.16 | 0.16 | 0.14 |
| `w_track` | 0.12 | 0.16 | 0.16 |
| `w_beta` | 0.30 | 0.06 | 0.08 |
| `w_tremor` | 0.18 | 0.14 | 0.06 |
| `w_safety` | 0.14 | 0.22 | 0.36 |
| `w_smooth` | 0.05 | 0.04 | 0.05 |
| `w_eff` | 0.05 | 0.08 | 0.10 |

**Design invariant:** weights sum to ~1.0 per task (excluding the violation penalty)
so reward is interpretable on a per-step [0, 1] scale.

**Range in practice:**
- Well-calibrated policy: 0.55–0.85 per step
- Naive zero-stim policy: 0.15–0.45 (collapses as tremor builds)
- Max-amp constant policy: 0.40–0.65 (safety term drags it down)

---

## 3. Episode-End Grader Score (Deterministic 0.0–1.0)

The grader is the authoritative benchmark signal. It is computed once at episode end
from the full trajectory and is deterministic given the same trajectory.

### Score Components

**`force_score`**
```
Weighted mean of force_preserved over all steps (early steps weighted ~1.35×
relative to terminal steps — mirrors clinical priority of immediate function).
force_score = weighted_mean(force_preserved) / target_force_preserved, clamped [0,1]
```
*Basis: force_preserved tracks the muscle output clinicians measure via dynamometry.*

**`beta_score`**
```
beta_score = 0.55 * weighted_mean(1 - beta_arv)
           + 0.45 * fraction_of_steps(beta_arv <= target_beta_arv)
```
*Basis: combines mean suppression depth with time-in-therapeutic-range (TTR),
mirroring the dual metric used in aDBS trials (Tinkhauser et al. 2017).*

**`tremor_score`**
```
tremor_score = 0.60 * weighted_mean(1 - tremor_arv)
             + 0.40 * fraction_of_steps(tremor_arv <= target_tremor_arv)
```

**`tracking_score`**
```
tracking_score = 0.45 * weighted_mean(1 - task_error / target_error)
               + 0.55 * weighted_mean(tracking_accuracy)
```
*Basis: voluntary motor task performance — the patient's ability to execute intended
movements is the primary reason for DBS implantation (Limousin et al. 1995).*

**`safety_score`**
```
overload_penalty = mean(max(0, (side_effect_load - budget) / (1 - budget)) for each step)
peak_penalty = max(overload per step)
safety_score = clamp(1 - (0.45*overload_penalty + 0.35*peak_penalty + 0.20*mean_violation) * 1.8)
```
*Basis: stimulation-induced dyskinesia emerges as a threshold effect — a single
brief overload can be as harmful as sustained moderate overload (Swann et al. 2018).
Hence peak_penalty is weighted separately from mean overload.*

**`efficiency_score`**
```
efficiency_score = (0.65*(1 - mean_amp/max_amp) + 0.35*(1 - mean_pw_norm))
                 * therapeutic_engagement
```
where `therapeutic_engagement = clamp(0.40*force_score + 0.30*beta_score + 0.30*tremor_score)`.

*Gating by therapeutic_engagement prevents reward for doing nothing — an agent that
uses zero DBS and thus has perfect efficiency scores zero on efficiency because
therapeutic_engagement collapses.*

**`smoothness_score`**
```
smoothness_score = 1 - mean(smoothness_cost_per_step)
```
*Basis: abrupt amplitude changes cause transient dyskinesia and patient discomfort
even within safe total dose (Velisar et al. 2019).*

**`terminal_stability_score`**
```
Computed on the final 5 steps only.
terminal_stability = 0.45*(last_force/target) + 0.30*(1 - last_tremor/target) + 0.25*(1 - last_error/target)
```
*Prevents strategies that front-load good performance and degrade at episode end.*

**`recovery_score`**
```
Compares first-6-step window vs last-8-step window for force, tremor, and tracking.
recovery_score = 0.40*force_recovery + 0.40*tremor_recovery + 0.20*tracking_recovery
```
*Key for tremor_correction task: did the agent actually rescue the patient from the
escalation window, or did it just survive?*

---

## 4. Task-Specific Grader Weights

```
beta_suppression:
  0.30 * beta_score + 0.18 * tremor_score + 0.16 * tracking_score
+ 0.14 * force_score + 0.14 * safety_score + 0.04 * smoothness + 0.04 * efficiency

tremor_correction:
  0.22 * safety_score + 0.16 * force_score + 0.16 * tracking_score
+ 0.14 * tremor_score + 0.12 * terminal_stability + 0.06 * beta_score
+ 0.06 * efficiency + 0.06 * recovery_score + 0.04 * smoothness

full_episode:
  0.36 * safety_score + 0.16 * tracking_score + 0.14 * force_score
+ 0.10 * efficiency + 0.08 * beta_score + 0.06 * tremor_score
+ 0.05 * smoothness + 0.05 * terminal_stability
```

**Why safety dominates full_episode (36%):** Over 100 steps, a policy that exhausts
the side-effect budget by step 60 leaves the patient unprotected for the remainder —
this is the most dangerous clinical outcome. The full-episode grader penalises this
more heavily than any other failure mode.

---

## 5. Hard-Failure Penalties

On top of the weighted score, the grader applies deterministic hard penalties for
clinically unacceptable outcomes:

| Condition | Penalty | Clinical basis |
|---|---|---|
| `safety_score < 0.20` | −0.12 | Sustained dyskinesia risk |
| `tracking_score < 0.20` | −0.08 | Patient cannot execute voluntary movement |
| `beta_score < 0.40` | −0.06 | DBS providing no measurable beta suppression |
| `tremor_score < 0.22` | −0.05 | Active tremor uncontrolled |
| `force_score < 0.55` | −0.04 | Severe motor function loss |
| No DBS + symptoms not suppressed (task-specific) | −0.16 to −0.22 | Untreated escalation |
| Constant max-amp + low efficiency | −0.14 | Battery waste + side-effect risk |

These penalties are **independent of the weighted score** — a policy that scores 0.65
on all weighted terms but triggers a hard-failure rule will land below the success
threshold for that task.

---

## 6. Anti-Hacking Analysis

The environment is explicitly designed to make the following RL shortcuts unprofitable:

| Exploit strategy | Blocking mechanism |
|---|---|
| Max amplitude every step | `efficiency_score` penalises mean amp; side-effect budget depletes safety score |
| Zero stimulation | `beta_score`, `tremor_score` collapse; hard-failure penalty for no-DBS + symptoms |
| Front-load good steps, degrade at end | `terminal_stability_score` grades only the final 5 steps |
| Tune for one task metric only | Multiple independent grader components — improving one at the cost of others is net-negative |
| Memorise the fixed Fleming trajectory | Per-episode trajectory noise (std=0.08 on beta/tremor, std=0.05 on force) prevents replay |
| Set motor_command=0 to avoid tracking risk | `tracking_score` measures `|target - effective|` — zero command scores poorly at any nonzero target |
| Very high frequency for marginal entrainment gain | `_freq_side_effect_factor` scales burden faster at >140 Hz, eating safety budget |

---

## 7. Validation: Expected Score Ranges

Based on the calibrated Fleming trajectory with episode noise (std=0.08):

| Policy | beta_suppression | tremor_correction | full_episode |
|---|---|---|---|
| zero_stim (amp=0) | 0.18–0.28 | 0.10–0.20 | 0.12–0.22 |
| constant (1.0 mA, 0.13 ms, 130 Hz) | 0.44–0.56 | 0.35–0.48 | 0.30–0.42 |
| constant (max_amp, max_pw, 130 Hz) | 0.30–0.42 | 0.28–0.38 | 0.20–0.30 |
| safety_aware (adaptive rule-based) | 0.52–0.64 | 0.42–0.58 | 0.45–0.58 |
| **success threshold** | **0.50** | **0.36** | **0.62** |

A well-designed LLM agent reading the observation fields and following the action
description strategy hints should reliably reach 0.52–0.65 on the easy task and
0.42–0.55 on the medium task. The hard task requires multi-step temporal reasoning
about side-effect accumulation and is expected to be challenging for zero-shot agents.

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
- Fleming JE et al. (2020). "Simulation of closed-loop deep brain stimulation control schemes." *PLOS Comput Biol* 16(8):e1008165.
