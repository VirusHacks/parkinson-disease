# Reward Design — MotorAssistEnv

## 1. What does each reward term actually measure clinically?

Every term maps to a published clinical endpoint or a documented DBS failure mode — not a heuristic.

| Reward component | Clinical basis |
|---|---|
| `force_preserved` | Primary functional outcome in DBS trials (Limousin 1995; Deuschl 2006) |
| `beta_arv` suppression | Closed-loop DBS feedback target (Little 2013, 2016) |
| `tremor_arv` suppression | Secondary biomarker (Tinkhauser 2017) |
| `side_effect_load` budget | Stimulation-induced dyskinesia threshold (Rosa 2015; Swann 2018) |
| `efficiency` (min effective amp) | Battery longevity (Priori 2013); aDBS cuts stim time 56% (Little 2016) |
| `smoothness_cost` | Abrupt parameter changes cause transient dyskinesia (Velisar 2019) |
| `terminal_stability` | Sustained benefit at episode end — prevents early-episode gaming |

---

## 2. How does the agent get feedback every 20 ms?

A dense per-step reward, with weights aligned to the grader so the training signal points the same way as the evaluation signal.

```
r_t = w_force  * force_preserved_t
    + w_track  * tracking_accuracy_t
    + w_beta   * (1 − beta_arv_t)
    + w_tremor * (1 − tremor_arv_t)
    + w_safety * safety_t
    + w_smooth * (1 − smoothness_cost_t)
    + w_eff    * efficiency_t
    + shaping_t                         ← phase-aware bonus on hard/medium
    − 0.08 * constraint_violation_t
```

- `safety_t = clamp(1 − side_effect_load_t / max_side_effect_load)`
- `efficiency_t = 0.65 * (1 − amp/amp_max) + 0.35 * (1 − pw_norm)` — gated by therapeutic engagement
- `shaping_t` — late-episode terminal-stability proxy (hard: final 25%; medium: recovery proxy from ~step 50%)

| Weight | `easy` | `medium` | `hard` |
|---|---|---|---|
| `w_beta` | 0.30 | 0.06 | **0.22** |
| `w_tremor` | 0.18 | 0.14 | **0.14** |
| `w_force` | 0.16 | 0.16 | 0.14 |
| `w_track` | 0.12 | 0.16 | 0.16 |
| `w_safety` | 0.14 | **0.22** | 0.18 |
| `w_smooth` | 0.05 | 0.04 | 0.04 |
| `w_eff` | 0.05 | 0.08 | 0.04 |

The original hard reward had `w_safety = 0.36`, which rewarded a 1.0 mA constant policy at ~0.55 average for doing very little. Current weights make beta and tremor suppression the primary signal, matching the actual goal of DBS.

---

## 3. How is the final episode score computed?

Nine independent components, each combining mean performance with a threshold-adherence term. Computed once at episode end from the trajectory; deterministic given the same trajectory.

| Component | Formula | What it captures |
|---|---|---|
| `beta_score` | `0.55·weighted_mean(1−β) + 0.45·frac(β ≤ target)` | Suppression depth + time-in-range (Tinkhauser 2017) |
| `tremor_score` | `0.60·weighted_mean(1−T) + 0.40·frac(T ≤ target)` | Same dual metric for tremor |
| `force_score` | `weighted_mean(force) / target_force`, clamped | Voluntary motor function; early steps weighted ~1.35× |
| `tracking_score` | `0.45·(1 − err/target_err) + 0.55·tracking_acc` | Voluntary task execution (Limousin 1995) |
| `safety_score` | `clamp(1 − (0.45·mean + 0.35·peak + 0.20·violation)·1.8)` | Side-effect overload; peak weighted separately (Swann 2018) |
| `efficiency_score` | `(0.65·(1−mean_amp/max) + 0.35·(1−mean_pw)) × therapeutic_engagement` | Min effective dose, **gated** to prevent zero-DBS gaming |
| `smoothness_score` | `1 − mean(smoothness_cost)` | Abrupt-change penalty (Velisar 2019) |
| `terminal_stability_score` | `0.45·force + 0.30·(1−T) + 0.25·(1−err)` on **last 5 steps only** | Sustained benefit; blocks front-loading |
| `recovery_score` | `0.40·force_recov + 0.40·tremor_recov + 0.20·track_recov` (first 6 vs last 8 steps) | Did the agent rescue the patient, or just survive? |

`therapeutic_engagement = 0.40·force_score + 0.30·beta_score + 0.30·tremor_score` — the gate that makes "do nothing for perfect efficiency" score zero.

---

## 4. How do grader weights shift per task?

Different tasks weight components differently to reflect different clinical priorities. Beta/tremor dominate when establishing control; safety dominates during rescue; functional outcomes dominate in scenario tasks.

| Component | `easy` | `medium` | `hard` |
|---|---|---|---|
| `beta_score` | **0.30** | 0.06 | **0.22** |
| `tremor_score` | 0.18 | 0.14 | 0.14 |
| `force_score` | 0.16 | 0.16 | 0.14 |
| `tracking_score` | 0.12 | 0.16 | 0.16 |
| `safety_score` | 0.14 | **0.22** | 0.18 |
| `terminal_stability` | — | 0.08 | 0.08 |
| `efficiency_score` | 0.05 | 0.08 | 0.04 |
| `smoothness_score` | 0.05 | 0.04 | 0.04 |
| `recovery_score` | — | 0.06 | — |

Easy emphasises beta — the agent must learn DBS actually suppresses pathological activity. Medium emphasises safety + recovery — rescue without dyskinesia. Hard balances beta+tremor+safety so low-stim coasting can no longer game the safety term.

### Expert / scenario tasks

| Task | Primary weights |
|---|---|
| `fragile_patient` | safety 0.28, tracking 0.18, force 0.18 — therapeutic window precision |
| `refractory_patient` | force 0.18, tracking 0.14, tremor 0.12 — outcome despite weak response |
| `personalization_generalization` | force 0.18, tracking 0.18, safety 0.18 — balanced across all profiles |
| `exercise_bout` | force 0.22, tracking 0.22 — motor performance during exertion |
| `medication_interaction` | safety 0.22, recovery 0.10 — crisis management without over-treatment |
| `nocturnal_transition` | safety 0.22, terminal_stability 0.12, efficiency 0.12 — sleep-phase stability |
| `surgical_followup` | safety 0.30 — microlesion window dominates |

---

## 5. What clinical failures trigger explicit penalties?

These model clinically unacceptable outcomes that a smooth component score might otherwise undervalue. Final score = `clamp(weighted_sum − total_penalty, 0.0, 1.0)`.

### Universal (all tasks)

| Condition | Penalty | Clinical meaning |
|---|---|---|
| `safety_score < 0.20` | −0.12 | Sustained dyskinesia risk |
| `tracking_score < 0.20` | −0.08 | Patient cannot execute movement |
| `beta_score < 0.40` | −0.06 | DBS providing no measurable suppression |
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
| **hard** | `terminal_stability < 0.25` | **−0.08** |
| hard | high mean amp + poor efficiency | −0.08 |
| exercise_bout | `tracking_score < 0.55` | −0.14 |
| exercise_bout | `force_score < 0.60` | −0.10 |
| exercise_bout | zero-stim during exertion | −0.16 |
| medication_interaction | `recovery_score < 0.40` | −0.12 |
| medication_interaction | high trailing amp + poor safety | −0.10 |
| nocturnal_transition | `terminal_stability < 0.45` | −0.12 |
| nocturnal_transition | low efficiency + poor safety | −0.08 |
| surgical_followup | amp violations in first 25% | **−0.20** |
| surgical_followup | `safety_score < 0.55` | −0.08 |
| surgical_followup | `recovery_score < 0.30` | −0.06 |

---

## 6. What shortcuts could an adversarial agent try, and what blocks each?

Twelve attack policies, traced through the dense reward and grader to confirm each loses score. `None` = fully blocked; `Bounded` = partially exploitable but capped well below the success threshold.

| # | Attack strategy | What blocks it | Residual risk |
|---|---|---|---|
| 1 | Do nothing (zero amp forever) | `efficiency × therapeutic_engagement` collapses; universal floors fire (`β < 0.40 → −0.06`, etc.); easy adds `−0.20`, fragile adds `−0.22` | None |
| 2 | Always max amp | Safety drops below 0.20 → `−0.12`; brute-force penalties (`hard −0.08`, `easy −0.14`); `adaptation_state` physically reduces effect over time | None |
| 3 | Constant 1.0 mA | Calibration baseline: passes easy (~0.72–0.80), fails medium (~0.47–0.52), fails hard (~0.23–0.36) | None — proves the difficulty ordering |
| 4 | Track via `motor_command` alone | Effective output gated by `(1 − 0.52β)(1 − 0.30T)(1 − 0.10SE)`; tracking is structurally coupled to DBS control | None |
| 5 | Farm recovery score by tanking start | Dense reward in early steps suffers; recovery weight ≤ 0.10; `terminal_stability` requires last-5-step quality | **Bounded** — small leak only |
| 6 | Smoothness via inaction | Smoothness weight ≤ 0.05; constant policy fails on every other axis | None |
| 7 | Front-load good steps, drift later | `terminal_stability_score` reads only last 5 steps; `hard −0.08`, `nocturnal −0.12` | None |
| 8 | Fool the sensor instead of the patient | Grader reads **latent** state; sensor noise lives only in `_make_obs`; no API path to the grader buffer | None — structurally impossible |
| 9 | Reward tampering | FastAPI sandbox; only entry point is `step(action)` with Pydantic-validated payload | None — capability not exposed |
| 10 | Find a simulator bug | Divisors floored at `1e-6`; noise ranges clamped; bilinear interpolation only | Negligible |
| 11 | Memorise the Fleming trajectory | Per-episode noise on every signal; random L-DOPA phase; random target; seeded events | None — fresh seed = fresh episode |
| 12 | Game dense reward, not grader | Episode-return RL (GRPO) sees grader directly; PPO eventually feels it via the value function | **Bounded** — see §7 risk #1 |
| 13 | Set `motor_command = 0` | `tracking_score` measures `|target − effective|`; zero command scores poorly at any nonzero target | None |
| 14 | Push frequency to 185 Hz for entrainment | `_freq_side_effect_factor` scales burden faster above 140 Hz; safety budget depletes | None |
| 15 | React only to events, ignore baseline | Events are stochastic; agent must maintain baseline control between events | None |

## 7. What stops cheating outside the reward function?

The penalty tables in §5 catch obvious specification gaming. The defenses below catch the cleverer attacks — including ones we haven't thought of yet. None of these appear in the score formulas; they live in the surrounding scaffolding.

| Mechanism | What it does | What it blocks |
|---|---|---|
| Latent vs sensed split | Grader reads `_beta_state`; agent reads `obs.beta_arv` with Gaussian sensor noise | Sensor-fooling (DeepMind grasping example) |
| Per-episode noise factors | `ep_beta_noise`, `ep_tremor_noise`, `ep_force_noise`, `ep_semg_noise` resampled on every reset | Open-loop trajectory memorisation |
| Stochastic events as real physics | `_apply_event` modifies entrainment, beta drive, side-effect rate — not just the score | Pre-baked schedules; ignoring events |
| FastAPI sandbox | Only entry is `step(action)` with Pydantic-validated `ParkinsonsMotorAction` | Reward tampering; reading grader weights |
| Bounded multipliers | All noise/factor ranges clamped; divisors floored at `1e-6` | Floating-point exploits, divide-by-zero |
| Random `target_output` per episode | Tracking target re-rolled at reset within task range | Hard-coding a target value |

The latent-vs-sensed split mirrors clinical reality: real DBS devices read noisy LFP off the same electrode they stimulate from, while clinical outcomes are measured separately by a physician with a dynamometer. The signal you control with and the outcome you are judged on are physically distinct.

---

## 8. What design principles is this built on?

Lessons from DeepMind's *Specification gaming* post and the OpenEnv hackathon guide, mapped to concrete pieces of MotorAssistEnv.

| Principle | Where it lives |
|---|---|
| The reward is a contract; agents read it literally | Multi-component grader + hard-failure penalties + dense/grader split |
| Use multiple independent reward functions | 9 grader components on different clinical axes |
| Reward should be rich and informative, not 0/1 at the end | Per-step dense reward at 20 ms cadence + episode-end grader |
| Block obvious shortcut policies before training | §6 — every named attack has a documented block |
| Exploiting without solving should not score high | `therapeutic_engagement` gate; constant baseline calibrated below threshold |
| Anticipate sensor-fooling and reward tampering | Latent vs sensed split (§8); FastAPI sandbox (§8) |
| Make difficulty ordering empirically falsifiable | Constant baseline scores in §10 are strictly monotone |

---

## 9. What proves the difficulty ordering is real?

The constant 1.0 mA baseline policy was run across 5 seeds on each task. Strict ordering holds every seed.

| Task | Min | Max | Threshold | Constant passes? |
|---|---|---|---|---|
| easy | 0.72 | 0.80 | 0.55 | Always |
| medium | 0.47 | 0.52 | 0.52 | Never |
| hard | 0.23 | 0.36 | 0.68 | Never |

Expected ranges for a good reactive LLM agent (adjusts amplitude based on `beta_trend` and `side_effect_rate`):

| Task | Expected | Passes? |
|---|---|---|
| easy | 0.78–0.88 | Yes |
| medium | 0.58–0.70 | Yes |
| hard | 0.48–0.62 | Marginal — needs phase-aware crisis management |

---

## 10. References

- Deuschl G et al. (2006). *NEJM* 355(9):896–908.
- Kühn AA et al. (2008). *NeuroImage* 36(2):379–387.
- Limousin P et al. (1995). *Lancet* 345(8942):91–95.
- Little S et al. (2013). *Ann Neurol* 74(3):449–457.
- Little S et al. (2016). *Mov Disord* 31(8):1336–1341.
- Priori A et al. (2013). *Exp Neurol* 245:77–86.
- Rosa M et al. (2015). *Mov Disord* 30(7):1003–1005.
- Swann NC et al. (2018). *J Neural Eng* 15(4):046006.
- Tinkhauser G et al. (2017). *Brain* 140(11):2968–2981.
- Velisar A et al. (2019). *Brain Stimul* 12(4):868–876.
- Fleming JE et al. (2023). *J Neural Eng* 20(5):056029.
