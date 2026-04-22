# Reward Design — MotorAssistEnv (DBS Parkinson's Environment)

## 1. Design Philosophy

The reward is the task specification. In this environment, the reward mirrors what a neurologist actually optimises for during DBS programming:

1. **Preserve motor function** — the patient must be able to move
2. **Suppress the oscillation** — the pathological beta band is the root cause
3. **Manage side effects** — too much stimulation causes dyskinesia, discomfort, and battery depletion
4. **Use the minimum effective amplitude** — conservative stimulation is safer long-term

The reward design follows three principles taken directly from clinical DBS practice:

- **The primary outcome is motor function** (`force_preserved`) — not the intermediate electrical signal
- **Dense feedback beats sparse** — the agent receives reward every step, not just at episode end
- **Multiple independent criteria prevent reward hacking** — the grader checks force, beta, side effects, and efficiency separately

---

## 2. Per-Step Reward (Dense Signal for Training)

At every step the agent receives an immediate dense reward:

```
r_t = 0.50 × force_preserved_t
    + 0.30 × (1 − task_error_t)
    + 0.15 × dbs_entrainment_t
    − 0.005 × dbs_amplitude_t
```

### Term Breakdown

| Term | Weight | Meaning |
|---|---|---|
| `force_preserved` | 0.50 | Primary: is the patient's muscle working? (fraction of healthy baseline) |
| `1 − task_error` | 0.30 | Secondary: is the voluntary motor command reaching the target? |
| `dbs_entrainment` | 0.15 | Bonus: is the DBS actually suppressing the pathological circuit? |
| `−dbs_amplitude` | 0.005 | Micro-penalty: discourages unnecessary over-stimulation |

**Where values come from:**
- `force_preserved` comes directly from the calibrated simulation data, boosted by the DBS entrainment the agent applied at the previous step
- `task_error = |target_output − effective_motor_output|` where `effective_motor_output` is the agent's command after Parkinsonian distortion
- `dbs_entrainment` is bilinearly interpolated from the 12×15 parameter sweep table for the agent's chosen amplitude and pulse width

**Range:** Per-step reward is typically in [0.2, 1.0] for a reasonable policy. An agent that delivers no DBS and issues zero motor command would score ~0.10–0.40 depending on the step.

---

## 3. Episode-End Grader Score (Deterministic 0.0–1.0)

At the end of every episode the registered grader produces a final score from the full episode trajectory. This score is the ground truth for judge evaluation and the `[END] score=` field in the inference script output.

### Shared Scoring Primitives

All three task graders are built on the same clinical primitives:

**`force_score`**  
Weighted mean of `force_preserved` across the episode (early steps weighted 2× relative to final steps, reflecting clinical priority of immediate motor function):
```
force_score = weighted_mean(force_preserved) / target_force_preserved
              clamped to [0, 1]
```

**`beta_score`**  
Fraction of steps where `beta_arv` stayed below the task's clinical threshold:
```
beta_score = count(beta_arv < target_beta_arv) / n_steps
```

**`side_effect_penalty`**  
1.0 if the side-effect load never exceeded budget; decays toward 0 proportional to the frequency and magnitude of violations:
```
side_effect_score = 1.0 − (mean_excess × violation_fraction × 2.0)
                    clamped to [0, 1]
```

**`amplitude_efficiency`**  
Rewards the minimum effective amplitude — an agent that blasts 3 mA every step is penalised:
```
efficiency = 1.0 − (mean_dbs_amplitude / max_allowed_amplitude)
```

**`final_state_bonus`**  
Binary: 1.0 if `force_preserved` at the **last** step ≥ 80% of the task's target. This prevents the agent from doing well early and catastrophically degrading at episode end.

---

## 4. Task-Specific Grader Weights

### Task 1 — `beta_suppression` (Easy)

```
score = 0.50 × beta_score
      + 0.25 × force_score
      + 0.15 × side_effect_score
      + 0.10 × amplitude_efficiency
```

**Why:** Beta suppression is the primary clinical objective for this early-phase window.

### Task 2 — `tremor_correction` (Medium)

```
score = 0.50 × force_score
      + 0.25 × beta_score
      + 0.15 × side_effect_score
      + 0.10 × final_state_bonus
```

**Why:** Once tremor is actively building, the primary concern shifts to **preventing motor function decay**. Beta suppression is still important but is now the mechanism, not the goal.

### Task 3 — `full_episode` (Hard)

```
score = 0.40 × force_score
      + 0.20 × beta_score
      + 0.20 × side_effect_score
      + 0.10 × amplitude_efficiency
      + 0.10 × final_state_bonus
```

**Why:** Over 100 steps, side-effect management becomes equally critical. An agent that suppresses beta perfectly but exhausts the side-effect budget by step 50 has failed clinically.

---

## 5. Grader Score Semantics

| Score range | Interpretation |
|---|---|
| 0.80 – 1.00 | Excellent: agent performs near-optimally for this clinical scenario |
| 0.60 – 0.79 | Good: meaningful benefit to patient, some room for improvement |
| 0.50 – 0.59 | Marginal: on the edge of clinical usefulness |
| 0.30 – 0.49 | Poor: agent fails to achieve the clinical objective on most steps |
| 0.00 – 0.29 | Failure: agent has made patient state worse or induced excessive side effects |

---

## 6. Anti-Hacking Design

The reward is deliberately structured to prevent common RL shortcuts:

| Potential hack | Protection |
|---|---|
| Apply maximum DBS (3 mA) every step | Amplitude efficiency term penalises this; side-effect budget caps it |
| Apply zero DBS (avoid side-effect penalty) | Force score collapses without DBS at high tremor steps |
| Stay in a high-reward step by not advancing | The simulator advances deterministically — no way to freeze time |
| Optimise only early steps | Final state bonus requires force to be preserved at episode end |
| Score well on one metric by sacrificing another | Independent grader components make multi-objective exploitation unprofitable |

---

## 7. Reward Signal Analysis

**Naive agent (constant 1.0 mA / 0.13 ms):**

| Task | Grader score | Success? |
|---|---|---|
| beta_suppression | 0.43 | ❌ |
| tremor_correction | 0.88 | ✅ |
| full_episode | 0.76 | ✅ |

**What a near-optimal agent must do differently:**
- `beta_suppression`: Use lower amplitude (~0.5–0.8 mA) to suppress beta without wasting the side-effect budget in a short window
- `tremor_correction`: Increase amplitude dynamically as `tremor_arv` spikes; back off when `side_effect_load` nears 0.50
- `full_episode`: Front-load high amplitude in the mid-episode to prevent the late-episode tremor catastrophe; sustain with moderate settings thereafter

**The learning gradient exists** — the reward clearly differentiates better DBS policies from worse ones. An agent that discovers higher amplitude (≥ 1.5 mA) in the mid-episode achieves meaningfully higher cumulative force preservation than the naive flat-1.0-mA baseline.
