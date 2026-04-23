# Inference Output Analysis

## Final Summary
```text
============================================================
SUMMARY
  beta_suppression       [########------------] 0.4215  FAIL
  tremor_correction      [######--------------] 0.3321  FAIL
  full_episode           [###############-----] 0.7631  PASS

  Mean score: 0.5056
============================================================
```

## Why do the early tasks fail while the hard task passes?

The reason the LLM failed the "easy" and "medium" tasks but passed the "hard" (`full_episode`) task comes down to strict clinical constraints and the AI's overly aggressive dosing strategy. 

Based on `dbs_tasks.py` and `dbs_graders.py`, each task enforces different clinical rules:

### 1. `beta_suppression` (Failed - 0.42)
* **Goal**: Gently suppress early beta-oscillations (20 steps).
* **Constraints**: Maximum amplitude of **1.0 mA**.
* **What happened**: The LLM frequently blasted 1.5 mA – 2.5 mA right out of the gate. Since this task heavily penalizes using too much amplitude (`efficiency_score`) and strictly caps the `max_side_effect_load` at 0.30, the LLM lost a massive chunk of its grade for overdosing the patient when it wasn't necessary yet.

### 2. `tremor_correction` (Failed - 0.33)
* **Goal**: Manage rapidly building tremor while keeping force function preserved (50 steps).
* **Constraints**: Needs to keep force preserved above **35%**, and amplitude capped at **2.0 mA**.
* **What happened**: The grading here puts 50% of the weight on keeping muscle force high and 10% as a "final bonus" for not collapsing at the end. Because the tremor ramps aggressively in these 50 steps, the AI failed to balance the DBS parameters. It likely drained the side-effect budget and lost the end-state bonus by letting the force collapse.

### 3. `full_episode` (Passed - 0.76)
* **Goal**: The complete clinical picture (100 steps).
* **Constraints**: Maximum amplitude generously raised to **3.0 mA** and the target force threshold loosely lowered to **25%** preserved.
* **What happened**: Since the limits are much looser (to account for the severe tremor at the end), the AI’s aggressive strategy (outputting 2.0 - 3.0 mA) suddenly becomes *valid*. It didn't get penalized for high amplitudes because the hard cap was set to 3.0 mA.

**Conclusion:** The benchmark LLM acts like a "hammer" that treats every symptom with maximum voltage. This aggressive strategy perfectly fits the loose constraints of the `full_episode`, but massively violates the strict safety and efficiency constraints in the earlier, milder tasks where a delicate touch is needed.
