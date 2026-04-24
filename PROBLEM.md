# Problem Statement — MotorAssistEnv: Closed-Loop DBS Agent for Parkinson's Disease

---

## 1. Summary

Parkinson's disease disrupts the basal ganglia circuit through pathological beta-band synchrony in the subthalamic nucleus (STN). This beta oscillation drives tremor, rigidity, and the progressive loss of voluntary motor function that makes everyday tasks — holding a cup, reaching for a door, writing a signature, fastening a button — increasingly impossible. It is not simply a disease of movement. It is a disease of _lost agency_. Patients describe not being able to do what they mentally intend to do. The signal from the brain to the muscles is scrambled.

Deep Brain Stimulation (DBS) is the gold standard intervention. A surgically implanted electrode delivers high-frequency (>100 Hz) electrical pulses to the STN, disrupting the pathological beta synchrony and partially restoring motor function. The challenge: the DBS device must be tuned continuously, requiring a trained neurologist. Too little amplitude → beta persists, tremor grows, the patient cannot move. Too much → cortical spreading causes dyskinesia, patient discomfort, battery depletion, and long-term tissue damage. The correct settings drift as the disease progresses, as the electrode placement shifts slightly, as the patient ages. Suboptimal programming is not an edge case — it is the norm. Most patients go months between tuning visits.

**MotorAssistEnv** frames this as a sequential decision problem for a reinforcement learning agent. The agent acts as an autonomous closed-loop BCI (Brain-Computer Interface) programmer: it observes the patient's real-time brain state at each 20 ms timestep and issues DBS amplitude and pulse width parameters to maximise motor function preservation while managing cumulative side effects.

The environment is not a toy simulation. Every observation the agent receives was produced by a peer-reviewed biophysical neural network simulation (Fleming et al. 2023) modelling the exact brain circuits disrupted in Parkinson's disease, at the level of individual neurons and synaptic connections.

---

## 2. Why This Problem Matters

### 2.1 Clinical Reality

This targets a real, unmet clinical need:

- **1 million+ patients** in the United States have Parkinson's disease. An estimated 10 million worldwide.
- **Deep Brain Stimulation** is effective in ~50,000+ patients, and the number is growing as indications expand.
- **Suboptimal DBS programming** is identified by neurologists as the #1 barrier to better outcomes after successful implantation. The device is there. The battery is charged. But the settings are wrong, and the patient is suffering.
- A neurologist's clinic visit for DBS programming takes 1–2 hours and can be scheduled only every 3–6 months at most centres. Between visits, the patient lives with whatever settings were last programmed — even if those settings are no longer appropriate.
- **Adaptive (closed-loop) DBS** is the next frontier of the field. The goal is a device that reads brain signals continuously and adjusts stimulation automatically — in real time, personalized to the patient's current state. RL is a natural fit for learning that policy.

### 2.2 The RL Problem Structure

From a technical perspective, DBS programming is a compelling RL problem because it has all the properties that make RL necessary and feasible:

- **Sequential action:** Each DBS setting choice affects the next state of the brain. Actions compound over time.
- **Non-stationary disturbances:** Tremor amplitude climbs progressively during an episode. A fixed policy fails. The agent must adapt.
- **Partial observability:** The agent sees local field potentials (LFP, represented by `beta_arv`) and surface EMG (`semg_arv`), but not individual neuron firing patterns. This matches the reality of what DBS hardware can measure.
- **Multi-objective trade-off:** The agent must simultaneously maximise motor function, suppress oscillation, and stay within the side-effect budget. No single scalar fully captures clinical success — multiple criteria must be jointly satisfied.
- **Dense feedback structure:** Unlike many medical environments where outcomes are only observed at discharge, DBS produces measurable physiological signals every 20 ms. This makes dense reward shaping natural and clinically meaningful.
- **Clear programmatic grading:** Success can be measured objectively as `force_preserved` > threshold and `side_effect_load` < budget — no human rater needed.

### 2.3 Why Not Just Use a Classical Controller

The ground-truth simulation runs a PID (Proportional-Integral-Derivative) closed-loop controller. RL is needed because:

- PID requires careful manual tuning of gain parameters for each patient. RL can adapt from data.
- PID cannot handle the multi-objective trade-off between force preservation and side-effect management without explicit engineering of the objective function.
- PID does not generalize across disease severity levels. An RL agent trained across a distribution of states can potentially transfer.
- In the real clinical setting, the "reward function" (what the patient actually wants) changes over time and is not fully specified. RL provides a framework for learning it from physiological feedback.

---

## 3. Data Foundation — The Fleming et al. (2023) Model

### 3.1 Why This Model Specifically

The environment is backed by a specific, peer-reviewed biophysical simulation:

> **Fleming, J.E., Senneff, S. and Lowery, M.M. (2023)**
> *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*
> Journal of Neural Engineering, 20(5), p.056029.

This model was chosen for three reasons that no other publicly available simulation offers simultaneously:

1. **It is the only model that connects brain → DBS → muscle force → surface EMG in one integrated pipeline.** Most neural simulations stop at the neuron level or at LFP. This model continues all the way to the musculoskeletal output — the force a patient's hand can exert. That is what patients care about.

2. **It produces real physical units** (mV, mA, mN) validated against real patient data. The beta ARV values, tremor amplitudes, and force outputs in this environment are not scaled arbitrary units. They are clinically measured quantities.

3. **It includes a ground-truth optimal controller** — a closed-loop PID/scheduler system published alongside the model. This provides a reference "what the best-known automated controller achieves" for every step of the simulation, allowing us to measure whether the RL agent has learned to beat or match clinical state-of-the-art.

### 3.2 What the Simulation Modelled

The simulation ran for ~75 seconds of simulated time and included:

- **Cortical layer:** ~100 pyramidal neurons with full Hodgkin-Huxley dynamics (sodium, potassium, M-current, leak channels), modelled via NEURON (the gold standard single-neuron simulator).
- **Basal Ganglia:** Subthalamic Nucleus (STN, 100 neurons), GPe (100 neurons), GPi (100 neurons) with biophysical ion channel models.
- **Thalamus:** Mediodorsal and ventrolateral nuclei modelling the relay of tremor signals to cortex.
- **Spinal cord:** A depressing spinal motoneuron pool receiving tremor-frequency drive.
- **Musculoskeletal:** Muscle force computed from motoneuron firing rates using a physiological Hill-type model.
- **DBS electrode:** Extracellular stimulation modelled with Finite-Element methods for current spread in tissue.
- **Total connections:** 5+ million individual synaptic connections modelled stochastically.

The simulation's output (stored in `parkinsons_Motor/fleming-model-based-brain/Model_Results/`) includes:
- 102 STN voltage traces (`.mat` files)
- 102 motoneuron voltage traces and spike times (`.mat` files)
- 34 CSV files of controller signals, sampled at 100 timesteps (t=10.02–12.00 s)
- A 12×15 DBS parameter sweep (`Collaterals_Entrained_values.txt`) mapping every combination of amplitude and pulse width to its cortical entrainment fraction.

### 3.3 What "Calibration" Means Here

The `core/calibration.py` module exposes the calibration interface that loads all simulation outputs and builds a `CalibratedBrainState` — a ground-truth 100-step timeline with the following at every timestep:

- Normalized neural signals (beta_arv, tremor_arv, semg_arv)
- Raw muscle force (mN) and force_preserved fraction
- Ground-truth DBS amplitude and pulse width used by the Fleming controller
- Physiological baselines (pre-DBS tremor, beta, force levels)
- The full 12×15 entrainment lookup table for the agent's DBS parameter queries

Every observation the RL agent will ever see is a transformation of this calibrated data — not a generative model, not a polynomial approximation. **The ground truth is the ground truth.**

---

## 4. System Architecture

```
═══════════════════════════════════════════════════════════════════
                    DATA LAYER (offline, fixed)
═══════════════════════════════════════════════════════════════════

  parkinsons_Motor/fleming-model-based-brain/
  ├── Model_Results/              ← 34 CSV controller files
  │   ├── tremor_ARV_Observer_values.csv
  │   ├── beta_ARV_Observer_values.csv
  │   ├── Force_amplitude_values.csv  (6.7M samples)
  │   └── ... (31 more files)
  ├── Collaterals_Entrained_values.txt  ← 12×15 DBS sweep
  ├── DBS_Amplitude_Interpolation_values.txt
  └── DBS_Pulse_Width_Interpolation_values.txt

═══════════════════════════════════════════════════════════════════
                    CALIBRATION (runs once, cached)
═══════════════════════════════════════════════════════════════════

  parkinsons_Motor/core/calibration.py
  └── calibrate() → CalibratedBrainState
      ├── 100-step timeline of WindowFeatures
      ├── Normalization bounds (from actual data maxima)
      ├── Physiological baselines (pre-DBS median values)
      └── 12×15 DBS entrainment matrix

═══════════════════════════════════════════════════════════════════
                    OPENENV ENVIRONMENT (online, per-episode)
═══════════════════════════════════════════════════════════════════

  parkinsons_Motor/
  ├── tasks/dbs_tasks.py          ← 3 clinical task specs (frozen dataclasses)
  ├── graders/dbs_graders.py      ← 3 deterministic graders (0.0–1.0)
  ├── core/models.py              ← Pydantic Action + Observation types
  ├── server/
  │   ├── parkinsons_Motor_environment.py  ← reset/step/state logic
  │   └── app.py                           ← FastAPI server
  └── client.py                   ← WebSocket client for inference

═══════════════════════════════════════════════════════════════════
                    VISUALISATION (separate, demo only)
═══════════════════════════════════════════════════════════════════

  static/myosuite_demo/           ← MyoSuite WebGL 3D arm visualiser
  /viewer endpoint                ← serves the demo
  Bridge: polls backend for tremor_arv → drives 3D arm jitter
  As agent suppresses beta → arm smoothes → patient performs task
```

---

## 5. What the Agent Must Learn

An agent that achieves high reward on this environment will have learned:

1. **Recognise the current disease phase** — `beta_arv` + `tremor_arv` encode where in the episode the patient currently is. Low tremor early means subtle DBS suffices. High tremor late means aggressive intervention is needed.

2. **Use DBS proportionally and early** — from the ground-truth data we know that the optimal policy front-loads stimulation to slow the tremor ramp-up, rather than reacting after tremor has already grown large.

3. **Navigate the bilinear entrainment surface** — the 12×15 lookup table is not linear. Very low amplitude (0–0.5 mA) produces near-zero entrainment. Going from 1.0 to 1.25 mA jumps entrainment from 36% to 66% at 0.15 ms. The agent must discover this non-linear mapping from experience.

4. **Balance amplitude and pulse width** — a narrow pulse at high amplitude vs a wide pulse at moderate amplitude can produce similar entrainment but different side-effect profiles. The optimal combination is not obvious.

5. **Respect the side-effect budget across the episode** — a greedy agent that maximises step-0 force by blasting 3 mA will exhaust its budget before the episode's critical mid-phase. The agent must plan temporally.

6. **Issue a compensatory motor command** — when the brain state is bad and force is degraded, the agent should also increase `motor_command` to partially compensate through effort what the brain cannot provide through smooth coordination.

This mirrors exactly what a trained neurologist or a modern closed-loop DBS programmer does — and it is not achievable by a fixed rule or a simple threshold policy.

---

## 6. Real-World Trajectory and Impact

**Immediate (this environment):**  
A trained agent policy that maps {brain state → DBS parameters} is a direct prototype for the inference firmware running on next-generation adaptive DBS implants. Medtronic Percept, Abbott Infinity, and Boston Scientific Vercise all support "sensing-and-stimulation" mode — they can read LFP from the same electrode used for stimulation and adjust parameters accordingly. An RL policy trained in this environment could, with appropriate hardware abstraction, be deployed directly.

**Near-term:**  
The environment serves as a reproducible benchmark for comparing DBS optimisation strategies:
- RL (this environment) vs PID (the Fleming ground-truth controller)
- Different RL algorithms: GRPO, PPO, SAC
- Different model architectures: MLP, Transformer, LSTM for the history window

**Long-term:**  
Patients with Parkinson's disease regain the ability to perform daily motor tasks — holding utensils, signing names, typing, walking — continuously optimised by AI rather than waiting months between clinic programming visits. The environment's `force_preserved` metric is a direct proxy for this: a 20-percentage-point improvement in mean episode force_preserved translates to a clinically meaningful improvement in the patient's ability to grasp and manipulate objects.

---

## 7. Why OpenEnv Is the Right Platform for This

The OpenEnv framework provides:

- A standardised `reset() / step() / state()` API that decouples environment logic from training code, allowing any RL algorithm or LLM agent to be plugged in.
- A containerised deployment path (Docker + Hugging Face Spaces) that ensures the environment is reproducible across machines and teams.
- A grader specification that produces deterministic 0.0–1.0 scores, enabling automated benchmarking and head-to-head comparison.
- A client library that handles WebSocket communication, allowing remote training against a deployed Space.

This environment is one of the few in the OpenEnv ecosystem that is grounded in real peer-reviewed biomedical simulation data, making it uniquely valuable as a benchmark for adaptive medical AI.
