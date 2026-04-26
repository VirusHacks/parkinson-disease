# Problem Statement — MotorAssistEnv: Closed-Loop DBS Agent for Parkinson's Disease

---

## 1. Summary

Parkinson's disease breaks the basal ganglia circuit. The clearest signature is pathological beta-band synchrony in the subthalamic nucleus (STN), and that one oscillation is responsible for a lot of what patients actually feel day-to-day: the tremor, the rigidity, and the slow loss of voluntary movement that turns ordinary tasks into hard ones. Holding a cup. Reaching for a door handle. Signing your name. Fastening a button. Over time, all of these become harder than they should be.

It helps to think about Parkinson's as more than a movement disorder. It is, in a real sense, a disease of *lost agency*. Patients describe knowing exactly what they want to do but not being able to make their body do it. The signal from the brain to the muscles is scrambled along the way.

Deep Brain Stimulation (DBS) is the gold standard intervention. A surgically implanted electrode delivers high-frequency (>100 Hz) pulses into the STN, which disrupts the pathological beta synchrony and partially restores motor function. The catch is that the device has to be tuned, and tuned continuously, by a trained neurologist. Push the amplitude too low and the beta returns, the tremor grows back, and the patient stops being able to move. Push it too high and the current spreads into nearby cortical regions, which causes dyskinesia, patient discomfort, faster battery drain, and eventually tissue damage. To make things harder, the right settings keep drifting — the disease progresses, the electrode shifts a fraction of a millimetre, the patient ages. Suboptimal programming is not an edge case. It is the default state most patients live in, because they only see their neurologist every few months.

**MotorAssistEnv** turns this into a sequential decision problem for a reinforcement learning agent. The agent plays the role of an autonomous closed-loop BCI (Brain-Computer Interface) programmer: every 20 ms it looks at the patient's current brain state and decides what DBS amplitude and pulse width to deliver, trying to keep motor function intact while staying within a cumulative side-effect budget.

This is not a toy simulation. Every observation the agent receives comes out of a peer-reviewed biophysical neural network model (Fleming et al. 2023) of the exact circuits that Parkinson's disrupts, simulated down to individual neurons and synaptic connections.

---

## 2. Why This Problem Matters

### 2.1 Clinical Reality

The clinical need behind this is real and unmet:

- More than **1 million patients** in the United States have Parkinson's disease, and an estimated 10 million worldwide.
- **Deep Brain Stimulation** is already implanted in roughly 50,000+ patients, and the number keeps growing as indications expand.
- Ask neurologists what the biggest barrier to better outcomes is *after* a successful implant, and the answer is almost always the same: **suboptimal DBS programming**. The hardware works. The battery is charged. The settings are simply wrong, and the patient suffers in the gap.
- A single DBS programming visit takes one to two hours of clinic time, and most centres can only schedule them every three to six months. Between visits, the patient is stuck with whatever the last programming session left behind, even when those settings have stopped being appropriate.
- **Adaptive (closed-loop) DBS** is where the field is headed. The vision is a device that listens to the brain continuously and adjusts stimulation on its own, in real time, personalised to the patient's current state. RL is a very natural fit for learning that policy.

### 2.2 The RL Problem Structure

DBS programming is also genuinely interesting from a pure RL perspective, because it has all the structural properties that make RL both necessary and tractable:

- **Sequential action.** Each setting you choose changes the brain state you'll see next. Actions compound across time.
- **Non-stationary disturbances.** Tremor amplitude climbs over the course of an episode. A fixed policy will eventually fail; the agent has to keep adapting.
- **Partial observability.** The agent only sees what real DBS hardware can measure: local field potentials (LFP, surfaced as `beta_arv`) and surface EMG (`semg_arv`). Individual neuron firing patterns are hidden.
- **Multi-objective trade-off.** The agent has to keep motor force up, suppress oscillation, and stay inside the side-effect budget all at the same time. There isn't a single scalar that captures clinical success cleanly; multiple criteria have to be jointly satisfied.
- **Dense feedback.** Unlike many medical settings where you only see an outcome at discharge, DBS produces meaningful physiological signals every 20 ms. That makes dense reward shaping both possible and clinically grounded.
- **Clear programmatic grading.** Success can be checked objectively — `force_preserved` above a threshold, `side_effect_load` below a budget — without needing a human to rate anything.

### 2.3 Why Not Just Use a Classical Controller

The Fleming simulation actually ships with a PID (Proportional-Integral-Derivative) closed-loop controller as its ground truth. So the obvious question is: why not just use that? A few reasons:

- PID requires careful manual tuning of gain parameters for each patient. RL can learn that adaptation directly from data.
- PID does not naturally express the multi-objective trade-off between force preservation and side-effect management. To get there, you have to hand-engineer the objective function.
- PID does not generalise across disease severity levels. An RL agent trained over a distribution of states has a real shot at transferring.
- In the actual clinic, the "reward function" — what the patient really wants — shifts over time and is never fully specified up front. RL gives you a framework for inferring it from physiological feedback instead of writing it down.

---

## 3. Data Foundation — The Fleming et al. (2023) Model

### 3.1 Why This Model Specifically

The whole environment is backed by one specific peer-reviewed biophysical simulation:

> **Fleming, J.E., Senneff, S. and Lowery, M.M. (2023)**
> *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*
> Journal of Neural Engineering, 20(5), p.056029.

We picked this model because no other publicly available simulation gives us all three of the following at once:

1. **It is the model that connects brain → DBS → muscle force → surface EMG in one integrated pipeline.** Most neural simulations stop at the neuron level, or at the LFP. This one keeps going all the way out to the musculoskeletal output — the actual force a patient's hand can produce. That is the thing patients care about.

2. **It produces real physical units** (mV, mA, mN) that have been validated against real patient data. The beta ARV values, tremor amplitudes, and force outputs in this environment are not arbitrary scaled units; they are clinically measured quantities.

3. **It comes with a ground-truth optimal controller** — the closed-loop PID/scheduler system published alongside the model. That gives us a concrete reference for "the best automated controller currently known," step by step, so we can actually measure whether the RL agent has learned to match or beat the clinical state of the art.

### 3.2 What the Simulation Modelled

The simulation runs for roughly 75 seconds of simulated time and includes:

- **Cortical layer.** ~100 pyramidal neurons with full Hodgkin-Huxley dynamics (sodium, potassium, M-current, leak channels), simulated in NEURON, the gold-standard single-neuron simulator.
- **Basal Ganglia.** Subthalamic Nucleus (STN, 100 neurons), GPe (100 neurons), and GPi (100 neurons), all with biophysical ion channel models.
- **Thalamus.** Mediodorsal and ventrolateral nuclei, modelling how tremor signals are relayed back up to cortex.
- **Spinal cord.** A depressing spinal motoneuron pool driven at tremor frequency.
- **Musculoskeletal.** Muscle force computed from motoneuron firing rates using a physiological Hill-type model.
- **DBS electrode.** Extracellular stimulation modelled with finite-element methods so that current spread through tissue is captured.
- **Total connections.** More than 5 million individual synaptic connections, modelled stochastically.

The outputs of that simulation (stored under `parkinsons_Motor/fleming-model-based-brain/Model_Results/`) include:

- 102 STN voltage traces (`.mat` files)
- 102 motoneuron voltage traces and spike times (`.mat` files)
- 34 CSV files of controller signals, sampled at 100 timesteps (t = 10.02–12.00 s)
- A 12×15 DBS parameter sweep (`Collaterals_Entrained_values.txt`) that maps every combination of amplitude and pulse width to its cortical entrainment fraction.

### 3.3 What "Calibration" Means Here

The `core/calibration.py` module is the interface that loads all of those simulation outputs and folds them into a `CalibratedBrainState` — a ground-truth 100-step timeline that, at every timestep, contains:

- Normalised neural signals (`beta_arv`, `tremor_arv`, `semg_arv`)
- Raw muscle force in mN, plus the `force_preserved` fraction
- The ground-truth DBS amplitude and pulse width that the Fleming controller actually used
- Physiological baselines (pre-DBS tremor, beta, and force levels)
- The full 12×15 entrainment lookup table that the agent's DBS parameter queries are resolved against

Every observation the RL agent will ever see is a transformation of this calibrated data — not a generative model, not a polynomial approximation. **The ground truth is the ground truth.**

---

## 4. System Architecture

Five layers, cleanly separated so the RL training loop never blocks on 3D rendering and the grader never touches the agent's observation path.

```
  ┌──────────────────────────────────────────────────────┐
  │  Biophysical Data Layer  (offline, fixed)             │
  │                                                       │
  │  34 CSVs: beta, tremor, force, sEMG timelines        │
  │  3 TXTs: 12×15 DBS entrainment sweep                 │
  │  Source: Fleming et al. (2023), peer-reviewed         │
  └────────────────────┬─────────────────────────────────┘
                       │  loaded once at startup
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │  Brain Calibrator  (runs once, cached)                │
  │                                                       │
  │  calibrate() → CalibratedBrainState                  │
  │  · 100-step physiological anchor (t = 10–12 s)       │
  │  · Normalisation bounds from data maxima              │
  │  · Pre-DBS baselines (β, tremor, force)               │
  │  · Bilinear DBS entrainment surface                   │
  └────────────────────┬─────────────────────────────────┘
                       │  calibrated state
                       ▼
  ┌──────────────────────────────────────────────────────┐     ┌─────────────────────┐
  │  MotorAssist Environment  (online, per-episode)       │────►│  3D Viewer          │
  │                                                       │     │                     │
  │  10 task specs  ·  9-component grader                │     │  MyoSuite arm model │
  │  Pydantic Action / Observation models                │     │  driven by          │
  │  reset() / step() / state()  via FastAPI             │     │  tremor_arv live     │
  │  Stochastic events · Patient profiles                │     │  (not in RL loop)   │
  └────────────────────┬─────────────────────────────────┘     └─────────────────────┘
                       │  obs (30 floats)  ↕  action (4 floats)
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │  Agent  (Qwen3-4B + LoRA, trained via GRPO)          │
  │                                                       │
  │  Reads: beta, tremor, force, device state, trends    │
  │  Writes: dbs_amplitude, pulse_width, freq, motor_cmd │
  └────────────────────┬─────────────────────────────────┘
                       │  episode return
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │  Grader  (deterministic math, no LLM-as-judge)       │
  │                                                       │
  │  Reads latent _beta_state directly                   │
  │  9 components · task-varying weights                 │
  │  score ∈ [0.0, 1.0]                                  │
  └──────────────────────────────────────────────────────┘
```

The grader reads `self._beta_state` directly; the agent reads it through Gaussian sensor noise via `_make_obs`. There is no API path from agent action to grader buffer — sensor-fooling is structurally impossible.

---

## 5. What the Agent Must Learn

An agent that does well on this environment will, by the time it converges, have picked up a few specific skills:

1. **Recognise the current disease phase.** `beta_arv` and `tremor_arv` together encode where in the episode the patient currently is. Low tremor early on means a subtle DBS push is enough. High tremor late means the agent needs to intervene aggressively.

2. **Use DBS proportionally and early.** From the ground-truth data we already know that the optimal policy front-loads stimulation to slow the tremor's ramp-up, instead of waiting and reacting after tremor has already grown large.

3. **Navigate the bilinear entrainment surface.** The 12×15 lookup table is far from linear. Very low amplitude (0–0.5 mA) gives near-zero entrainment, but going from 1.0 mA to 1.25 mA at 0.15 ms jumps entrainment from 36% to 66%. The agent has to discover this non-linear mapping the hard way, from experience.

4. **Balance amplitude and pulse width.** A narrow pulse at high amplitude and a wider pulse at moderate amplitude can produce roughly the same entrainment but very different side-effect profiles. There is no obvious "right" combination — it has to be learned.

5. **Respect the side-effect budget across the full episode.** A greedy agent that maxes out step-0 force by blasting 3 mA will burn through its budget long before the episode's critical mid-phase. The agent has to plan over time, not just react.

6. **Issue a compensatory motor command.** When the brain state is bad and force is degraded, the agent should also push `motor_command` up, partially compensating through effort for what the brain can no longer provide through smooth coordination.

That list is, almost line for line, what a trained neurologist or a modern closed-loop DBS programmer does in the clinic — and it isn't something a fixed rule or a simple threshold policy can pull off.

---

## 6. Real-World Trajectory and Impact

**Immediate (this environment).**
A trained agent policy that maps {brain state → DBS parameters} is essentially a prototype for the inference firmware that will run on next-generation adaptive DBS implants. Medtronic Percept, Abbott Infinity, and Boston Scientific Vercise already support a "sensing-and-stimulation" mode — they can read LFP off the same electrode they use to stimulate, and adjust parameters accordingly. With the right hardware abstraction, a policy trained in this environment could be deployed directly onto that kind of device.

**Near-term.**
The environment also doubles as a reproducible benchmark for comparing DBS optimisation strategies against each other:

- RL (this environment) vs PID (the Fleming ground-truth controller)
- Different RL algorithms — GRPO, PPO, SAC
- Different model architectures over the history window — MLP, Transformer, LSTM

**Long-term.**
The end state we are aiming at is patients with Parkinson's disease getting back the ability to do daily motor tasks — holding utensils, signing their name, typing, walking — with stimulation that is being continuously optimised by AI in the background, instead of waiting months at a time for the next clinic programming visit. The `force_preserved` metric in this environment is a fairly direct proxy for that: a 20-percentage-point improvement in mean episode `force_preserved` translates to a clinically meaningful improvement in how well a patient can grasp and manipulate objects.

---

## 7. Why OpenEnv Is the Right Platform for This

A few things about OpenEnv made it the natural fit:

- It gives you a standardised `reset() / step() / state()` API, which cleanly separates the environment's logic from the training code. Any RL algorithm, or any LLM agent, can be plugged in on top.
- It defines a containerised deployment path (Docker + Hugging Face Spaces) so the environment is reproducible across machines and across teams without anyone having to babysit dependencies.
- It standardises the grader interface around deterministic 0.0–1.0 scores, which makes automated benchmarking and head-to-head comparison straightforward.
- It ships a client library that handles WebSocket communication, so you can train remotely against a deployed Space without having to build that plumbing yourself.

On top of all of that, this is one of the very few environments in the OpenEnv ecosystem that is grounded in real, peer-reviewed biomedical simulation data — which is what makes it valuable as a serious benchmark for adaptive medical AI, rather than just another simulator.
