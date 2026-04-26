# Problem Statement - MotorAssistEnv: Closed-Loop DBS Agent for Parkinson's Disease

---

## 1. Summary

Parkinson's disease breaks the basal ganglia circuit. The clearest signature is pathological beta-band synchrony in the subthalamic nucleus (STN), and that one oscillation is responsible for a lot of what patients actually feel day-to-day: the tremor, the rigidity, and the slow loss of voluntary movement that turns ordinary tasks into hard ones. Holding a cup. Reaching for a door handle. Signing your name. Fastening a button. Over time, all of these become harder than they should be.

It helps to think about Parkinson's as more than a movement disorder. It is, in a real sense, a disease of *lost agency*. Patients describe knowing exactly what they want to do but not being able to make their body do it. The signal from the brain to the muscles is scrambled along the way.

Deep Brain Stimulation (DBS) is the gold standard intervention. A surgically implanted electrode delivers high-frequency (>100 Hz) pulses into the STN, which disrupts the pathological beta synchrony and partially restores motor function. The catch is that the device has to be tuned, and tuned continuously, by a trained neurologist. Push the amplitude too low and the beta returns, the tremor grows back, and the patient stops being able to move. Push it too high and the current spreads into nearby cortical regions, which causes dyskinesia, patient discomfort, faster battery drain, and eventually tissue damage. To make things harder, the right settings keep drifting - the disease progresses, the electrode shifts a fraction of a millimetre, the patient ages. Suboptimal programming is not an edge case. It is the default state most patients live in, because they only see their neurologist every few months.

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

### 2.2 Why This Is a Perfect RL Environment

DBS programming is not just a medically important problem - it is structurally one of the most natural sequential decision problems for RL to solve. It has every property that makes RL both *necessary* and *tractable*, and none of the properties that make many RL settings contrived.

**1. Sequential decisions with compounding effects.**
Each DBS setting changes the brain state the agent sees at the next step. A poor amplitude choice at step 10 doesn't just hurt step 10 - it degrades entrainment, lets beta build back up, and forces the agent into a harder recovery for the next 140 steps. You cannot optimise each step independently. The agent has to reason across the full episode.

**2. No fixed optimal policy - context is everything.**
The right amplitude and pulse width depend on the patient profile, the current disease phase, medication timing, whether a stochastic event (tachyphylaxis, motor surge, off-medication crisis) has fired, and what the agent has already done in the last five steps. A lookup table will not work. A hard-coded schedule will not work. Only a policy that reads and reasons about combinations of signals can adapt.

**3. Partial observability with real sensor noise.**
The agent only sees what a real DBS device can measure: noisy LFP-derived beta amplitude (`beta_arv`) and surface EMG (`semg_arv`). The true STN firing state, the true tremor trajectory, the true force capability - all hidden behind Gaussian measurement noise resampled every episode. The agent must act under genuine uncertainty, not just mathematical uncertainty.

**4. Dense reward every 20 ms.**
Unlike most medical settings where outcomes arrive only at discharge or follow-up, DBS produces meaningful physiological signals at every timestep. Every action gets a reward signal. Credit assignment is tractable, reward shaping is clinically motivated rather than arbitrary, and the agent gets enough gradient signal to learn even on very short training runs.

**5. Multi-objective trade-off with no single shortcut.**
Beta suppression, tremor reduction, motor force preservation, side-effect budget, stimulation efficiency, and movement smoothness must all be satisfied simultaneously. Any single-objective strategy fails: maximum amplitude collapses the safety budget; zero stimulation collapses motor function; front-loading good early steps triggers the terminal-stability penalty. The only path to a high score is actually treating the patient across all axes at once.

**6. Non-stationary dynamics that never sit still.**
Tachyphylaxis degrades DBS effectiveness as the episode progresses. Medication wears off. Motor surges and off-medication crises fire stochastically on a per-episode seed. The patient's brain is actively changing underneath the agent's policy the entire time. A policy that works at step 1 must still work at step 150, on a brain that has been adapting and fighting back.

**7. Long-horizon budget management.**
The cumulative side-effect load accumulates across the full episode. An agent that over-stimulates in the first 30 steps to lock in early beta suppression will exhaust its safety budget before the mid-episode crises even arrive - and then spend the second half of the episode penalised on every step. The agent has to plan across time, not just react to the current observation.

**8. Clear, objective, deterministic grading.**
Success is checkable without a human rater. `force_preserved` above threshold, `side_effect_load` below budget, `beta_score` above clinical target - all computed from deterministic math on the trajectory, with no LLM-as-judge anywhere in the loop. This makes automated benchmarking, curriculum learning, and comparative evaluation straightforward.

**9. Biophysically valid dynamics that support transfer.**
Because the environment is calibrated against peer-reviewed simulation data, what the agent learns reflects real physiological relationships - not artefacts of an arbitrary reward function or a synthetic simulator. A policy that learns to suppress STN beta while managing side-effect load in MotorAssistEnv has learned something that corresponds to real DBS control. That is the foundation for eventual real-world transfer.

### 2.3 Why Not Just Use a Classical Controller

The Fleming simulation actually ships with a PID (Proportional-Integral-Derivative) closed-loop controller as its ground truth. So the obvious question is: why not just use that?

- **PID requires manual gain tuning per patient.** RL learns the adaptation directly from physiological feedback without hard-coded gains.
- **PID cannot naturally express multi-objective trade-offs.** Getting a PID to simultaneously manage force, beta, side-effect load, and smoothness requires extensive hand-engineering of the objective function. RL learns the trade-off from experience.
- **PID does not generalise across disease severity or patient profiles.** An RL agent trained on a distribution of states - responsive, balanced, refractory patients - has a genuine shot at transfer. A PID controller tuned for one severity level fails on another.
- **PID cannot reason about context.** A language model trained with RL can read "beta is rising, side-effect load is at 70% of budget, the last 5 steps showed tachyphylaxis onset" and reason about what combination of signals means and what to do next. A PID controller cannot.
- **The clinic reward function is never fully specified upfront.** What patients actually want shifts over time and differs between patients. RL gives a framework for inferring it from physiological feedback. PID requires you to write it down exactly in advance.

---

## 3. Data Foundation - The Fleming et al. (2023) Model

### 3.1 Why This Model Specifically

The whole environment is backed by one specific peer-reviewed biophysical simulation:

> **Fleming, J.E., Senneff, S. and Lowery, M.M. (2023)**
> *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*
> Journal of Neural Engineering, 20(5), p.056029.

We picked this model because no other publicly available simulation gives us all three of the following at once:

1. **It is the model that connects brain → DBS → muscle force → surface EMG in one integrated pipeline.** Most neural simulations stop at the neuron level, or at the LFP. This one keeps going all the way out to the musculoskeletal output - the actual force a patient's hand can produce. That is the thing patients care about.

2. **It produces real physical units** (mV, mA, mN) that have been validated against real patient data. The beta ARV values, tremor amplitudes, and force outputs in this environment are not arbitrary scaled units; they are clinically measured quantities.

3. **It comes with a ground-truth optimal controller** - the closed-loop PID/scheduler system published alongside the model. That gives us a concrete reference for "the best automated controller currently known," step by step, so we can actually measure whether the RL agent has learned to match or beat the clinical state of the art.

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

The `core/calibration.py` module is the interface that loads all of those simulation outputs and folds them into a `CalibratedBrainState` - a ground-truth 100-step timeline that, at every timestep, contains:

- Normalised neural signals (`beta_arv`, `tremor_arv`, `semg_arv`)
- Raw muscle force in mN, plus the `force_preserved` fraction
- The ground-truth DBS amplitude and pulse width that the Fleming controller actually used
- Physiological baselines (pre-DBS tremor, beta, and force levels)
- The full 12×15 entrainment lookup table that the agent's DBS parameter queries are resolved against

Every observation the RL agent will ever see is a transformation of this calibrated data - not a generative model, not a polynomial approximation. **The ground truth is the ground truth.**

---

## 4. System Architecture

Five layers, cleanly separated so the RL training loop never blocks on 3D rendering and the grader never touches the agent's observation path.

```
  ┌──────────────────────────────────────────────────────┐
  │  Biophysical Data Layer  (offline, fixed)            │
  │                                                      │
  │  34 CSVs: beta, tremor, force, sEMG timelines        │
  │  3 TXTs: 12×15 DBS entrainment sweep                 │
  │  Source: Fleming et al. (2023), peer-reviewed        │
  └────────────────────┬─────────────────────────────────┘
                       │  loaded once at startup
                       ▼                                            
  ┌──────────────────────────────────────────────────────┐
  │  Brain Calibrator  (runs once, cached)               │
  │                                                      │
  │  calibrate() → CalibratedBrainState                  │
  │  · 100-step physiological anchor (t = 10–12 s)       │
  │  · Normalisation bounds from data maxima             │
  │  · Pre-DBS baselines (β, tremor, force)              │
  │  · Bilinear DBS entrainment surface                  │
  └────────────────────┬─────────────────────────────────┘
                       │  calibrated state
                       ▼
  ┌──────────────────────────────────────────────────────┐     ┌─────────────────────┐
  │  MotorAssist Environment  (online, per-episode)      │────►│  3D Viewer          │
  │                                                      │     │                     │
  │  10 task specs  ·  9-component grader                │     │  MyoSuite arm model │
  │  Pydantic Action / Observation models                │     │  driven by          │
  │  reset() / step() / state()  via FastAPI             │     │  tremor_arv live    │
  │  Stochastic events · Patient profiles                │     │  (not in RL loop)   │
  └────────────────────┬─────────────────────────────────┘     └─────────────────────┘
                       │  obs (30 floats)  ↕  action (4 floats)
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │  Agent  (Qwen3-4B + LoRA, trained via GRPO)          │
  │                                                      │
  │  Reads: beta, tremor, force, device state, trends    │
  │  Writes: dbs_amplitude, pulse_width, freq, motor_cmd │
  └────────────────────┬─────────────────────────────────┘
                       │  episode return
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │  Grader  (deterministic math, no LLM-as-judge)       │
  │                                                      │
  │  Reads latent _beta_state directly                   │
  │  9 components · task-varying weights                 │
  │  score ∈ [0.0, 1.0]                                  │
  └──────────────────────────────────────────────────────┘
```

The grader reads `self._beta_state` directly; the agent reads it through Gaussian sensor noise via `_make_obs`. There is no API path from agent action to grader buffer - sensor-fooling is structurally impossible.

---

## 5. What Problems Can the Agent Solve Here?

An agent trained on MotorAssistEnv is not learning to play a game. It is learning to solve a set of concrete clinical control problems that real patients face, in a simulation grounded in real physiology.

### Problems the agent solves

**Problem 1: Initial DBS titration (easy task).**
A newly implanted patient. Beta oscillations are elevated but stable. The agent must find the right stimulation level without over-stimulating - keeping the patient in the therapeutic window without triggering dyskinesia. This is the first thing a DBS programmer does in every clinical visit. The agent learns it by exploring the reward landscape across 36 steps.

**Problem 2: Symptom rescue without adverse effects (medium task).**
Mid-episode, symptoms flare. Beta rises. Tremor increases. The agent has to push stimulation up - but not so far that it exhausts the side-effect budget. Then it has to back off when symptoms stabilise. This is the rescue-and-recover pattern that real adaptive DBS trials demonstrate. The agent learns when to intervene aggressively and when to pull back.

**Problem 3: Multi-crisis management in a refractory patient (hard task).**
Four overlapping crises fire simultaneously across a 150-step episode: tachyphylaxis (DBS effectiveness degrades), off-medication emergency (beta surges), dyskinesia spikes (side-effect budget threatens to collapse), and motor surges (voluntary motor output becomes erratic). The patient's brain has already stopped responding to the amplitude levels that worked 30 seconds ago. The agent must continuously re-calibrate its strategy, ration the safety budget, and sustain clinical outcomes across a refractory brain that is actively fighting back.

**Problem 4: Patient-class generalisation.**
Different patients have fundamentally different physiology. The fragile patient has 1.4× side-effect sensitivity - the usable amplitude window is roughly half that of a standard patient. The refractory patient barely responds to stimulation; more amplitude doesn't help, only pulsed strategies with real rest periods do. The `personalization_generalization` task resets to a different patient profile every episode. The agent must read the reset context and adapt from step 1 without any per-patient history.

**Problem 5: Clinical scenario reasoning.**
Some tasks require the agent to recognize and respond to specific clinical contexts, not just react to biomarker levels:
- During an **exercise bout**, motor demand spikes and DBS must track it. Stopping stimulation mid-exertion is a hard clinical failure.
- During a **medication interaction**, the agent must recognise L-DOPA wearing off from the beta trend and act before tremor becomes uncontrollable.
- During a **nocturnal transition**, targets tighten as the patient winds down toward sleep - less stimulation is needed, but tighter biomarker control is required simultaneously.
- In the **surgical follow-up** window, the first 25% of the episode has a hard amplitude ceiling of 0.6 mA. Any violation is catastrophic. The agent must be conservative early, then gradually titrate up as the microlesion window closes.

### What a well-trained agent looks like

By the time an agent is doing well across these tasks, it has picked up the following skills - which map almost line-for-line to what a trained DBS neurologist does in the clinic:

1. **Phase recognition.** Reading `beta_arv`, `tremor_arv`, and their 5-step trends to identify where in the episode the patient currently is and what trajectory they are on.
2. **Proportional early intervention.** Front-loading stimulation to slow the tremor ramp-up, instead of waiting until tremor is large and recovery is expensive.
3. **Non-linear entrainment navigation.** Discovering from experience that going from 1.0 mA to 1.25 mA at 0.15 ms pulse width jumps cortical entrainment from 36% to 66% - a discontinuity that no linear model would predict.
4. **Amplitude/pulse-width decoupling.** Recognising that the same entrainment can be achieved at very different side-effect costs by trading amplitude against pulse width - and learning to choose the combination that protects the safety budget.
5. **Budget-aware long-horizon planning.** Not spending safety headroom in the first 30 steps to look good early, because the mid-episode crises will need that headroom.
6. **Compensatory motor coordination.** When the brain state is bad and DBS can't fully compensate, pushing `motor_command` up to partially restore voluntary motor output through effort.

That is not a checklist of engineered rules. It is a learned policy - and it is exactly what RL is for.

---

## 6. Real-World Trajectory and Impact

**Immediate (this environment).**
A trained agent policy that maps {brain state → DBS parameters} is essentially a prototype for the inference firmware that will run on next-generation adaptive DBS implants. Medtronic Percept, Abbott Infinity, and Boston Scientific Vercise already support a "sensing-and-stimulation" mode - they can read LFP off the same electrode they use to stimulate, and adjust parameters accordingly. With the right hardware abstraction, a policy trained in this environment could be deployed directly onto that kind of device.

**Near-term.**
The environment also doubles as a reproducible benchmark for comparing DBS optimisation strategies against each other:

- RL (this environment) vs PID (the Fleming ground-truth controller)
- Different RL algorithms - GRPO, PPO, SAC
- Different model architectures over the history window - MLP, Transformer, LSTM

**Long-term.**
The end state we are aiming at is patients with Parkinson's disease getting back the ability to do daily motor tasks - holding utensils, signing their name, typing, walking - with stimulation that is being continuously optimised by AI in the background, instead of waiting months at a time for the next clinic programming visit. The `force_preserved` metric in this environment is a fairly direct proxy for that: a 20-percentage-point improvement in mean episode `force_preserved` translates to a clinically meaningful improvement in how well a patient can grasp and manipulate objects.

---

## 7. Why OpenEnv Is the Right Platform for This

A few things about OpenEnv made it the natural fit:

- It gives you a standardised `reset() / step() / state()` API, which cleanly separates the environment's logic from the training code. Any RL algorithm, or any LLM agent, can be plugged in on top.
- It defines a containerised deployment path (Docker + Hugging Face Spaces) so the environment is reproducible across machines and across teams without anyone having to babysit dependencies.
- It standardises the grader interface around deterministic 0.0–1.0 scores, which makes automated benchmarking and head-to-head comparison straightforward.
- It ships a client library that handles WebSocket communication, so you can train remotely against a deployed Space without having to build that plumbing yourself.

On top of all of that, this is one of the very few environments in the OpenEnv ecosystem that is grounded in real, peer-reviewed biomedical simulation data - which is what makes it valuable as a serious benchmark for adaptive medical AI, rather than just another simulator.
