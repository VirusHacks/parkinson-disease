# Problem Statement — MotorAssistEnv: Closed-Loop DBS Agent for Parkinson's Disease

## 1. Summary

Parkinson's disease disrupts the basal ganglia circuit through pathological beta-band synchrony in the subthalamic nucleus (STN). This beta oscillation drives tremor, rigidity, and the progressive loss of voluntary motor function that makes everyday tasks — holding a cup, reaching for a door, writing — increasingly impossible.

Deep Brain Stimulation (DBS) is the gold standard intervention. A surgically implanted electrode delivers high-frequency electrical pulses to the STN, suppressing the beta oscillation and partially restoring motor function. The challenge: DBS must be tuned continuously by a trained neurologist. Too little amplitude → beta persists, tremor grows. Too much → cortical side effects, battery drain, and patient discomfort. Suboptimal settings are the norm, not the exception.

**MotorAssistEnv** frames this as a sequential decision problem for a reinforcement learning agent. The agent acts as an autonomous closed-loop BCI (Brain-Computer Interface) programmer: it observes the patient's real-time brain state and issues DBS parameters step-by-step to maximise motor function while managing side effects.

---

## 2. Why This Problem Matters

This is not a toy benchmark. It targets a real unmet clinical need:

- **1 million+ patients** in the US alone have Parkinson's disease. DBS is used in ~50,000+ patients worldwide.
- **Suboptimal DBS programming** is identified as the #1 barrier to better outcomes. Programming is manual, expensive, requires specialist time, and degrades between clinic visits.
- A learned closed-loop policy that automatically adapts stimulation could **improve patient quality of life continuously** — not just during clinic visits.

From an RL perspective this environment provides:
- Dense, step-wise feedback grounded in real neuroscience measurements
- Partially observable dynamics (the agent cannot directly measure interneuron firing)
- Non-stationary disturbances (tremor escalates deterministically from real simulation data)
- A multi-objective trade-off: suppress oscillation, preserve force, limit side effects
- Clear programmatic success criteria graded at episode end

---

## 3. Data Foundation — The Fleming Model

The environment is backed by the **Fleming et al. (2023)** biophysical simulation:

> Fleming, J.E., Senneff, S. and Lowery, M.M. (2023).
> *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease.*
> Journal of Neural Engineering, 20(5), p.056029.

This peer-reviewed simulation models:
- ~100 cortical neurons, ~100 STN neurons, GPe, GPi, Thalamus, and a motoneuron pool
- 5+ million individual synaptic connections
- A closed-loop DBS controller running for ~75 seconds of simulated brain time
- Real physical units throughout: mV, mA, mN

The calibration pipeline (`brain_calibrator.py`) extracts a 100-step ground-truth trajectory (t=10.02–12.00 s, 20 ms intervals) from the simulation's CSV outputs. **Every number in the agent's observation came from a peer-reviewed neuroscience model — not synthetic approximation.**

---

## 4. Agent's Role

The RL agent replaces the human neurologist. At each 20 ms step it receives the patient's brain state and must output:

- `dbs_amplitude` (mA): How much electrical current to deliver to the STN electrode
- `dbs_pulse_width` (ms): Width of each stimulation pulse (controls spatial spread)
- `motor_command` (normalised float): The intended voluntary motor output (reaching/holding command)

The agent's DBS choices directly affect the _next_ step's brain state via a bilinear entrainment lookup derived from the 12×15 DBS parameter sweep in the Fleming simulation.

---

## 5. Environment Architecture

```
fleming-model-based-brain (real simulation data)
         │
         └── brain_calibrator.py
                  │
                  └── 100-step CalibratedBrainState
                           │
                           └── ParkinsonsMotorEnvironment (OpenEnv)
                                    │
                                    ├── tasks/      (3 clinical scenarios)
                                    ├── graders/    (deterministic 0.0–1.0 scores)
                                    └── inference.py (LLM agent loop)

Visualisation (separate, demo only):
  static/myosuite_demo/  →  /viewer endpoint
  Tremor severity → 3D arm jitter in WebGL
  As agent suppresses beta → arm smoothes out → patient performs task
```

The MyoSuite 3D visualisation runs separately from the RL loop and reads real-time brain state from the FastAPI server to drive visual tremor in the arm model.

---

## 6. What a Successful Agent Learns

An agent that achieves high reward on this environment will have learned:

1. **Recognise disease state** — rising `beta_arv` + growing `tremor_arv` = deteriorating condition
2. **Use DBS proportionally** — more tremor/beta requires more stimulation, but not blindly
3. **Find effective DBS parameters** — amplitude and pulse width together determine cortical entrainment
4. **Balance treatment and side effects** — sustained high amplitude exhausts the side-effect budget
5. **Compensate with motor command** — when brain state is bad, issue a stronger voluntary signal

This mirrors exactly what a trained neurologist or a modern closed-loop DBS programmer does.

---

## 7. Real-World Trajectory and Impact

**Immediate:** A trained agent policy that maps {brain state → DBS parameters} is a direct prototype for firmware running on next-generation adaptive DBS implants (Medtronic Percept, Abbott Infinity).

**Near-term:** The environment serves as a reproducible benchmark for comparing DBS optimisation strategies — RL vs PID vs model-predictive control.

**Long-term:** Patients with Parkinson's disease regain the ability to perform daily motor tasks — holding utensils, signing names, typing — continuously optimised by AI rather than waiting months between clinic programming visits.
