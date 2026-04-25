# Project Context: MotorAssistEnv

## 1. What we are building

We are building an OpenEnv-compatible RL environment for **closed-loop Deep Brain Stimulation (DBS) optimization** in Parkinson's Disease.

The environment challenges an agent to behave as a real-time clinical programmer. It must observe noisy, pathological brain signals (beta oscillations and tremor) and tune an electrical implant to restore the patient's muscle function without causing debilitating side effects.

This is not a toy robot simulation. It is a genuine BCI (Brain-Computer Interface) challenge backed by peer-reviewed biophysical neuronal simulation data (Fleming et al. 2023).

## 2. Why this idea was chosen

We wanted an environment that is:
- **Phenomenally impactful:** Affects a real disease that hinders 10 million people.
- **Scientifically rigorous:** Avoids made-up physics, relying instead on heavy neuroscience computation.
- **Hackathon-friendly:** Runs instantaneously per step for rapid agent training, fully decoupling heavy 3D visualizations from the core RL loop.
- **Novel (Wildcard Theme):** Explores adaptive neuro-implants rather than standard Web automation or robotic arms.

The OpenEnv hackathon guidelines demand real-world tasks, strict `reset`/`step` APIs, deterministic scoring, and measurable progress. This project maps perfectly to those requirements by treating DBS parameter tuning as a sequential, multi-objective optimization problem.

## 3. The Grand Pivot: From Robotics to Neuroscience

Originally, this project attempted to train an agent to control a 3D robotic arm in MuJoCo/MyoSuite while fighting synthetic tremor. 

**We pivoted.**
Training an agent on heavy 3D muscle physics is too slow for a fast-paced RL hackathon. Furthermore, the true innovation was the Parkinsonian neuro-simulation we had already connected. 

**The new hybrid strategy:**
1. **The Core RL Environment** operates purely on the 1D clinical brain data. This makes training blistering fast and scientifically accurate.
2. **The 3D Demo** is entirely frontend. It watches the AI suppress the brain tremor in the backend, and visually smooths out a 3D robotic arm on the screen in real-time.

This separation of concerns provides a massive technical edge. We get lightning-fast OpenAI Gym compatibility alongside a jaw-dropping visual pitch.

## 4. What makes winning projects look strong

Winning projects exhibit:
- **A real system, not a toy:** We use actual cortical entrainment tensors.
- **Clean interaction loops:** Pure JSON `[amplitude, pulse_width]` actions.
- **Verifiable objectives:** 0.0-1.0 Graders that can't be tricked by LLM-Judges.
- **Curriculum design:** Easy (early stage tremor) $\rightarrow$ Hard (full clinical progression).

## 5. Final project narrative

MotorAssistEnv establishes a bridge between Reinforcement Learning and Clinical Neuromodulation. By training agents to stabilize pathological brain activity in a high-fidelity simulator, we lay the groundwork for next-generation, self-updating neural implants that adapt to human needs in real-time.
