# Architecture: MotorAssistEnv

## 1. Overview

MotorAssistEnv is designed as a **modular RL system** that separates:
- environment dynamics
- reward / verification
- agent learning
- curriculum adaptation

The goal is to create a **self-improving RL loop** while keeping the environment verifiable, stable, and hackathon-friendly.

---

## 2. High-Level Architecture

```
+-------------------------------------------------------------+
|                    SELF-IMPROVING LOOP                      |
+-------------------------------------------------------------+

   +-------------------+
   | Curriculum Engine |
   | (difficulty ctrl) |
   +---------+---------+
             |
             v
   +-------------------+
   | Impairment Model  |
   | (tremor, delay)   |
   +---------+---------+
             |
             v
   +---------------------------+
   |   MotorAssist Environment |
   | (state, dynamics, tasks)  |
   +---------+-----------------+
             |
             v
   +---------------------------+
   | RL Agent (PPO / GRPO)     |
   | learns correction policy  |
   +---------+-----------------+
             |
             v
   +---------------------------+
   | Reward & Verifier System  |
   | (deterministic metrics)   |
   +---------+-----------------+
             |
             v
   +---------------------------+
   | Training Loop (TRL)       |
   | updates policy            |
   +---------------------------+

             ^
             |
   +---------------------------+
   | Metrics & Logging         |
   | (success, stability, etc) |
   +---------------------------+
```

---

## 3. Key Design Decision: No LLM-as-Judge

### ❌ Why NOT use LLM judge:
- Non-deterministic → bad for evaluation
- Easy to game → reward hacking risk
- Slow → violates hackathon constraints
- Judges prefer **verifiable rewards**

### ✅ What we use instead:
- Deterministic reward functions
- Physics + metrics based evaluation

---

## 4. Core Components

---

### 4.1 Environment Layer

Handles:
- state transitions
- physics / movement
- impairment injection

```
state_t → action → impaired_action → next_state
```

Includes:
- tremor noise
- delay
- stochastic disturbances

---

### 4.2 Impairment Model

Simulates Parkinson-like effects:

```
u_t = a_t + tremor + noise + delay
```

Types:
- high-frequency tremor
- latency
- freezing events (rare)

---

### 4.3 RL Agent

Learns:

```
correction = f(state)
```

Final control:

```
u_t = impaired_signal + correction
```

---

### 4.4 Reward & Verifier

Deterministic evaluation:

- distance to target
- stability (variance)
- smoothness
- success condition

No subjective scoring.

---

### 4.5 Curriculum Controller

Adjusts difficulty dynamically:

- tremor ↑ as agent improves
- delay ↑
- task complexity ↑

---

### 4.6 Training Loop

- PPO / GRPO
- rollout collection
- reward computation
- policy update

---

## 5. Data Flow (Step Function)

```
obs_t
  ↓
agent(action)
  ↓
impaired_action = apply_impairment(action)
  ↓
next_state = physics(impaired_action)
  ↓
reward = compute_reward(next_state)
  ↓
return (next_state, reward, done)
```

---

## 6. Stochasticity Design

We introduce **controlled randomness**:

| Component | Randomized |
|----------|-----------|
| Tremor amplitude | ✓ |
| Delay | ✓ |
| Noise | ✓ |
| Freeze events | ✓ (rare) |

### Why:
- improves robustness
- avoids overfitting
- mimics real-world variability

---

## 7. Safety & Anti-Hacking Layer

- action clipping
- penalty for oscillation
- penalty for inactivity
- timeout enforcement

---

## 8. Metrics Layer

Tracked continuously:

- success rate
- stability score
- smoothness
- tremor reduction %
- completion time

---

## 9. Optional Extensions (Advanced)

### 9.1 Personalization Layer
- different patient profiles

### 9.2 Adversarial Noise Generator
- increases difficulty where agent fails

### 9.3 Hierarchical Tasks
- multi-phase control

---

## 10. Why This Architecture Works

This system is:

### ✔ Modular
Each component is independent and testable

### ✔ Verifiable
No black-box reward

### ✔ Scalable
Can extend to biomechanics later

### ✔ Learnable
Curriculum ensures non-zero reward

### ✔ Realistic
Includes noise, delay, stochasticity

---

## 11. Final Insight

This is not a simulation of the brain.

It is a system that learns:

> **how to act under imperfect control**

That is what makes it both:
- scientifically meaningful
- and practically useful
