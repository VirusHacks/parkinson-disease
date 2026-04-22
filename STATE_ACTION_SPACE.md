# State and Action Space Design: MotorAssistEnv

## 1. Overview

The effectiveness of an RL environment depends heavily on how the state and action spaces are defined.

The design must balance:
- realism,
- learnability,
- and stability.

If the state is incomplete → agent cannot learn  
If the action space is too large → training becomes unstable  
If too simple → environment becomes unrealistic  

---

## 2. State Space (Observation)

The agent observes a structured state vector:

\[
s_t = (x_t, v_t, g_t, \hat{u}_{t-1}, \text{impairment params})
\]

---

## 2.1 Core Components

### Position
\[
x_t \in \mathbb{R}^d
\]

Current end-effector position

---

### Velocity
\[
v_t \in \mathbb{R}^d
\]

Captures motion dynamics

---

### Target
\[
g_t \in \mathbb{R}^d
\]

Goal position

---

### Previous Action / Control
\[
\hat{u}_{t-1}
\]

Helps model temporal consistency

---

### Impairment Parameters

- tremor amplitude
- delay factor
- noise level

---

## 2.2 Optional (Advanced)

- history window (last k states)
- phase indicator (for multi-step tasks)
- success flags

---

## 2.3 Design Rationale

- position + target → task objective
- velocity → dynamics awareness
- previous action → smooth control
- impairment params → adaptation

---

## 3. Partial Observability

In real systems, not all internal states are visible.

We optionally simulate this by:
- adding noise to observations
- hiding some variables

---

### Why this matters

Encourages:
- robust policies
- better generalization

---

## 4. Action Space

We define a continuous control action:

\[
a_t \in \mathbb{R}^d
\]

---

## 4.1 Interpretation

Action represents a **correction signal** applied to movement.

---

## 4.2 Action Constraints

\[
a_t \in [-a_{\max}, a_{\max}]
\]

---

## 4.3 Executed Control

Due to impairment:

\[
u_t = f(a_t, \text{noise}, \text{delay}, \text{tremor})
\]

---

## 4.4 Impairment Model

The actual executed action is:

\[
u_t = a_t + \epsilon_t + \delta_t
\]

where:
- \( \epsilon_t \sim \text{noise} \)
- \( \delta_t \sim \text{tremor} \)

---

## 5. Transition Function

\[
s_{t+1} = f(s_t, u_t)
\]

---

## 5.1 Dynamics

- position updated via control
- velocity updated accordingly
- noise injected

---

## 6. Temporal Structure

The environment is:

- sequential
- time-dependent
- partially observable

---

## 7. Action Design Tradeoffs

### Continuous vs Discrete

| Choice | Reason |
|------|------|
| Continuous | realistic motor control |
| Discrete | easier learning |

We choose **continuous** for realism.

---

### Direct vs Structured Actions

We can extend to:

```json
{
  "type": "stabilize",
  "intensity": 0.3
}