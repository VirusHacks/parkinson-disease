# Tasks Design: MotorAssistEnv

## 1. Overview

The task design in MotorAssistEnv follows a curriculum-based progression from simple to complex motor control problems.

This is necessary because reinforcement learning requires:
- non-zero success probability,
- gradual exploration,
- and increasing difficulty over time.

If tasks are too difficult initially, the agent will never receive meaningful reward and learning will stall. :contentReference[oaicite:0]{index=0}

The environment is structured into **three levels of difficulty**:
- Easy (short horizon, high signal)
- Medium (moderate horizon, partial observability)
- Hard (long horizon, multi-phase tasks)

---

## 2. Task Taxonomy

We define tasks along two axes:

### 2.1 Difficulty
- Easy
- Medium
- Hard

### 2.2 Horizon Length
- Short-horizon (single objective)
- Medium-horizon (multi-step but linear)
- Long-horizon (multi-phase with dependencies)

---

## 3. Easy Tasks (Bootstrap Phase)

### 3.1 Objective
Learn basic stabilization and control under impairment.

### 3.2 Example Tasks

#### Task E1: Static Stabilization
- Maintain position at a fixed target
- No movement required
- Only tremor + noise present

#### Task E2: Micro Correction
- Small displacement from target
- Agent must correct and stabilize

---

### 3.3 Properties

| Property | Value |
|--------|------|
| Horizon | Short |
| Noise | Low |
| Delay | None / minimal |
| Success Probability | High |

---

### 3.4 Why this stage matters

- Ensures early reward signal
- Teaches agent control primitives
- Prevents collapse due to sparse reward

---

## 4. Medium Tasks (Control + Movement)

### 4.1 Objective
Combine movement + stabilization.

---

### 4.2 Example Tasks

#### Task M1: Point-to-Point Reaching
- Move from start → target
- Moderate tremor
- Minor delay

#### Task M2: Hold After Reach
- Reach target
- Maintain position for N steps

---

### 4.3 Properties

| Property | Value |
|--------|------|
| Horizon | Medium |
| Noise | Moderate |
| Delay | Present |
| Success Probability | Medium |

---

### 4.4 Key challenges

- overshooting
- oscillation
- instability after reaching

---

## 5. Hard Tasks (Real-world Simulation)

### 5.1 Objective
Perform multi-step real-world motor sequences.

---

### 5.2 Example Tasks

#### Task H1: Reach → Stabilize → Hold
- Sequential control phases
- Must not lose stability

#### Task H2: Pick-and-Place (Abstracted)
- Move to object
- stabilize
- move to target location

#### Task H3: Continuous Control Task
- follow moving target trajectory

---

### 5.3 Properties

| Property | Value |
|--------|------|
| Horizon | Long |
| Noise | High |
| Delay | Significant |
| Stochastic Events | Yes |

---

### 5.4 Key challenges

- delayed credit assignment
- compounding errors
- stability over time

---

## 6. Stochastic Variations (Realism Layer)

To avoid overfitting:

- random tremor amplitude
- random delay
- noise injection
- occasional “freeze” events

---

### Why this matters

Real-world motor systems are non-deterministic.

Adding controlled stochasticity:
- improves robustness
- prevents policy collapse
- aligns with real-world conditions

---

## 7. Curriculum Strategy

### Phase 1:
- Only Easy tasks
- High reward density

### Phase 2:
- Mix Easy + Medium

### Phase 3:
- Medium + Hard

### Phase 4:
- Fully mixed distribution

---

### Key Principle

> The agent must **experience success early** to learn.

---

## 8. Task Evaluation Metrics

Each task exposes:

- success flag
- completion time
- stability score
- trajectory smoothness
- failure modes

---

## 9. Failure Modes to Track

- oscillation near target
- freezing behavior
- unstable corrections
- failure to complete sequence

---

## 10. Why this task design works

This structure satisfies:

- verifiable outcomes
- progressive difficulty
- multi-step interaction
- measurable improvement

These are the key properties expected in strong RL environments for OpenEnv-style systems.