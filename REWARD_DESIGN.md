# Reward Design for MotorAssistEnv

## 1. Purpose

Reward design is the most critical part of this environment.

In reinforcement learning, the reward is not just feedback. It is the *specification* of the task. If the reward is too sparse, the agent never learns. If it is too vague, it optimizes the wrong proxy. If it is too easy to exploit, training looks successful while the policy remains useless in the real task.

For MotorAssistEnv, the reward must do four things at once:

1. encourage the agent to reach and stabilize the target,
2. reward smooth and realistic motion,
3. support long-horizon tasks without collapsing into sparsity,
4. resist reward hacking and degenerate shortcuts.

This document defines the reward mathematically and explains why each term exists.

---

## 2. Design principles

The reward design follows these principles:

### 2.1 Verifiable outcomes first
The primary signal should come from objective environment state, not from subjective judgments.

### 2.2 Dense but not over-shaped
The agent must receive meaningful feedback at every step, but shaping should not dominate the task objective.

### 2.3 Multi-objective balance
The environment should reward:
- goal attainment,
- stability,
- smoothness,
- efficiency,
- and safety.

### 2.4 Anti-hacking by construction
The reward should make obvious loopholes unprofitable:
- freezing in place,
- oscillating to exploit a metric,
- taking unrealistically large actions,
- or maximizing a proxy while ignoring the actual goal.

### 2.5 Curriculum compatibility
The same reward family must work across:
- stabilization,
- reaching,
- and multi-step manipulation.

---

## 3. Notation

Let:

- \( s_t \) be the state at time \( t \),
- \( a_t \) be the agent action,
- \( x_t \in \mathbb{R}^d \) be the current end-effector position,
- \( g_t \in \mathbb{R}^d \) be the goal position,
- \( v_t \in \mathbb{R}^d \) be the velocity,
- \( u_t \in \mathbb{R}^d \) be the executed control after impairment/noise,
- \( \Delta u_t = u_t - u_{t-1} \),
- \( T \) be the episode length,
- \( \mathbb{1}[\cdot] \) be the indicator function.

Define:

- distance error:
  \[
  e_t = \|x_t - g_t\|_2
  \]

- speed:
  \[
  \|v_t\|_2
  \]

- action jerk proxy:
  \[
  j_t = \|\Delta u_t - \Delta u_{t-1}\|_2
  \]

- tremor proxy:
  \[
  \tau_t = \mathrm{Var}(x_{t-k:t})
  \]
  for a short sliding window of recent positions.

---

## 4. Core reward structure

The per-step reward is a weighted sum:

\[
r_t = w_{\text{goal}} r_{\text{goal},t}
    + w_{\text{stab}} r_{\text{stab},t}
    + w_{\text{smooth}} r_{\text{smooth},t}
    + w_{\text{eff}} r_{\text{eff},t}
    + w_{\text{progress}} r_{\text{progress},t}
    + r_{\text{success},t}
    - r_{\text{penalty},t}
\]

The final episode return is discounted:

\[
R = \sum_{t=0}^{T-1} \gamma^t r_t
\]

Recommended discount factor:

\[
\gamma \in [0.95, 0.99]
\]

For MotorAssistEnv, the practical default is:

\[
\gamma = 0.98
\]

Why:
- the environment is continuous-control and medium-horizon,
- future stability matters,
- and we do not want the agent to over-optimize immediate movement at the expense of long-term control.

---

## 5. Reward components

## 5.1 Goal reward

The most important term is proximity to the goal:

\[
r_{\text{goal},t} = - \frac{e_t}{e_{\max}}
\]

where \( e_{\max} \) is a normalization constant representing a large, but valid, task distance.

This term ensures that the agent is always pushed toward the target.

### Why this matters
Without a distance-based term, the agent may learn motion patterns that look stable but do not actually move toward the task goal.

---

## 5.2 Stability reward

To encourage the agent to hold position and reduce drift:

\[
r_{\text{stab},t} = - \frac{\tau_t}{\tau_{\max}}
\]

where \( \tau_t \) is a short-window variance of the end-effector trajectory.

### Why this matters
For Parkinson-like impairment, being close to the target is not enough. The policy must also remain steady. This term gives positive pressure against wobble and tremor amplification.

---

## 5.3 Smoothness reward

To discourage jerky corrections:

\[
r_{\text{smooth},t} = - \frac{\|a_t - a_{t-1}\|_2}{a_{\max}}
\]

or, if using executed control:

\[
r_{\text{smooth},t} = - \frac{\|u_t - u_{t-1}\|_2}{u_{\max}}
\]

### Why this matters
A policy that “overshoots and corrects” can technically reach the target while being unusable in practice. Smoothness reduces that failure mode.

---

## 5.4 Efficiency reward

A small efficiency term prevents the agent from using unnecessarily large actions:

\[
r_{\text{eff},t} = - \frac{\|a_t\|_2^2}{a_{\max}^2}
\]

### Why this matters
This keeps the policy from brute-forcing the environment with large control bursts. It also encourages energy-aware behavior, which is important for assistive systems.

---

## 5.5 Progress reward

To reduce sparsity in long tasks, we add a dense progress signal:

\[
r_{\text{progress},t} = \frac{e_{t-1} - e_t}{e_{\max}}
\]

This term is positive when the agent gets closer to the goal and negative when it moves away.

### Why this matters
Long-horizon tasks can be too sparse if the agent only gets a reward at the end. Progress shaping gives local learning signal without replacing the real objective.

---

## 5.6 Success bonus

When the task is completed:

\[
r_{\text{success},t} = \alpha_{\text{succ}} \cdot \mathbb{1}[e_t < \epsilon]
\]

with a separate condition for stability if required:

\[
\mathbb{1}[e_t < \epsilon \;\wedge\; \tau_t < \delta]
\]

Recommended bonus:

\[
\alpha_{\text{succ}} \in [1.0, 5.0]
\]

### Why this matters
Dense shaping helps learning, but the final task completion must remain the strongest signal. The success bonus anchors the policy to the actual objective.

---

## 5.7 Penalty term

We add penalties for clearly undesirable behavior:

\[
r_{\text{penalty},t}
=
\lambda_{\text{clip}} \cdot \mathbb{1}[\text{action clipped}]
+
\lambda_{\text{freeze}} \cdot \mathbb{1}[\text{frozen too long}]
+
\lambda_{\text{jump}} \cdot \mathbb{1}[\text{unrealistic jump}]
+
\lambda_{\text{timeout}} \cdot \mathbb{1}[t = T-1 \wedge \text{not success}]
\]

### Why this matters
This is the anti-hacking layer. It ensures the agent cannot maximize reward by hiding from the task, exploiting saturation, or producing impossible motion.

---

## 6. Recommended weight ranges

A good starting point is:

\[
w_{\text{goal}} = 1.00
\]
\[
w_{\text{stab}} = 0.30
\]
\[
w_{\text{smooth}} = 0.20
\]
\[
w_{\text{eff}} = 0.05
\]
\[
w_{\text{progress}} = 0.40
\]

These are not magic constants. They are sensible initial values for a task where goal achievement is primary, but motion quality still matters.

### Why this balance works

- **goal** gets the largest weight because task completion matters most,
- **progress** gets a strong weight because long-horizon learning needs dense feedback,
- **stability** and **smoothness** are important secondary objectives,
- **efficiency** should remain a light regularizer, not dominate the policy.

If efficiency is too strong, the agent may become overly conservative. If smoothness is too strong, the agent may refuse to move. If progress is too weak, learning can stall.

---

## 7. Task-specific reward variants

The same family of reward functions should be adapted slightly for each task.

## 7.1 Stabilization task

Primary objective:
keep the end-effector near a fixed point.

Recommended reward:

\[
r_t = 1.0 \cdot r_{\text{goal},t}
    + 0.6 \cdot r_{\text{stab},t}
    + 0.2 \cdot r_{\text{smooth},t}
    + 0.05 \cdot r_{\text{eff},t}
    + r_{\text{success},t}
    - r_{\text{penalty},t}
\]

Why:
stability matters more than forward progress here.

---

## 7.2 Reaching task

Primary objective:
move from start to target efficiently without oscillation.

Recommended reward:

\[
r_t = 1.0 \cdot r_{\text{goal},t}
    + 0.3 \cdot r_{\text{stab},t}
    + 0.3 \cdot r_{\text{smooth},t}
    + 0.05 \cdot r_{\text{eff},t}
    + 0.4 \cdot r_{\text{progress},t}
    + r_{\text{success},t}
    - r_{\text{penalty},t}
\]

Why:
the agent should be rewarded for moving toward the target, but still penalized for chaotic paths.

---

## 7.3 Manipulation task

Primary objective:
complete a sequence of subgoals such as reach → grasp → hold → place.

Recommended reward:

\[
r_t = 1.0 \cdot r_{\text{goal},t}
    + 0.4 \cdot r_{\text{stab},t}
    + 0.3 \cdot r_{\text{smooth},t}
    + 0.05 \cdot r_{\text{eff},t}
    + 0.5 \cdot r_{\text{progress},t}
    + r_{\text{success},t}
    - r_{\text{penalty},t}
\]

Why:
long-horizon manipulation needs stronger progress shaping and stronger stability pressure.

---

## 8. Normalization

All reward terms should be normalized to roughly comparable scales.

A practical design rule:

- each shaped term should usually live in \([-1, 0]\) or \([0, 1]\),
- the success bonus should be larger than any single shaping term,
- and penalties should be large enough to matter but not so large that they dominate learning.

### Why normalization matters
If one term is numerically much larger than the others, it will dominate training regardless of our intended weight design.

---

## 9. Episode termination and terminal rewards

An episode ends when one of the following happens:

1. success condition is satisfied,
2. time limit is reached,
3. a hard safety constraint is violated.

At termination:

- success gets a positive terminal reward,
- timeout gets a penalty,
- catastrophic failure gets a strong negative reward.

Example:

\[
r_{\text{terminal}} =
\begin{cases}
+\alpha_{\text{succ}} & \text{if success}\\
-\alpha_{\text{timeout}} & \text{if timeout}\\
-\alpha_{\text{fail}} & \text{if invalid state}
\end{cases}
\]

### Why this matters
This makes the episode boundary meaningful and prevents the agent from exploiting endless indecision or unsafe behavior.

---

## 10. Reward hacking risks

Reward hacking is one of the biggest risks in this environment.

## 10.1 Freeze-to-win
The agent may discover that staying still avoids penalties.

**Fix:** require progress or phase completion, not just low movement.

---

## 10.2 Oscillation gaming
The agent may find a periodic motion that exploits how a stability metric is computed.

**Fix:** use a windowed variance plus smoothness penalty and inspect trajectories visually.

---

## 10.3 Large-action exploitation
The agent may use large action bursts if the penalty is too weak.

**Fix:** clip actions and penalize clipping events.

---

## 10.4 Proxy reward exploitation
The agent may maximize a shaped metric without actually succeeding.

**Fix:** keep the success condition dominant and separate from shaping terms.

---

## 10.5 Terminal loopholes
The agent may learn to end episodes in a way that avoids penalties.

**Fix:** only allow early termination on real success, not on arbitrary agent choice.

---

## 11. Why multiple reward signals are necessary

A single reward term is rarely sufficient for a realistic assistive-control problem.

We need multiple signals because:
- goal distance alone ignores motion quality,
- smoothness alone ignores task completion,
- efficiency alone can cause inactivity,
- and success-only reward is too sparse.

This is why the reward should combine outcome-based reward with process-aware shaping. That matches the broader OpenEnv guidance to use multiple independent reward functions and to monitor for reward hacking during training. fileciteturn2file0 fileciteturn2file1

---

## 12. Suggested default configuration

For the first implementation, use this reward stack:

\[
r_t = 1.0 r_{\text{goal},t}
    + 0.4 r_{\text{progress},t}
    + 0.3 r_{\text{stab},t}
    + 0.2 r_{\text{smooth},t}
    + 0.05 r_{\text{eff},t}
    + r_{\text{success},t}
    - r_{\text{penalty},t}
\]

with:

\[
\gamma = 0.98
\]

and reward clipping to keep magnitudes bounded.

This is a strong starting point because it:
- gives the agent dense feedback,
- keeps the main objective dominant,
- and includes protection against the easiest forms of reward hacking.

---

## 13. What to monitor during training

Do not monitor only the total reward.

Also track:
- success rate,
- final distance to target,
- stability variance,
- action magnitude,
- action clipping frequency,
- timeout rate,
- and trajectory smoothness.

If reward rises but stability or success does not, the reward is being exploited or misweighted.

---

## 14. Tuning strategy

Start simple:
1. validate the environment manually,
2. train with the default reward,
3. inspect rollouts,
4. identify failure modes,
5. adjust weights gradually.

### Practical tuning order
- first fix success definition,
- then fix progress shaping,
- then fix stability and smoothness,
- then fine-tune efficiency,
- then add stricter penalties if needed.

### Why this order works
It prevents overfitting to a shaped metric before the actual task is learned.

---

## 15. Final design goal

The ideal reward function for MotorAssistEnv should make the agent behave like a competent assistive controller:

- it should move when movement is needed,
- stabilize when stability is needed,
- complete tasks efficiently,
- and avoid pathological shortcuts.

If the reward is well designed, the agent will learn a policy that is not only numerically high-scoring, but also genuinely useful.

That is the standard this environment should meet.
