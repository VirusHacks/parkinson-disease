# Project Context: MotorAssistEnv

## 1. What we are building

We are building an OpenEnv-compatible RL environment for assistive motor control under Parkinson-like impairment.

The environment is designed so an agent can learn to compensate for unstable movement dynamics and improve performance on structured tasks such as:
- stabilizing a hand or end effector,
- reaching a target,
- grasping and holding an object,
- and completing multi-step manipulation sequences.

This is not a full biological simulation. The focus is on the functional problem: impaired movement, noisy control, delayed response, and the need for stable corrective action.

## 2. Why this idea was chosen

We wanted a problem that is:
- real-world,
- meaningful to judges,
- clearly structured as an RL environment,
- and novel enough to stand out.

The hackathon guidelines emphasize real-world tasks, standard `reset()` / `step()` / `state()` interfaces, typed models, 3+ tasks with grader scores, meaningful reward shaping, and a reproducible baseline script. They also explicitly care about reward quality, environment design, documentation, and demonstrable improvement. fileciteturn0file0

This idea fits those constraints well because it naturally supports:
- clear task definitions,
- measurable progress,
- curriculum learning,
- stochastic disturbances,
- and strong before/after evaluation.

## 3. High-level framing

The best framing is:

> Learn a compensatory control policy for impaired motor dynamics.

That is better than framing it as:
- “simulate Parkinson’s brain,” or
- “build a muscle simulator,” or
- “make a robot hand drink coffee.”

The reason is simple: judges care about whether the environment is useful, trainable, and verifiable. The environment should feel like a real-world benchmark rather than a science fair demo.

## 4. Lessons learned from the discussion

### A. The environment should be small enough to learn
If the problem is too hard, the agent never gets useful reward. The hackathon guide explicitly notes that RL only works when the model has some chance of success and that early tasks should make success possible. fileciteturn1file0

### B. The reward should be layered
A single reward number is too easy to game. We want several independent components:
- target distance,
- stability,
- smoothness,
- completion bonus,
- anti-cheat penalties.

### C. The environment should include stochasticity
Real-world motor control is not perfectly deterministic. We should randomize:
- tremor amplitude,
- delay,
- action noise,
- task start state,
- and difficulty profile.

### D. Long-horizon tasks need decomposition
Hard tasks like “pick a cup and place it” should be built from smaller phases:
- reach,
- stabilize,
- grasp,
- hold,
- place.

### E. Baselines matter
We need a simple baseline policy before training the RL agent, so we can show measurable improvement.

## 5. What makes winning projects look strong

From the winning projects we reviewed, the common patterns were:
- they model a real system, not an abstract toy;
- they expose a clean interaction loop;
- they use verifiable objectives;
- they include multiple reward signals;
- they support curriculum or adaptive difficulty;
- they demonstrate improvement clearly;
- and they present the project with crisp documentation and a strong story.

That is the standard we should emulate.

## 6. Design decisions we already agreed on

### Environment
A structured motor-control environment with impairment dynamics, stochastic disturbances, and clear action/observation definitions.

### Tasks
At minimum:
1. stabilization,
2. reaching,
3. manipulation.

### Rewards
Dense, layered, and anti-hacking-aware.

### Metrics
Success rate, stability, smoothness, tremor reduction, completion time, and timeout rate.

### Training
A baseline controller first, then RL training, then comparison.

### Presentation
The project should read like a serious benchmark built by professional ML and systems engineers.

## 7. What this context document is for

This file is the shared reference for the project.

It should help with:
- writing the repository README,
- explaining the problem in the demo,
- defining reward logic,
- designing the environment schema,
- and keeping the team aligned on scope.

## 8. Final project narrative

MotorAssistEnv is an RL benchmark for impaired motor control that aims to teach agents how to stabilize, correct, and complete everyday movement tasks under Parkinson-like disturbance patterns.

Its value comes from the combination of:
- real-world relevance,
- clear RL structure,
- verifiable task design,
- and a path toward assistive control research.

The project is intentionally designed to be both scientifically credible and hackathon-friendly.
