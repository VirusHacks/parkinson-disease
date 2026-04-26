# MotorAssistEnv

> **OpenEnv Hackathon India 2026** · Theme #3.1 — World Modeling / Professional Tasks

[![HF Space](https://img.shields.io/badge/HuggingFace-Space-blue)](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor) [![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/meta-pytorch/OpenEnv) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[The Problem](#the-problem) · [What This Is](#what-this-is) · [How It Works](#how-it-works) · [Tasks](#tasks) · [Training](#training) · [Results](#results) · [Quick Start](#quick-start) · [Docs](#docs)

---

## The Problem

Parkinson's disease disrupts the brain signals responsible for movement, leading to tremors, stiffness, and slowed motor control. Deep Brain Stimulation can reduce these symptoms — but tuning stimulation parameters today is still a difficult, manual trial-and-error process.

A neurologist sets amplitude, frequency, and pulse width once, and the device runs at those fixed settings for months. Meanwhile, the patient's brain changes every single day. Medication wears off. Stress spikes oscillations. Exercise shifts the baseline. The device doesn't know any of this. It just keeps firing.

**The gap between visits is where patients suffer.**

Adaptive DBS — where the device reads biomarkers and adjusts stimulation in real time — is where the field is heading. Clinical trials have already shown it reduces side effects and improves motor outcomes. What's missing is the *policy*: a controller intelligent enough to do the adapting continuously, not just on the day of the clinic visit.

Nobody had tried training a language model to be that controller. Until now.

---

## What This Is

**MotorAssistEnv** is a scientifically grounded reinforcement learning environment for training AI agents to optimize Parkinson's treatment automatically.

At its core, it simulates the brain's Parkinsonian motor circuitry — including the Subthalamic Nucleus, where abnormal neural activity is strongly linked to motor symptoms — and models how Deep Brain Stimulation interacts with this circuit in real time.

Through the OpenEnv interface, DBS parameters — amplitude, pulse width, and frequency — become the agent's action space. Each action updates the patient's neural and motor state, creating a true closed-loop control environment. The environment is calibrated against **Fleming et al. (2023)**, a peer-reviewed Hodgkin-Huxley simulation of 400 neurons published in the *Journal of Neural Engineering*. This is not a proxy. The numbers are real.

To make this physically grounded, the environment integrates **Meta's MyoSuite biomechanical simulation framework**, allowing neural activity to directly drive anatomically accurate musculoskeletal movement. The agent is not learning on abstract numbers alone — it is learning through realistic body dynamics. As Parkinsonian activity increases, the model develops visible tremor and slower movement. As stimulation improves, the body regains smoother and stronger motor control.

The agent receives rich observations — tremor severity, beta activity, movement force, tracking quality, side-effect estimates — then learns policies that maximize symptom relief while minimizing adverse effects. By combining neuroscience simulation, real biomechanical feedback, and reinforcement learning, MotorAssistEnv creates a high-fidelity digital patient environment that brings AI training significantly closer to real-world therapeutic deployment.

**This is the first RL benchmark for adaptive closed-loop DBS control with language models. We believe it is a major step toward adaptive, personalized, and autonomous neurostimulation systems for the future of Parkinson's care.**

---

## How It Works

One step = one 20 ms DBS cycle, the same cadence as real hardware. The agent and the patient's brain run in a continuous closed loop — sense, decide, stimulate, repeat.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    CLOSED-LOOP CONTROL CYCLE                    │
  │                       (every 20 ms)                             │
  └─────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────┐
  │         Patient Brain State          │  ← noisy sensor reading
  │                                      │    (what the DBS device sees)
  │  beta_arv    ← suppress this         │
  │  tremor_arv  ← suppress this         │
  │  force       ← protect this          │
  │  side_effect ← don't exceed budget   │
  │  beta_trend  ← getting better/worse? │
  │  + 25 more fields                    │
  └─────────────────┬────────────────────┘
                    │  observe
                    ▼
  ┌──────────────────────────────────────┐
  │     Agent  (Qwen3-4B + LoRA)         │
  │     trained with GRPO                │
  └─────────────────┬────────────────────┘
                    │  decide
                    ▼
  ┌──────────────────────────────────────┐
  │              DBS Action              │
  │                                      │
  │  amplitude   (mA)                    │
  │  pulse_width (µs)                    │
  │  frequency   (Hz)                    │
  │  motor_command                       │
  └─────────────────┬────────────────────┘
                    │  stimulate
                    ▼
  ┌──────────────────────────────────────┐
  │        Patient Brain Responds        │  ← true latent state
  │                                      │    (what the grader reads)
  │  new beta, tremor, force, side-fx    │
  │  → dense reward signal               │
  └─────────────────┬────────────────────┘
                    │
          ┌─────────┴──────────┐
          │ next step (loop)   │  ──────────────────────────► back to top
          │                    │
          │ at episode end ↓   │
          └─────────┬──────────┘
                    ▼
  ┌──────────────────────────────────────┐
  │    9-component clinical grader       │
  │    deterministic · cannot be gamed   │
  │    final score  ∈  [0.0 , 1.0]       │
  └──────────────────────────────────────┘
```

The agent never sees the true brain state — it sees a noisy sensor reading, just like a real DBS device. The grader reads the true latent state. This is exactly how clinical outcomes work: a noisy LFP signal to control with, a neurologist's assessment to be judged by. The two are deliberately separate.

---

## Why This Is a Perfect RL Environment

Most RL environments are designed to be solvable. This one is designed to be *learnable* — which is harder, and more useful.

DBS control for Parkinson's has every structural property that makes RL both necessary and tractable:

**Sequential decisions that compound.** Every stimulation setting changes the brain state the agent will see at the next step. A bad choice at step 10 echoes forward for 140 more steps. A fixed dose that works early in an episode may actively harm the patient by step 80. The agent has to think across time, not just react to the current snapshot.

**No fixed optimal policy.** The right amplitude and pulse width depend on the patient's profile, the current disease phase, medication level, and the history of what the agent has already done. A lookup table won't work. A PID controller tuned for one patient fails on another. Only a policy that can read context and reason about combinations of signals can adapt.

**Partial observability with real sensor noise.** The agent sees what a DBS device actually sees: noisy LFP readings and surface EMG. The true neural firing state is hidden. The agent must infer what's happening inside the brain from incomplete, noisy signals — and still act confidently enough to help.

**Dense feedback every 20 ms.** Unlike most medical environments where outcomes arrive at discharge, DBS produces meaningful physiological signals at every timestep. Every action gets a reward. That makes credit assignment tractable and reward shaping clinically grounded, not arbitrary.

**Multi-objective trade-off with no easy shortcut.** Beta suppression, tremor reduction, motor force preservation, side-effect budget, stimulation efficiency, movement smoothness — all simultaneously. Max out any one of them and the others collapse. The environment was explicitly designed so that every known single-objective gaming strategy fails. The only path to a high score is actually treating the patient.

**Non-stationary dynamics.** Tachyphylaxis degrades DBS effectiveness over time. Medication wears off mid-episode. Motor surges fire stochastically. The environment doesn't sit still. A policy that works at step 1 has to keep working at step 150 — on a brain that has been actively fighting back.

**Long-horizon budget management.** The side-effect load accumulates across the episode. An agent that over-stimulates early to lock in a good beta score will exhaust its safety budget and spend the second half of the episode in a clinically unacceptable regime. The agent has to plan, ration, and recover — not just react.

> These are exactly the challenges that break classical controllers. PID tracks a setpoint. A language model trained with RL can read the combination of signals, reason about where the episode is heading, and adjust strategy accordingly. That is the gap MotorAssistEnv was built to bridge.

---

## Tasks

Ten tasks across a strict difficulty ladder. The ordering isn't aspirational — it's empirically proven. A constant 1.0 mA baseline was run across 5 seeds per task:

| Task | Steps | What the Agent Faces | Pass Threshold |
|---|:---:|---|:---:|
| `easy` | 36 | Calm patient, steady biomarkers. Prove DBS works. | 0.55 |
| `medium` | 60 | Mid-episode deterioration. Rescue without causing dyskinesia. | 0.52 |
| `hard` | 150 | Four simultaneous crises: tachyphylaxis + off-med emergency + dyskinesia spikes + motor surges. Refractory patient. | 0.42 |

> **Constant baseline scores:** easy 0.72–0.80 ✅ · medium 0.47–0.52 ❌ · hard 0.23–0.36 ❌
>
> Passing means the agent reasoned. Constant stimulation always fails medium and hard.

**Plus 7 expert tasks** that test transfer, not just performance: fragile patients with narrow therapeutic windows, refractory patients who stop responding, surgical follow-up windows where exceeding 0.6 mA is catastrophic, nocturnal transitions that tighten every biomarker target mid-episode.

An agent that passes all three buckets hasn't memorised a dose — it's understood what DBS is for.

---

## Reward Design

Two layers. A dense per-step signal that gives gradient feedback every 20 ms, and a deterministic 9-component grader at episode end that cannot be gamed.

Every reward term maps to a clinical measurement from a published study:

| Term | What it measures | Clinical source |
|---|---|---|
| Beta ARV suppression | Primary DBS objective | Little et al. 2013, 2016 |
| Tremor ARV suppression | Secondary biomarker | Tinkhauser et al. 2017 |
| Force preserved | Voluntary motor function | Limousin et al. 1995 |
| Safety (side-effect load) | Dyskinesia risk | Swann et al. 2018 |
| Efficiency | Battery longevity | Priori et al. 2013 |
| Smoothness | Abrupt-change dyskinesia | Velisar et al. 2019 |
| Terminal stability | Last 5 steps only — blocks front-loading | — |

**The reward cannot be gamed.** We tested 15 adversarial strategies — zero stimulation, constant maximum amplitude, front-loading early steps, sensor-fooling, memorising trajectories — and every single one is explicitly blocked. The only way to score well is to actually treat the patient.

> Full reward formula, per-task weight tables, and all 15 exploit-blocks: [REWARD_DESIGN.md](./REWARD_DESIGN.md)

---

## Training

### Approach: SFT First, Then GRPO

Starting GRPO from a blank model on a medical control task is a recipe for failure. Without SFT, the model has no idea what a valid DBS action looks like — it produces outputs like "2.2 mA constant amplitude" that trip the safety budget immediately. We saw this exact failure in our zero-shot baseline runs.

The training pipeline:

1. **SFT** — Roll out the reference adaptive policy. Each step becomes a training example: *this observation → this action*. The model learns what clinical validity looks like before RL starts.
2. **GRPO Run 1** — 67 steps, group size 6, curriculum across easy/medium/hard. The model learns which valid action is best. *(79 minutes, Kaggle T4)*
3. **GRPO Run 2** — 270 steps, full epoch, cosine LR schedule to completion. Policy continues improving. *(37 minutes)*

**Total: 337 GRPO steps · 116 minutes · free Kaggle T4 GPU**

### What the Training Looked Like

![GRPO Training Dashboard](./plots/09_combined_dashboard.png)

*Four signals, all healthy: policy loss bounded and converging, reward stable above 0.86, KL divergence rising (the policy is genuinely moving away from base), reward std nonzero (GRPO always has signal to work with).*

![Policy Loss](./plots/01_policy_loss.png)

*Loss spikes early as the model explores, then settles. This is normal GRPO behaviour: high-variance exploration followed by exploitation of discovered strategies.*

![Mean Reward](./plots/02_mean_reward.png)

*Mean reward averaged 0.867 over Run 1, peaking at 0.985 at step 9. The high starting point is the SFT benefit — the model begins competent and GRPO refines from there.*

![KL Divergence](./plots/03_kl_divergence.png)

*KL divergence climbed from 0.41 to 0.77 across both runs — a total drift of +0.37 from base. This is the fingerprint of genuine learning. A flat line would mean nothing changed.*

![Phase Comparison](./plots/07_phase_comparison.png)

*Reward stays stable across training phases (early 0.865 → late 0.859) while KL keeps climbing (0.43 → 0.51). The policy is still moving at step 67. Not stuck. Not collapsed.*

**Format compliance across all 337 steps: 100%.** Not a single generation failed to produce parseable JSON within the token budget. SFT eliminated cold-start failures before GRPO began.

---

## Results

### Zero-Shot Baselines: What Existing Models Can Do

Before training, three production LLMs were benchmarked zero-shot — just a system prompt, no fine-tuning:

![Baseline Scores by Task](./plots/benchmark/10_score_by_model_task.png)

Every model passes easy. Medium and hard reveal the gap. The 7B model scores **0.255 on medium** — *lower than a constant 1.0 mA policy* — and **0.019 on hard**, essentially noise. The problem is clear from the amplitude traces:

![Amplitude Traces](./plots/benchmark/14_amplitude_traces.png)

*The 7B model slams amplitude to 1.5–2.2 mA and holds it. No adaptation. The safety budget collapses by step 30. The 72B model on easy adjusts dynamically between 0.8 and 1.1 mA — that's the adaptive behaviour we're training for.*

![Pass Rate Heatmap](./plots/benchmark/12_pass_rate_heatmap.png)

*Easy: all models pass. Medium and hard: only the 72B passes. The environment is doing its job — easy tasks are accessible, hard tasks require genuine adaptive control.*

### After Training: Our 4B Model vs. All Baselines

![Trained Model vs Baselines](./plots/trained_vs_baseline.png)

**The trained Qwen3-4B passes all three tasks: easy 0.830, medium 0.610, hard 0.480.**

- Against zero-shot 7B: our 4B model (43% fewer parameters) scores 0.610 vs 0.255 on medium, 0.480 vs 0.019 on hard. **Not close.**
- Against zero-shot 72B: our 4B model matches it on medium (0.610 vs 0.615) using **18× fewer parameters**.

That's what SFT + GRPO training does. A smaller model, trained for under 2 hours on free compute, learns to do what a much larger model barely manages — and what a smaller model cannot do at all.

### Summary Numbers

| | Value |
|---|---|
| Total training steps | **337** (67 + 270) |
| GPU time | **116 minutes** on free Kaggle T4 |
| KL drift from base | **+0.37** (0.41 → 0.77) |
| Peak training reward | **0.985** (step 9) |
| Format compliance | **100%** across 337 steps |
| Trained 4B — easy | **0.830** ✅ |
| Trained 4B — medium | **0.610** ✅ |
| Trained 4B — hard | **0.480** ✅ |
| 7B zero-shot — medium | 0.255 ❌ |
| 7B zero-shot — hard | 0.019 ❌ |

> Full training narrative with all plots: [Results.md](./Results.md)

---

## Quick Start

**Run the environment**

```bash
uv run --project parkinsons_Motor server
# → http://localhost:8000  |  /docs  |  /viewer (3D)
```

**Use it from Python**

```python
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment
from parkinsons_Motor.core.models import ParkinsonsMotorAction

env = ParkinsonsMotorEnvironment()
obs = env.reset(task_id="hard", seed=42)

for _ in range(obs.metadata["episode_steps"]):
    action = ParkinsonsMotorAction(
        dbs_amplitude=1.2,
        dbs_pulse_width=130,
        dbs_frequency=130.0,
        motor_command=obs.target_output,
    )
    obs = env.step(action)
    if obs.done:
        print(f"score={obs.grader_score:.3f}  pass={obs.episode_success}")
        break
```

**Train with GRPO on Colab/Kaggle**

Open [`colab_train_motorassist.ipynb`](./colab_train_motorassist.ipynb) — TRL GRPOTrainer + Unsloth 4-bit + LoRA on Qwen3-4B. Runs end-to-end in under 2 hours on a free T4.

**Live environment:** [huggingface.co/spaces/virustechhacks/parkinsons_Motor](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor)

---

## Why It Matters

More than 10 million people live with Parkinson's disease worldwide. More than 50,000 have a DBS implant. The hardware works. The settings are almost always suboptimal, and patients spend months in the gap between programming visits.

Adaptive DBS is proven in clinical trials to reduce side effects and improve outcomes. What's missing is the policy. Training that policy on real patients is dangerous and slow. MotorAssistEnv is the simulation-first benchmark that lets a language model practice the control problem at scale — grounded in peer-reviewed biophysics, with a 10-task curriculum, a 9-component clinical grader, and an empirically falsifiable difficulty ladder.

If an RL-trained LLM can pass this benchmark, it becomes a serious candidate for the closed-loop policy in next-generation DBS firmware.

**Nobody had built this benchmark before. We think it was the perfect place for RL to enter the picture.**

---

## Docs

| Document | What's inside |
|---|---|
| [Results.md](./Results.md) | Full training story — baseline tests, SFT, GRPO runs, plots, scores |
| [REWARD_DESIGN.md](./REWARD_DESIGN.md) | Every reward term with clinical citations, exploit-block audit |
| [TASKS.md](./TASKS.md) | All 10 tasks, parameters, difficulty proof |
| [STATE_ACTION_SPACE.md](./STATE_ACTION_SPACE.md) | All 30 observation fields, 4 action knobs, patient profiles |
| [RESEARCH_AND_REFERENCES.md](./RESEARCH_AND_REFERENCES.md) | 25-source annotated bibliography |
| [blog.md](./blog.md) | The human story — written for anyone, not just researchers |
| [colab_train_motorassist.ipynb](./colab_train_motorassist.ipynb) | Runnable training notebook |

---

## Scientific Grounding

The environment dynamics are calibrated from Fleming et al. (2023, *J Neural Eng* 20(5):056029) — a Hodgkin-Huxley simulation of cortex, STN, GPe, GPi, thalamus, spinal motoneurons, and Hill-type muscle model (~5M synaptic connections). Force values are in real millinewtons. Beta values are normalised against real pre-DBS LFP recordings. The 12×15 DBS entrainment surface the agent navigates was published in that paper.

Every reward term has a peer-reviewed citation. Every exploit block has been tested and documented.

---

*MIT License · [HF Space](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor) · OpenEnv Hackathon India 2026*
