# We Built the First RL Benchmark for Brain Implant Control 

*OpenEnv Hackathon India 2026 · MotorAssistEnv · Qwen3-4B + LoRA + GRPO*

---

There are 50,000 people alive today with a computer in their brain.

Not metaphorically. A physical device - titanium casing, the size of a matchbox, sitting in their chest - sends electrical pulses through a wire that runs up through the neck, through the skull, and terminates in the subthalamic nucleus, deep inside the basal ganglia. It fires 130 times per second. It has been doing this, without interruption, for years.

These people have Parkinson's disease. And for them, this device - a Deep Brain Stimulator - is nothing short of a miracle. The tremor that made holding a glass impossible stops. The rigidity that made buttoning a shirt take ten minutes eases. They get their bodies back.

But here is what nobody in the room tells them after the surgery:

**The device you just had implanted in your brain is dumb.**

It fires at whatever settings the neurologist chose on the day you left the clinic. Three months ago. Six months ago. Your brain has changed since then - medication cycles, stress, aging, disease progression - and the device does not know. It cannot know. It just keeps firing.

The gap between programming visits is where patients lose the gains from their surgery. And today, in 2026, with everything we know about AI and reinforcement learning, that gap has no solution.

**Until now.**

---

## What We Built, and Why It Has Never Existed Before

MotorAssistEnv is the **first reinforcement learning benchmark for adaptive closed-loop Deep Brain Stimulation control with language models**.

That sentence is not marketing. Search the literature. There is no RL environment for LLM-based DBS programming. There is no benchmark. There is no training pipeline. The closest prior work - `cviaai/RL-DBS` - uses classical RL, not language models, and does not expose a standard OpenEnv interface. Nobody has built what we built.

The reason it has never existed is that it is genuinely hard to build. You cannot fake this environment with a toy physics system. DBS control involves real biophysics - ion channel dynamics, synaptic connections, cortical entrainment, musculoskeletal force transmission. Get the simulation wrong and you train an agent that learns nothing transferable.

So we went to the peer-reviewed literature and found the one model that does the full pipeline correctly.

**Fleming et al. (2023)** - published in the *Journal of Neural Engineering* - simulates 400 neurons across cortex, subthalamic nucleus, globus pallidus, thalamus, spinal cord, and musculoskeletal output using Hodgkin-Huxley dynamics, finite-element DBS electrode modelling, and a Hill-type muscle model. Over 5 million individual synaptic connections. Force values in real millinewtons. Beta oscillations normalised against real pre-DBS LFP recordings from patients. This is not a proxy environment. Every number in our benchmark traces back to this paper.

We took that model. We built an OpenEnv interface on top of it. We added Meta's **MyoSuite biomechanical framework** so that neural activity drives anatomically accurate musculoskeletal movement in real time. We designed 10 clinical tasks, a 9-component grader where every term maps to a published clinical measurement, and 15 adversarial exploit-blocks to ensure the reward cannot be gamed.

Then we trained a language model on it.

---

## First, We Needed to Know How Badly Today's LLMs Fail

Before training a single parameter, we benchmarked three production-ready models - Qwen2.5-7B, Qwen2.5-72B, and Mistral-7B - zero-shot against our environment. No fine-tuning. Just a system prompt, an observation, and a request for a clinical action.

The results were not subtle.

![Baseline Model Scores by Task](plots/benchmark/10_score_by_model_task.png)

Every model passes the easy task. That's the smoke test - it proves the model can parse the observation and produce a valid action. Easy is designed to be accessible.

Medium and hard are not.

The 7B model scores **0.255 on medium**. A constant 1.0 mA policy - no AI, no reasoning, just fixed stimulation - scores **0.47**. The 7B model is not failing. It is actively making a simulated patient worse than doing nothing.

Here is exactly what it was doing:

![DBS Amplitude Traces by Model and Task](plots/benchmark/14_amplitude_traces.png)

Amplitude slammed to 1.5–2.2 mA and held. No adaptation. No reading the beta trend. No response to the side-effect load warnings accumulating in the observation with every step. The safety budget collapsed at step 30. The model spent the remaining 120 steps stimulating a patient whose dyskinesia budget was already exhausted.

Hard task score: **0.019.**

That number is not a poor performance. It is evidence that the problem we built is a real problem. A model with 7 billion parameters - trained on the entire internet - has no idea how to run a brain implant. Zero-shot prompting cannot give it that knowledge. Only training can.

The pass/fail breakdown makes this undeniable:

![Pass Rate Heatmap](plots/benchmark/12_pass_rate_heatmap.png)

Easy: all models pass. Medium and hard: only the 72B passes - and it passes because raw scale carries general reasoning part of the way there. A 72B model is not the answer. Training is.

**This is exactly the capability gap that RL was invented to close.**

---

## The Training Pipeline: Why We Did It in Two Stages

The single biggest mistake in RL for domain-specific tasks is starting from scratch and hoping exploration finds good actions.

We had seen the failure mode firsthand in the 7B zero-shot run: a capable model that does not know what clinically valid DBS control looks like defaults to maximum amplitude and holds it. If you start GRPO from that baseline, your first hundred training steps are teaching format compliance and clinical validity - wasted gradient on things that should already be known.

**Stage 1 - Supervised Fine-Tuning.** We rolled out a reference heuristic controller against the live environment. Every 20 ms step became one training example: *this is what the patient's brain looks like right now - this is what a clinically reasonable response looks like.* Qwen3-4B trained on this data first. After SFT, the model knows the JSON schema, knows that 0.85 mA is a reasonable starting amplitude, knows that when side-effect load spikes you reduce stimulation, not increase it. It is inside the valid clinical action space before GRPO begins.

**Stage 2 - GRPO.** With clinical validity already learned, GRPO has exactly one job: which valid action is *best*? Group size 6 - six rollouts per prompt - so GRPO always has relative advantages to compute. Curriculum across easy, medium, and hard tasks. LoRA rank 16, 33 million trainable parameters out of 4 billion. Free Kaggle T4 GPU.

This is a deliberate, principled pipeline. SFT teaches what is valid. GRPO teaches what is optimal. Neither step alone would work.

---

## What the Training Showed

![GRPO Training Dashboard](plots/09_combined_dashboard.png)

**Run 1: 67 steps, 79 minutes.** Policy loss spiked early as the model explored beyond the SFT distribution, then settled and converged. Mean reward held above 0.86 from step 1 - the SFT benefit. The model began competent; GRPO refined from there.

But the single most important metric in the entire training run is not reward. It is this:

![KL Divergence from Base Model](plots/03_kl_divergence.png)

KL divergence - the distance between the trained policy and the base Qwen3-4B - climbed from **0.41 to 0.57** across Run 1. Steadily. Consistently. Every step moving in the right direction.

A flat KL line means training changed nothing. A rising KL line means the policy is genuinely moving away from its starting point, discovering and committing to new strategies. Ours rose throughout both runs.

**Run 2: 270 steps, 37 minutes.** Full epoch, cosine LR schedule decaying to near-zero. KL divergence reached **0.77**. Total policy drift from base Qwen3-4B across both runs: **+0.37**.

The rewards in Run 2 oscillated between 0.63 and 0.88 with higher variance - because this epoch contained more hard episodes where the reward is genuinely noisy and the clinical situation is genuinely difficult. The model was being challenged. That variance is not a failure signal. It is evidence that the curriculum was working.

**Format compliance across all 337 steps: 100%.** Every single generation produced parseable JSON within the token budget. SFT eliminated cold-start failures completely.

**Total training: 337 GRPO steps. 116 minutes. A free Kaggle T4.**

---

## The Result: A 4B Model That Beats Zero-Shot 7B on Every Task

![Trained Qwen3-4B vs Zero-Shot Baselines](plots/trained_vs_baseline.png)

| Model | Easy (thr 0.55) | Medium (thr 0.52) | Hard (thr 0.42) |
|---|:---:|:---:|:---:|
| **Trained Qwen3-4B (ours)** | **0.830 ✅** | **0.610 ✅** | **0.480 ✅** |
| Zero-shot Qwen2.5-72B | 0.773 ✅ | 0.615 ✅ | 0.605 ✅ |
| Zero-shot Mistral-7B | 0.655 ✅ | 0.489 ❌ | 0.348 ❌ |
| Zero-shot Qwen2.5-7B | 0.718 ✅ | 0.255 ❌ | **0.019 ❌** |

Our trained Qwen3-4B **passes all three tasks**. Easy 0.830. Medium 0.610. Hard 0.480 - a task the 7B model scores 0.019 on with zero training.

That gap - 0.019 to 0.480 on the same hard task - is not an incremental improvement. It is the entire argument for why RL training exists. A model 43% smaller than the failing baseline, trained for under 2 hours on free compute, learns to do something a much larger model cannot do at all without training.

On medium, our 4B matches the zero-shot 72B (0.610 vs 0.615). We erased an 18× parameter advantage with a principled two-stage training pipeline and a reward function that actually defines what good DBS control means.

**This is what training is for.**

---

## Why This Environment Is the Right Environment for This Problem

DBS control is not just a medically important problem. It is structurally one of the most natural RL problems that exists in medicine:

**Actions compound across time.** A poor stimulation choice at step 10 shapes the brain state at step 50. You cannot optimise each step independently.

**No fixed optimal policy.** The right amplitude depends on the patient profile, the disease phase, medication timing, stochastic events that have fired, and what you did in the last five steps. No lookup table solves this.

**Dense reward every 20 ms.** Unlike most medical settings where outcomes appear at discharge, DBS produces physiological signals every timestep. Credit assignment is tractable.

**Multi-objective with no shortcut.** Beta suppression, tremor control, force preservation, side-effect budget, efficiency, smoothness - all simultaneously. Every single-objective exploitation strategy fails. The only path to a high score is actually treating the patient.

**Non-stationary dynamics.** Tachyphylaxis, medication wearing off, motor surges - the brain keeps changing underneath the agent's policy. A policy that works at step 1 must work at step 150 on a brain that is actively fighting back.

Classical controllers - PID, threshold-based - cannot handle this. They can track a setpoint. They cannot reason, they cannot generalise across patients, and they cannot adapt to events they were not programmed for. Language models can do all three. RL gives them the feedback to get better at it.

MotorAssistEnv is the benchmark that makes this trainable, measurable, and reproducible.

---

## This Is Bigger Than a Hackathon

There are 10 million people worldwide living with Parkinson's disease. Adaptive DBS - devices that continuously adjust stimulation to the patient's real-time brain state - has been proven in clinical trials to reduce side effects and improve motor outcomes compared to fixed stimulation. The hardware exists. The sensing exists. The firmware to run the intelligent policy does not.

MotorAssistEnv is the simulation-first benchmark that makes it possible to train that policy before it ever touches a patient. What an agent learns here - to suppress beta oscillations through the same biophysical mechanisms as the real model, to stay inside a side-effect budget grounded in real clinical measurements, to adapt across patient profiles and disease states - is not arbitrary. It is the beginning of a real answer.

We built this in a weekend on free compute. No proprietary data. No hospital access. Just peer-reviewed science, an open framework, and the conviction that this problem deserved to be solved.

**The policy that adaptive DBS needs - the one that keeps patients well between visits, the one that learns their brain and adjusts in real time - we believe can be trained. We built the environment to do it. This is that environment.**

---

## Run It Yourself

Everything is live and open.

| | |
| :--- | :--- |
| 🟢 **Live environment** | [huggingface.co/spaces/virustechhacks/parkinsons_Motor](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor) |
| 🤖 **Trained model** | [huggingface.co/virustechhacks/dbs-grpo-qwen3-4b](https://huggingface.co/virustechhacks/dbs-grpo-qwen3-4b) |
| 🎬 **Demo video** | [youtu.be/ocF6SzPHexE](https://youtu.be/ocF6SzPHexE) |
| 📓 **Training notebook** | [Open in Colab](https://colab.research.google.com/drive/1zJTiyyTcD_BahARPGa_2xlzH9MGCb8ye?usp=sharing) - runs end-to-end on a free T4 in under 2 hours |
| 📈 **Training logs** | [wandb.ai/daksh-jain24-spit/parkinsons-motor-env](https://wandb.ai/daksh-jain24-spit/parkinsons-motor-env) |
| 💻 **GitHub** | [github.com/VirusHacks/parkinson-disease](https://github.com/VirusHacks/parkinson-disease/) |
| 📊 **Full results** | [Results.md](./Results.md) |
| 🔬 **Reward design** | [REWARD_DESIGN.md](./REWARD_DESIGN.md) |

You can run the benchmark against your own model with a single script. You can extend the environment with new patient profiles, new crisis events, or new task objectives. The full documentation covers every reward term, every exploit-block, every state variable.

---

*Built for the OpenEnv Hackathon India 2026 - on the conviction that the most important frontier for AI in medicine is not diagnosis or image reading, but real-time adaptive control of the devices that are already inside patients.*

*[HF Space](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor) · [Model](https://huggingface.co/virustechhacks/dbs-grpo-qwen3-4b) · [Video](https://youtu.be/ocF6SzPHexE) · [Colab](https://colab.research.google.com/drive/1zJTiyyTcD_BahARPGa_2xlzH9MGCb8ye?usp=sharing) · [WandB](https://wandb.ai/daksh-jain24-spit/parkinsons-motor-env) · [GitHub](https://github.com/VirusHacks/parkinson-disease/)*
