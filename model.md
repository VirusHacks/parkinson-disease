---
base_model: unsloth/qwen3-4b-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:unsloth/qwen3-4b-unsloth-bnb-4bit
  - grpo
  - lora
  - sft
  - transformers
  - trl
  - unsloth
  - parkinson
  - dbs
  - deep-brain-stimulation
  - medical-ai
  - reinforcement-learning
  - openenv
  - rlvr
license: mit
---

# MotorAssist — Qwen3-4B LoRA for Adaptive Deep Brain Stimulation Control

**Qwen3-4B + LoRA · SFT → GRPO · MotorAssistEnv · OpenEnv Hackathon India 2026**

This is a LoRA adapter fine-tuned on **Qwen3-4B** (4-bit quantised via Unsloth) to act as a real-time adaptive Deep Brain Stimulation (DBS) controller for Parkinson's disease patients. The model was trained using a two-stage pipeline — Supervised Fine-Tuning (SFT) followed by GRPO reinforcement learning — inside **MotorAssistEnv**, the first RL benchmark for closed-loop DBS control with language models.

The trained model **passes all three core clinical tasks** (easy, medium, hard) and **outperforms zero-shot Qwen2.5-7B on every task** despite having 43% fewer parameters.

---

## Model Details

### What This Model Does

The model acts as a closed-loop DBS policy. At every 20 ms timestep (the same cadence as real DBS hardware), it receives 30 sensor readings from a simulated Parkinson's patient and outputs four stimulation parameters:

- `dbs_amplitude` (mA) — stimulation strength
- `dbs_pulse_width` (µs) — pulse duration
- `dbs_frequency` (Hz) — pulse rate
- `motor_command` — voluntary motor intent

Its goal: suppress pathological beta oscillations and tremor, preserve voluntary motor function, and stay within the side-effect budget — simultaneously, across a 36–150 step episode.

### Model Description

| Property | Value |
| :--- | :--- |
| **Base model** | `unsloth/qwen3-4b-unsloth-bnb-4bit` (4-bit QLoRA) |
| **Adapter type** | LoRA (PEFT) |
| **LoRA rank** | 16 |
| **Trainable parameters** | ~33M out of 4B (0.81%) |
| **Training stages** | Stage 1: SFT · Stage 2: GRPO (two runs) |
| **Total GRPO steps** | 337 (Run 1: 67 · Run 2: 270) |
| **Training compute** | ~116 minutes · free Kaggle T4 GPU |
| **Framework** | TRL GRPOTrainer + Unsloth + HuggingFace PEFT |
| **Developed by** | OpenEnv Hackathon India 2026 team |
| **License** | MIT |
| **Environment** | [MotorAssistEnv](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor) |

### Model Sources

| Resource | Link |
| :--- | :--- |
| **Environment (HF Space)** | [huggingface.co/spaces/virustechhacks/parkinsons_Motor](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor) |
| **Demo video (YouTube)** | [youtu.be/ocF6SzPHexE](https://youtu.be/ocF6SzPHexE) |
| **Training notebook (Colab)** | [Open in Colab](https://colab.research.google.com/drive/1zJTiyyTcD_BahARPGa_2xlzH9MGCb8ye?usp=sharing) |
| **Training logs (WandB)** | [wandb.ai/daksh-jain24-spit/parkinsons-motor-env](https://wandb.ai/daksh-jain24-spit/parkinsons-motor-env) |
| **GitHub source** | [github.com/VirusHacks/parkinson-disease](https://github.com/VirusHacks/parkinson-disease/) |
| **Blog post** | [blog.md](https://github.com/VirusHacks/parkinson-disease/blob/main/blog.md) |
| **Full results** | [Results.md](https://github.com/VirusHacks/parkinson-disease/blob/main/Results.md) |
| **Reward design** | [REWARD_DESIGN.md](https://github.com/VirusHacks/parkinson-disease/blob/main/REWARD_DESIGN.md) |

---

## Uses

### Direct Use

Load the adapter on top of `unsloth/qwen3-4b-unsloth-bnb-4bit` and run it against the MotorAssistEnv environment. At each step, format the observation JSON as a system + user prompt and decode the model's output as a `ParkinsonsMotorAction` JSON object.

The model expects the OpenEnv observation schema (30 fields: beta_arv, tremor_arv, force_preserved, side_effect_load, etc.) and produces a JSON action with `dbs_amplitude`, `dbs_pulse_width`, `dbs_frequency`, and `motor_command`.

### Downstream Use

- **RL research:** Use as a strong initialisation point for further GRPO training on MotorAssistEnv with more compute.
- **DBS policy research:** Evaluate against new patient profiles or clinical scenarios by extending the task suite.
- **Adaptive neurostimulation:** With patient-specific calibration, hardware integration, and clinical validation, the policy structure is a prototype for next-generation adaptive DBS firmware.

### Out-of-Scope Use

- **Clinical deployment without validation.** This model was trained on a simulation. Real-world DBS programming requires patient-specific calibration, hardware validation, regulatory approval, and clinical trials.
- **General medical advice.** This is a research model for a specific sequential control task. It is not a general medical assistant.
- **Non-DBS tasks.** The model was fine-tuned specifically for the MotorAssistEnv observation/action schema. It will not generalise to other control tasks without retraining.

---

## How to Get Started

```python
from unsloth import FastLanguageModel

# Load the base model + LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    "virustechhacks/dbs-grpo-qwen3-4b",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)
FastLanguageModel.for_inference(model)

# Format an observation from MotorAssistEnv and generate a DBS action
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are an adaptive DBS controller. Output a JSON action."},
        {"role": "user", "content": "<observation JSON from env.step() or env.reset()>"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
action_json = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# Parse action_json as ParkinsonsMotorAction
```

To run the full benchmark against the live environment:

```bash
# Run environment server
uv run --project parkinsons_Motor server

# Run benchmark
python scripts/run_model_benchmark.py --model virustechhacks/dbs-grpo-qwen3-4b
```

---

## Training Details

### The Problem

Standard DBS devices are open-loop: a neurologist sets amplitude, pulse width, and frequency once, and the device fires at those fixed settings for months. Meanwhile, the patient's brain changes every day — medication wears off, stress spikes oscillations, disease progresses. Adaptive DBS (closed-loop, real-time adjustment) is proven to improve outcomes, but the policy that drives it does not exist. This model is a trained candidate for that policy.

### Training Data — Stage 1: SFT

**Source:** Reference heuristic adaptive controller (rule-based) rolled out against the live MotorAssistEnv environment across multiple episodes. Every 20 ms timestep produced one training example:

- **Input:** Full 30-field patient observation JSON
- **Output:** Clinically valid DBS action JSON (`dbs_amplitude`, `dbs_pulse_width`, `dbs_frequency`, `motor_command`)

The SFT data teaches the model what a clinically valid DBS response looks like before GRPO begins, eliminating cold-start format failures and allowing GRPO to focus entirely on policy quality rather than schema compliance.

### Training Procedure — Stage 2: GRPO

**Algorithm:** Group Relative Policy Optimisation (GRPO) via TRL `GRPOTrainer`

**Two training runs:**

| | Run 1 | Run 2 |
| :--- | :--- | :--- |
| Steps | 67 | 270 |
| Group size | 6 | 6 |
| LR schedule | constant | cosine decay |
| GPU time | ~79 min | ~37 min |
| Task curriculum | easy / medium / hard | easy / medium / hard |

**GRPO reward:** Per-step environment reward (from MotorAssistEnv's dense reward function) normalised by theoretical maximum (÷ 1.2) to prevent saturation at 1.0, plus format bonus for valid JSON output.

#### Training Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| Base model | `unsloth/qwen3-4b-unsloth-bnb-4bit` |
| Quantisation | 4-bit (NF4) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training precision | bf16 |
| Batch size (effective) | 6 per device × 4 grad accum = 24 |
| Group size (GRPO) | 6 |
| Max completion length | 128 tokens |
| Optimizer | AdamW (Unsloth fused) |
| Hardware | Kaggle T4 (free tier) |
| Total wall time | ~116 minutes |

#### Training Results

| Metric | Value |
| :--- | :--- |
| Total GRPO steps | **337** |
| Mean reward (Run 1) | **0.867** |
| Peak reward | **0.985** (step 9) |
| KL divergence (start → end) | **0.41 → 0.77** (+0.37 total drift) |
| Format compliance | **100%** across all 337 steps |
| Policy loss (final) | **0.008–0.011** range |

---

## Evaluation

### Environment

**MotorAssistEnv** — a peer-reviewed biophysical RL environment for adaptive DBS control, calibrated against Fleming et al. (2023, *J Neural Eng* 20(5):056029). Grader is deterministic math; no LLM-as-judge.

### Tasks

| Task | Steps | Clinical scenario | Pass threshold |
| :--- | :---: | :--- | :---: |
| `easy` | 36 | Calm patient, steady biomarkers | 0.55 |
| `medium` | 60 | Mid-episode symptom flare, rescue without dyskinesia | 0.52 |
| `hard` | 150 | Four simultaneous crises: tachyphylaxis + off-med + dyskinesia + motor surge | 0.42 |

### Results

| Model | Easy | Medium | Hard |
| :--- | :---: | :---: | :---: |
| **MotorAssist Qwen3-4B (this model)** | **0.830 ✅** | **0.610 ✅** | **0.480 ✅** |
| Qwen2.5-72B (zero-shot) | 0.773 ✅ | 0.615 ✅ | 0.605 ✅ |
| Mistral-7B (zero-shot) | 0.655 ✅ | 0.489 ❌ | 0.348 ❌ |
| Qwen2.5-7B (zero-shot) | 0.718 ✅ | 0.255 ❌ | **0.019 ❌** |
| Constant 1.0 mA baseline | 0.72–0.80 ✅ | 0.47–0.52 ❌ | 0.23–0.36 ❌ |

**Key findings:**
- Our trained 4B model passes all three tasks. Zero-shot 7B scores **0.019 on hard** — near-total collapse. Our trained model scores **0.480** on the same task.
- On medium, trained 4B (0.610) matches zero-shot 72B (0.615) using **18× fewer parameters**.
- The gap from zero-shot 7B to trained 4B on hard (0.019 → 0.480) is the entire argument for why RL training exists.

---

## Technical Specifications

### Model Architecture

- **Base:** Qwen3-4B transformer decoder, 4-bit NF4 quantised via Unsloth's bitsandbytes integration
- **Adapter:** LoRA (rank 16, alpha 16) injected at all attention + FFN projection layers
- **Trainable parameters:** ~33M / 4B (0.81%)
- **Inference format:** Chat template with system prompt (DBS controller role) + user message (observation JSON) → assistant message (action JSON)

### Compute Infrastructure

| Property | Value |
| :--- | :--- |
| **Hardware** | Kaggle T4 GPU (free tier, 16GB VRAM) |
| **Training time** | ~116 minutes total (79 + 37) |
| **Framework** | PyTorch · Transformers · TRL · Unsloth · PEFT |
| **Estimated CO₂** | Minimal — free T4, ~2h, datacenter renewable fraction unknown |

---

## Citation

If you use MotorAssistEnv or this model in your research, please cite:

```bibtex
@misc{motorassistenv2026,
  title        = {MotorAssistEnv: The First RL Benchmark for Adaptive Closed-Loop DBS Control with Language Models},
  author       = {OpenEnv Hackathon India 2026 Team},
  year         = {2026},
  howpublished = {OpenEnv Hackathon India 2026},
  url          = {https://github.com/VirusHacks/parkinson-disease/}
}

@article{fleming2023multivariable,
  title   = {Multivariable closed-loop control of deep brain stimulation for Parkinson's disease},
  author  = {Fleming, John E and Senneff, Stephanie and Lowery, Madeleine M},
  journal = {Journal of Neural Engineering},
  volume  = {20},
  number  = {5},
  pages   = {056029},
  year    = {2023},
  publisher = {IOP Publishing}
}
```

---

## Model Card Authors

OpenEnv Hackathon India 2026 · [MotorAssistEnv](https://huggingface.co/spaces/virustechhacks/parkinsons_Motor)

### Framework Versions

- PEFT 0.18.1
- TRL (latest)
- Unsloth (latest)
- Transformers (latest)
