---
title: Parkinsons Motor Environment Server
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# MotorAssistEnv: Closed-Loop DBS Agent for Parkinson's Disease

An OpenEnv-compliant reinforcement learning environment backed by the peer-reviewed Fleming et al. (2023) biophysical simulation. MotorAssistEnv trains autonomous agents to act as continuous Brain-Computer Interface (BCI) programmers, optimizing Deep Brain Stimulation (DBS) parameters to preserve patient motor function while managing side effects.

## 🚀 Quick Start (Benchmarking)

The simplest way to interact with the environment is running our local LLM baseline inference script:

1. **Configure Environment Variables**:
   In the root of the repository, create or edit your `.env` file with your API credentials.
   ```env
   # .env
   API_KEY="sk-..."
   API_BASE_URL=https://api.openai.com/v1
   MODEL_NAME=gpt-4o-mini
   ```

   The benchmark runner also supports `OPENAI_API_KEY`, `HF_TOKEN`, `LLM_PROVIDER`,
   `OPENAI_MODEL`, and `HF_MODEL_NAME`. If an OpenAI key is present, it defaults to
   the OpenAI API unless you explicitly force another provider.

2. **Run the Benchmark**:
   Start the LLM inference loop and local API which will evaluate your model against our 3 clinically-grounded tasks (`beta_suppression`, `tremor_correction`, `full_episode`).
   ```bash
   uv run --project parkinsons_Motor python run_local_inference.py
   ```

## Using the OpenEnv Client Directly

```python
from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv

# Connect directly to the local instance or Docker container
with ParkinsonsMotorEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id="tremor_correction")
    
    # Observe brain state
    print(f"Tremor ARV: {result.observation.tremor_arv}")
    
    # Emit DBS parameters
    action = ParkinsonsMotorAction(
        motor_command=0.4, 
        dbs_amplitude=1.5, 
        dbs_pulse_width=0.13
    )
    result = env.step(action)
    print(f"Force Preserved: {result.observation.force_preserved}, Reward: {result.reward}")
```

## Problem Space & Biological Accuracy
This environment is not a toy. It leverages a rigorous 100-step trajectory processed from over 5 million modeled neural synapses. The environment heavily penalizes arbitrary maximum-voltage policies, requiring models to discover delicate, non-linear `amplitude` vs `pulse_width` optimization curves depending on the severity of the patient's dynamically-escalating tremor.

*For full scientific context, please see `PROBLEM.md` in the repository root.*

## Deploying to Hugging Face Spaces

You can deploy this OpenEnv environment using the `openenv push` command:

```bash
openenv push .\parkinsons_Motor --repo-id your-namespace/parkinsons_Motor
```
This enables the interactive visualization UI (`/web`) via our Gradio MyoSuite arm integration, API Documentation (`/docs`), and Persistent WebSocket evaluation.
