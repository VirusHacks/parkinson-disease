import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("dbs_grpo_local.ipynb")
OUTPUT_PATH = Path(__file__).with_name("dbs_grpo_kaggle.ipynb")


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

cells[0]["source"] = lines(
    """# DBS GRPO Training for Kaggle

Train Qwen2.5-1.5B via GRPO to act as an adaptive DBS controller for a Parkinson's patient. This Kaggle copy preserves the original training and evaluation logic, but changes the runtime setup so it can run inside a Kaggle GPU notebook without relying on a local `.venv` or raw `asyncio.run(...)`.
"""
)

cells[1]["source"] = lines(
    """import os
import shutil
import sys
from pathlib import Path


def _is_repo_root(path: Path) -> bool:
    return (path / "parkinsons_Motor" / "pyproject.toml").exists()


def _find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if _is_repo_root(candidate):
            return candidate

    for base in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if not base.exists():
            continue
        for match in base.rglob("parkinsons_Motor/pyproject.toml"):
            return match.parent.parent

    raise FileNotFoundError(
        "Could not find the repository root. Upload the whole repo, not only the notebook."
    )


SOURCE_REPO_ROOT = _find_repo_root()
WORK_REPO_ROOT = Path("/kaggle/working") / SOURCE_REPO_ROOT.name

if str(SOURCE_REPO_ROOT).startswith("/kaggle/input/"):
    if not _is_repo_root(WORK_REPO_ROOT):
        print(f"Copying repo from read-only input to writable workdir: {SOURCE_REPO_ROOT} -> {WORK_REPO_ROOT}")
        shutil.copytree(
            SOURCE_REPO_ROOT,
            WORK_REPO_ROOT,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc", "outputs", ".ipynb_checkpoints"
            ),
        )
    REPO_ROOT = WORK_REPO_ROOT.resolve()
else:
    REPO_ROOT = SOURCE_REPO_ROOT.resolve()

PKG_ROOT = REPO_ROOT / "parkinsons_Motor"
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("repo root:", REPO_ROOT)
print("package root:", PKG_ROOT)
print("cwd:", Path.cwd())
print("python:", sys.executable)
"""
)

cells[2]["source"] = lines(
    """# Kaggle copy: the original local notebook uninstalled torch packages here.
# That is intentionally disabled because Kaggle already provides the runtime torch build.
print("Kaggle preflight: keeping the active torch installation intact.")
"""
)

cells[3]["source"] = lines(
    """# Kaggle copy: the original local notebook force-reinstalled torch into `.venv/bin/python`.
# The setup cell below installs only the missing training dependencies into the active kernel.
print("Kaggle preflight: skipping local `.venv` torch reinstall.")
"""
)

cells[4]["source"] = lines(
    """# Duplicate torch reinstall cell from the local notebook is intentionally disabled in the Kaggle copy.
print("Kaggle preflight: duplicate torch reinstall removed.")
"""
)

cells[5]["source"] = lines(
    """try:
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
except Exception as e:
    print("Torch import note:", e)
"""
)

cells[6]["source"] = lines(
    """try:
    import sys
    import torch

    print("python:", sys.executable)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
except Exception as e:
    print("Torch diagnostics note:", e)
"""
)

cells[8]["source"] = lines(
    """import asyncio
import atexit
import json
import os
import pathlib
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Dirs
for d in ["outputs", "outputs/checkpoints", "outputs/final_model", "outputs/plots"]:
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

install_cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "openenv-core[core]>=0.2.2",
    "unsloth",
    "unsloth_zoo",
    "trl>=0.12.0",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "matplotlib",
    "nest_asyncio",
]
print("Installing Kaggle notebook dependencies...")
subprocess.check_call(install_cmd)

import nest_asyncio
import numpy as np
import torch

nest_asyncio.apply()

# Determinism
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. In Kaggle, enable a GPU accelerator before running this notebook."
    )

name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name} | VRAM: {vram_gb:.1f} GB")


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def _safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Launch OpenEnv server as background subprocess
SERVER_URL = "http://127.0.0.1:8000"
server_log_path = pathlib.Path("outputs") / "server.log"
server_log = open(server_log_path, "w", encoding="utf-8")
server_proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "parkinsons_Motor.server.app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ],
    cwd=str(REPO_ROOT),
    stdout=server_log,
    stderr=subprocess.STDOUT,
)


def _shutdown_server():
    proc = globals().get("server_proc")
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


atexit.register(_shutdown_server)


def _wait_for_server(url, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"{url}/docs", timeout=1.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    return False


if not _wait_for_server(SERVER_URL):
    _shutdown_server()
    server_log.flush()
    raise RuntimeError(
        f"OpenEnv server failed to start within 60s. Check {server_log_path} for details."
    )

print(f"Server ready. Logs: {server_log_path}")
"""
)

cells[9]["source"] = lines(
    """import subprocess

subprocess.run(["nvidia-smi"], check=False)
"""
)

cells[16]["source"] = lines(
    """import asyncio
import matplotlib.pyplot as plt
from parkinsons_Motor.client import ParkinsonsMotorEnv
from parkinsons_Motor.models import ParkinsonsMotorAction


def policy_no_dbs(obs):
    return {"dbs_amplitude": 0.0, "dbs_pulse_width": 0.06, "motor_command": 0.5}


def policy_safety_aware(obs):
    beta = float(obs.get("beta_arv", 0.0))
    seload = float(obs.get("side_effect_load", 0.0))
    if beta > 0.6 and seload < 0.5:
        amp, pw = 2.0, 0.13
    elif seload >= 0.5:
        amp, pw = 0.5, 0.08
    else:
        amp, pw = 1.5, 0.11
    return {"dbs_amplitude": amp, "dbs_pulse_width": pw, "motor_command": 0.4}


def policy_const_mid(obs):
    return {"dbs_amplitude": 2.5, "dbs_pulse_width": 0.13, "motor_command": 0.4}


def _obs_to_dict(o):
    # Observation is a dataclass-like object. Grab the named fields we care about.
    return {k: float(getattr(o, k, 0.0)) for k in FIELD_LABELS}


async def _run_policy_episode(policy_fn, task_id):
    env = ParkinsonsMotorEnv(base_url=SERVER_URL)
    per_step = []
    try:
        r = await env.reset(task_id=task_id)
        obs = r.observation
        done = False
        while not done:
            a = policy_fn(_obs_to_dict(obs))
            step_res = await env.step(ParkinsonsMotorAction(
                dbs_amplitude=a["dbs_amplitude"],
                dbs_pulse_width=a["dbs_pulse_width"],
                dbs_frequency=130.0,
                motor_command=a["motor_command"],
                task_id=task_id,
            ))
            per_step.append(float(step_res.reward or 0.0))
            obs = step_res.observation
            done = bool(step_res.done)
    finally:
        await env.close()
    return per_step


def run_policy(policy_fn, task_id, n_episodes=5):
    return [run_async(_run_policy_episode(policy_fn, task_id)) for _ in range(n_episodes)]


baselines = {
    "no_dbs": policy_no_dbs,
    "safety_aware": policy_safety_aware,
    "const_mid": policy_const_mid,
}
baseline_curves = {}
for name, fn in baselines.items():
    runs = run_policy(fn, "beta_suppression", n_episodes=5)
    L = min(len(r) for r in runs)
    arr = np.array([r[:L] for r in runs])  # (episodes, steps)
    baseline_curves[name] = arr
    print(f"{name}: mean total reward = {arr.sum(axis=1).mean():.3f}")

plt.figure(figsize=(8, 5))
for name, arr in baseline_curves.items():
    plt.plot(arr.mean(axis=0), label=name)
plt.xlabel("Step")
plt.ylabel("Mean reward")
plt.title("Baseline policies — beta_suppression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/plots/baseline_curves.png", dpi=140)
plt.show()
"""
)

cells[18]["source"] = lines(
    """from transformers import GenerationConfig

FastLanguageModel.for_inference(model)  # enables fast inference kernels


@torch.no_grad()
def _generate(prompt: str, max_new_tokens: int = 80, temperature: float = 0.7) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    resp = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return resp


async def _run_llm_episode(task_id: str):
    env = ParkinsonsMotorEnv(base_url=SERVER_URL)
    prompts, responses, rewards = [], [], []
    grader_score = -1.0
    try:
        r = await env.reset(task_id=task_id)
        obs = r.observation
        max_steps = int(getattr(obs, "metadata", {}).get("episode_steps", 100) or 100)
        step = 0
        done = False
        while not done:
            prompt = build_prompt(_obs_to_dict(obs), task_id, step, max_steps)
            resp = _generate(prompt)
            action = parse_action(resp) or FALLBACK_ACTION
            step_res = await env.step(ParkinsonsMotorAction(
                dbs_amplitude=action["dbs_amplitude"],
                dbs_pulse_width=action["dbs_pulse_width"],
                dbs_frequency=130.0,
                motor_command=action["motor_command"],
                task_id=task_id,
            ))
            prompts.append(prompt)
            responses.append(resp)
            rewards.append(float(step_res.reward or 0.0))
            obs = step_res.observation
            done = bool(step_res.done)
            step += 1
            if done:
                grader_score = float(getattr(obs, "grader_score", -1.0) or -1.0)
    finally:
        await env.close()
    return {
        "prompts": prompts,
        "responses": responses,
        "rewards": rewards,
        "total_reward": float(sum(rewards)),
        "grader_score": grader_score,
    }


def run_episode(model_, tokenizer_, task_id, device_=None):
    return run_async(_run_llm_episode(task_id))


def collect_rollouts(model_, tokenizer_, task_id, n_episodes, device_=None):
    return [run_episode(model_, tokenizer_, task_id, device_) for _ in range(n_episodes)]


# Sanity check — 1 episode, untrained model
print("Sanity: running 1 episode on beta_suppression...")
sanity = run_episode(model, tokenizer, "beta_suppression", device)
assert isinstance(sanity["rewards"][0], float), "reward is not a float — abort"
print("Step 1 prompt (truncated):\\n", sanity["prompts"][0][-400:])
print("\\nStep 1 response:", sanity["responses"][0])
print("Step 1 reward:", sanity["rewards"][0])
print("Total reward:", sanity["total_reward"], "| grader:", sanity["grader_score"])
"""
)

cells[20]["source"] = lines(
    """# NOTE ON LEARNING-MODE CONTRIBUTION:
# TRL's GRPOTrainer (>=0.12) is built around (prompts_dataset, reward_funcs) — it generates
# completions itself and scores them. Our setting is different: the reward depends on a
# *sequential environment rollout*, not a single completion. There are two valid approaches:
#
#   (A) Manual GRPO loop: collect rollouts -> group-normalize rewards -> PPO-style update.
#       Simpler, no TRL wiring, and maps 1:1 onto our env. This is the approach below.
#
#   (B) TRL GRPOTrainer + env-backed reward_fn: treat each (obs -> action) step as an
#       independent prompt; the reward_fn runs a short env probe. Cleaner APIs but loses
#       multi-step credit assignment.
#
# The cell below implements (A). If you want (B) instead, replace the training loop with:
#     from trl import GRPOTrainer, GRPOConfig
#     trainer = GRPOTrainer(model, reward_funcs=[env_reward_fn], args=cfg, train_dataset=ds)
#     trainer.train()
# and implement `env_reward_fn(prompts, completions, **kw) -> list[float]`.

from torch.nn.utils import clip_grad_norm_

FastLanguageModel.for_training(model)

training_log = []

CURRICULUM = [
    {"task_id": "beta_suppression", "steps": 150, "stage": 1},
    {"task_id": "tremor_correction", "steps": 100, "stage": 2},
]

LR = 5e-6
PER_DEVICE_BS = 4
GRAD_ACC = 2
MAX_PROMPT_LEN = 512
MAX_COMPLETION_LEN = 80
N_ROLLOUT_EPISODES = 6  # group size for group-relative advantages

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
)


def _tokenize_pair(prompt: str, response: str, max_p=MAX_PROMPT_LEN, max_c=MAX_COMPLETION_LEN):
    p_ids = tokenizer(prompt, truncation=True, max_length=max_p, return_tensors="pt").input_ids[0]
    r_ids = tokenizer(
        response,
        truncation=True,
        max_length=max_c,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    input_ids = torch.cat([p_ids, r_ids], dim=0)
    labels = torch.cat([torch.full_like(p_ids, -100), r_ids], dim=0)
    return input_ids, labels


def _grpo_update_from_rollouts(rollouts, batch_size=PER_DEVICE_BS, grad_acc=GRAD_ACC):
    \"\"\"Group-relative policy update.
    - Compute group-normalized advantages from episode total_rewards.
    - Each (prompt, response) step inherits its episode's advantage.
    - Loss = - advantage * mean_logprob(response | prompt).
    \"\"\"
    totals = np.array([r["total_reward"] for r in rollouts], dtype=np.float32)
    mu, sd = totals.mean(), totals.std() + 1e-6
    adv_per_ep = (totals - mu) / sd

    flat = []  # (prompt, response, advantage)
    for adv, ep in zip(adv_per_ep, rollouts):
        for p, r in zip(ep["prompts"], ep["responses"]):
            flat.append((p, r, float(adv)))

    random.shuffle(flat)
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    n_micro = 0
    for i in range(0, len(flat), batch_size):
        batch = flat[i:i + batch_size]
        loss_accum = 0.0
        for (p, r, adv) in batch:
            ids, labels = _tokenize_pair(p, r)
            ids = ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            out = model(input_ids=ids, labels=labels)
            # out.loss is mean NLL over response tokens -> logprob proxy
            loss = out.loss * (-adv)  # maximize logprob when adv>0
            loss = loss / (len(batch) * grad_acc)
            loss.backward()
            loss_accum += loss.item()
        total_loss += loss_accum
        n_micro += 1
        if n_micro % grad_acc == 0:
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
    if n_micro % grad_acc != 0:
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()
    return total_loss


def _grpo_step_with_oom_retry(rollouts):
    global PER_DEVICE_BS
    try:
        return _grpo_update_from_rollouts(rollouts, batch_size=PER_DEVICE_BS)
    except torch.cuda.OutOfMemoryError:
        _safe_empty_cache()
        new_bs = max(1, PER_DEVICE_BS // 2)
        print(f"[OOM] retrying with batch_size={new_bs}")
        PER_DEVICE_BS = new_bs
        return _grpo_update_from_rollouts(rollouts, batch_size=new_bs)


for stage_cfg in CURRICULUM:
    task_id = stage_cfg["task_id"]
    stage = stage_cfg["stage"]
    n_steps = stage_cfg["steps"]
    print(f"\\n===== Stage {stage}: {task_id} ({n_steps} GRPO steps) =====")
    for gstep in range(n_steps):
        # 1) collect rollouts (inference mode)
        FastLanguageModel.for_inference(model)
        rollouts = collect_rollouts(model, tokenizer, task_id, N_ROLLOUT_EPISODES, device)
        # 2) policy update (training mode)
        FastLanguageModel.for_training(model)
        loss = _grpo_step_with_oom_retry(rollouts)

        if gstep % 10 == 0:
            mean_r = float(np.mean([r["total_reward"] for r in rollouts]))
            mean_g = float(np.mean([r["grader_score"] for r in rollouts]))
            entry = {
                "stage": stage,
                "step": gstep,
                "mean_reward": mean_r,
                "mean_grader_score": mean_g,
                "loss": loss,
            }
            training_log.append(entry)
            print(
                f"[stage {stage} step {gstep:3d}] reward={mean_r:.3f} "
                f"grader={mean_g:.3f} loss={loss:.4f}"
            )

    _safe_empty_cache()
    ckpt_dir = f"outputs/checkpoints/stage{stage}"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Saved LoRA checkpoint -> {ckpt_dir}")

# Final merged model (NOT naive 4bit merge)
model.save_pretrained_merged("outputs/final_model", tokenizer, save_method="merged_16bit")
print("Saved merged final model -> outputs/final_model")

with open("outputs/training_log.json", "w", encoding="utf-8") as f:
    json.dump(training_log, f, indent=2)
"""
)

cells[24]["source"] = lines(
    """# Load final model back (Unsloth 4-bit)
del model
_safe_empty_cache()

final_model, final_tok = FastLanguageModel.from_pretrained(
    model_name="outputs/final_model",
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(final_model)
# rebind module-level names so run_episode uses the final model
model = final_model
tokenizer = final_tok

EVAL_TASKS = ["beta_suppression", "tremor_correction"]
eval_curves = {}
summary_rows = []

for task in EVAL_TASKS:
    llm_runs = collect_rollouts(model, tokenizer, task, 10, device)
    base_runs = run_policy(policy_safety_aware, task, n_episodes=10)

    Ll = min(len(r["rewards"]) for r in llm_runs)
    Lb = min(len(r) for r in base_runs)
    llm_arr = np.array([r["rewards"][:Ll] for r in llm_runs])
    base_arr = np.array([r[:Lb] for r in base_runs])
    eval_curves[task] = (base_arr, llm_arr, llm_runs)

    llm_graders = [r["grader_score"] for r in llm_runs]
    summary_rows.append((
        task,
        "llm_trained",
        llm_arr.sum(axis=1).mean(),
        float(np.mean(llm_graders)),
        float(np.mean([g >= 0.6 for g in llm_graders])),
    ))
    summary_rows.append((
        task,
        "safety_aware",
        base_arr.sum(axis=1).mean(),
        float("nan"),
        float("nan"),
    ))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, task in zip(axes, EVAL_TASKS):
    base_arr, llm_arr, llm_runs = eval_curves[task]
    ax.plot(base_arr.mean(axis=0), label="safety_aware")
    ax.plot(llm_arr.mean(axis=0), label="llm_trained")
    ax.set_title(f"{task} | grader={np.mean([r['grader_score'] for r in llm_runs]):.3f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean reward")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/plots/comparison_curves.png", dpi=140)
plt.show()

# Summary table
print(f"\\n{'task':<22} {'model':<15} {'mean_reward':>12} {'mean_grader':>12} {'pass@0.6':>10}")
for row in summary_rows:
    t, m, r, g, p = row
    print(f"{t:<22} {m:<15} {r:>12.3f} {g:>12.3f} {p:>10.2f}")


# Side-by-side trace: base model vs trained, on tremor_correction
async def _trace_episode(which_model, which_tok, task_id):
    env = ParkinsonsMotorEnv(base_url=SERVER_URL)
    trace = []
    try:
        r = await env.reset(task_id=task_id)
        obs = r.observation
        done = False
        step = 0
        max_steps = int(getattr(obs, "metadata", {}).get("episode_steps", 100) or 100)
        while not done:
            prompt = build_prompt(_obs_to_dict(obs), task_id, step, max_steps)
            ids = which_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = which_model.generate(
                    **ids,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=which_tok.eos_token_id,
                )
            resp = which_tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
            action = parse_action(resp) or FALLBACK_ACTION
            sr = await env.step(ParkinsonsMotorAction(
                dbs_amplitude=action["dbs_amplitude"],
                dbs_pulse_width=action["dbs_pulse_width"],
                dbs_frequency=130.0,
                motor_command=action["motor_command"],
                task_id=task_id,
            ))
            o = sr.observation
            trace.append((
                step,
                float(o.beta_arv),
                float(o.tremor_arv),
                float(o.force_preserved),
                float(o.dbs_amplitude_ma),
                float(sr.reward or 0.0),
            ))
            obs = o
            done = bool(sr.done)
            step += 1
    finally:
        await env.close()
    return trace


base_model_fresh, base_tok_fresh = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(base_model_fresh)

print("\\nTrace: untrained base model on tremor_correction")
tr_base = run_async(_trace_episode(base_model_fresh, base_tok_fresh, "tremor_correction"))
print("\\nTrace: trained model on tremor_correction")
tr_trained = run_async(_trace_episode(final_model, final_tok, "tremor_correction"))

print(
    f"\\n{'step':>4} | {'beta':>6} {'tremor':>7} {'force':>6} {'amp':>5} {'r':>6}   || "
    f"{'beta':>6} {'tremor':>7} {'force':>6} {'amp':>5} {'r':>6}"
)
for (a, b) in zip(tr_base, tr_trained):
    print(
        f"{a[0]:>4} | {a[1]:>6.3f} {a[2]:>7.3f} {a[3]:>6.3f} {a[4]:>5.2f} {a[5]:>6.3f}   || "
        f"{b[1]:>6.3f} {b[2]:>7.3f} {b[3]:>6.3f} {b[4]:>5.2f} {b[5]:>6.3f}"
    )
"""
)

cells[26]["source"] = lines(
    """try:
    _shutdown_server()
    print("Server stopped.")
except Exception as e:
    print("Server shutdown note:", e)
"""
)

OUTPUT_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT_PATH}")
