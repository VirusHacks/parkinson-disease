"""
Generator for `dbs_train_gpu.ipynb` -- Kaggle GPU T4 SFT + curriculum GRPO notebook.

Why a generator and not the .ipynb directly:
- ipynb is JSON; editing by hand is painful.
- Each phase is a logical unit; this script keeps cell sources next to a short
  comment explaining what that cell does.
- `python build_train_gpu_notebook.py` regenerates the notebook from scratch.

Usage:
    python build_train_gpu_notebook.py        # writes ./dbs_train_gpu.ipynb
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List, Dict, Any


OUTPUT_PATH = Path(__file__).with_name("dbs_train_gpu.ipynb")


def _src(text: str) -> List[str]:
    """Convert a multi-line string to the list-of-lines format ipynb expects."""
    return text.splitlines(keepends=True)


def md(cells: List[Dict[str, Any]], text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": _src(text.strip("\n") + "\n"),
    })


def code(cells: List[Dict[str, Any]], text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text.strip("\n") + "\n"),
    })


def build_notebook() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 0. Header / overview                                                #
    # ------------------------------------------------------------------ #
    md(cells, """
# DBS - SFT cold-start + Curriculum GRPO on Kaggle GPU T4

Trains a small instruction-tuned LLM as a closed-loop adaptive DBS controller
for the **Parkinson's Motor** OpenEnv environment.

**Pipeline (5 phases):**

| Phase | Purpose | Output |
|---|---|---|
| 0 | Run rule-based baselines (`safety_aware`, `phase_aware`) on every task | `outputs/runs/baseline_rule_based.json` |
| 1 | Generate teacher trajectories, filter by grader score, format as SFT pairs | `outputs/sft_data/teacher_traces.jsonl` |
| 2 | SFT cold-start (LoRA) on Qwen2.5-Coder-1.5B-Instruct | `outputs/sft_adapter/` |
| 3 | Curriculum GRPO: easy -> easy+medium -> easy+medium+hard (manual loop, episode-level group-relative advantages) | `outputs/grpo_adapter/` |
| 4 | Held-out comparison on `hard`, `fragile_patient`, `refractory_patient`, `personalization_generalization` | `outputs/runs/post_train_eval.json` |
| 5 | Save merged LoRA -> fp16 model + plots | `outputs/final_model/` |

**Stack:** `transformers` + `peft` LoRA on a single Kaggle T4 GPU, **fp16** with
`torch.cuda.amp.autocast` + `GradScaler`. T4 has compute capability 7.5 with no
hardware bf16 support, so we use fp16 (Tensor Cores accelerate fp16 matmuls).

**Model choice:** Qwen2.5-Coder-1.5B-Instruct. Strong JSON-formatting prior (Coder
family) and small enough that GRPO rollouts stay snappy on a single T4 (15 GB).
Switching to `Qwen2.5-Coder-3B-Instruct` is a one-line `MODEL_NAME` change. For
`7B` you would also want `BitsAndBytesConfig(load_in_4bit=True)` (qlora) -- see
the comment in the model-loading cell.

**Resumability:** every phase checks for its output artefact and skips if found.
You can re-run the notebook top-to-bottom on a fresh Kaggle session and it picks
up where it left off.
""")

    # ------------------------------------------------------------------ #
    # 1. Repo path resolution + working directory                         #
    # ------------------------------------------------------------------ #
    md(cells, """
## 0a. Repo discovery

Tries each of these in order, stops at the first one that yields a working repo:

1. **Already on disk.** Current dir or any ancestor that contains
   `parkinsons_Motor/`.
2. **Kaggle dataset.** Anything matching `parkinsons_Motor/` under
   `/kaggle/input/` (one or two-level nested datasets are both fine).
   Mirrored into `/kaggle/working/<repo>/` so it's writable.
3. **GitHub clone.** Falls back to `git clone https://github.com/VirusHacks/parkinson-disease.git`
   into `/kaggle/working/parkinson-disease`. Edit `GIT_REPO_URL` /
   `GIT_REPO_BRANCH` below if you've forked.

If you want to short-circuit the search, set `REPO_ROOT_OVERRIDE` to the
absolute path of the repo before running the cell.
""")

    code(cells, '''
import os
import shutil
import subprocess
import sys
from pathlib import Path


GIT_REPO_URL    = "https://github.com/VirusHacks/parkinson-disease.git"
GIT_REPO_BRANCH = "main"
REPO_ROOT_OVERRIDE = os.environ.get("REPO_ROOT_OVERRIDE")  # absolute path or None

_MARKERS = [
    Path("parkinsons_Motor") / "pyproject.toml",
    Path("parkinsons_Motor") / "__init__.py",
    Path("parkinsons_Motor") / "server" / "app.py",
]


def _is_repo_root(path: Path) -> bool:
    return any((path / m).exists() for m in _MARKERS)


def _search_dirs() -> list[Path]:
    """All places we'll look for an already-mounted/cloned repo."""
    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)
    for base in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if base.exists():
            try:
                candidates.extend(p for p in base.iterdir() if p.is_dir())
            except PermissionError:
                pass
    return candidates


def _scan_for_repo() -> Path | None:
    seen: set[Path] = set()
    for cand in _search_dirs():
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if _is_repo_root(cand):
            return cand

    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not base.exists():
            continue
        for marker_rel in _MARKERS:
            try:
                hit = next(base.rglob(str(marker_rel)), None)
            except (PermissionError, OSError):
                hit = None
            if hit is not None:
                return hit.parent.parent
    return None


def _clone_repo(target: Path) -> Path:
    if not (target / ".git").exists() and not _is_repo_root(target):
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {GIT_REPO_URL} (branch {GIT_REPO_BRANCH}) -> {target}")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "--branch", GIT_REPO_BRANCH,
            GIT_REPO_URL, str(target),
        ])
    if not _is_repo_root(target):
        raise FileNotFoundError(
            f"Cloned but {target} is missing expected files: {_MARKERS[0]}"
        )
    return target


if REPO_ROOT_OVERRIDE:
    SOURCE_REPO_ROOT = Path(REPO_ROOT_OVERRIDE).resolve()
    if not _is_repo_root(SOURCE_REPO_ROOT):
        raise FileNotFoundError(
            f"REPO_ROOT_OVERRIDE={REPO_ROOT_OVERRIDE} is not a valid repo "
            f"root (missing {_MARKERS[0]})"
        )
else:
    found = _scan_for_repo()
    if found is None:
        target = Path("/kaggle/working") / "parkinson-disease"
        if not target.parent.exists():
            target = Path.cwd() / "parkinson-disease"
        SOURCE_REPO_ROOT = _clone_repo(target)
    else:
        SOURCE_REPO_ROOT = found

WORK_REPO_ROOT = Path("/kaggle/working") / SOURCE_REPO_ROOT.name
if (str(SOURCE_REPO_ROOT).startswith("/kaggle/input/")
        and Path("/kaggle/working").exists()):
    if not _is_repo_root(WORK_REPO_ROOT):
        print(f"Copying repo: {SOURCE_REPO_ROOT} -> {WORK_REPO_ROOT}")
        shutil.copytree(
            SOURCE_REPO_ROOT,
            WORK_REPO_ROOT,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc",
                "outputs", ".ipynb_checkpoints",
            ),
        )
    REPO_ROOT = WORK_REPO_ROOT.resolve()
else:
    REPO_ROOT = SOURCE_REPO_ROOT.resolve()

PKG_ROOT = REPO_ROOT / "parkinsons_Motor"
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for sub in [
    "outputs",
    "outputs/runs",
    "outputs/sft_data",
    "outputs/sft_adapter",
    "outputs/grpo_adapter",
    "outputs/checkpoints",
    "outputs/final_model",
    "outputs/plots",
]:
    (REPO_ROOT / sub).mkdir(parents=True, exist_ok=True)

print("source    :", SOURCE_REPO_ROOT)
print("repo root :", REPO_ROOT)
print("package   :", PKG_ROOT)
print("cwd       :", Path.cwd())
print("python    :", sys.executable)
''')

    # ------------------------------------------------------------------ #
    # 2. Dependency install                                               #
    # ------------------------------------------------------------------ #
    md(cells, """
## 0b. Install dependencies

Kaggle GPU notebooks ship `torch` (CUDA 12.x) + a recent `transformers`. We
add `peft`, `accelerate`, `openenv-core`, plus plotting / dataset bits.

If you switch `MODEL_NAME` to a 7B model, also install `bitsandbytes` and use
`BitsAndBytesConfig(load_in_4bit=True)` in the model-load cell -- the 7B fp16
weights (~14 GB) won't comfortably fit on a single T4 alongside activations.
""")

    code(cells, '''
import subprocess, sys

DEPS = [
    "openenv-core[core]>=0.2.2",
    "transformers>=4.45,<4.55",
    "peft>=0.13",
    "accelerate>=0.34",
    "datasets>=3.0",
    "matplotlib",
    "nest_asyncio",
    "pydantic>=2.0",
]

cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + DEPS
print("Installing:", " ".join(DEPS))
subprocess.check_call(cmd)
print("Dependencies OK")
''')

    # ------------------------------------------------------------------ #
    # 3. CUDA device + diagnostics                                        #
    # ------------------------------------------------------------------ #
    md(cells, """
## 0c. CUDA device setup

Kaggle T4 sessions expose either 1 or 2 T4 GPUs (15 GB each). We use a single
GPU for simplicity. Multi-GPU via `accelerate launch` is possible but not
needed for a 1.5 B model.

Why **fp16** and not bf16: T4 (compute capability 7.5) has no hardware bf16
support. PyTorch can run bf16 ops as software fallback but it's slower and not
worth it. fp16 + Tensor Cores is the right path on T4.

Stability strategy: load weights in fp16, keep LoRA params in fp32 (peft does
this by default), train with `torch.cuda.amp.autocast(dtype=torch.float16)`
plus a `GradScaler` so loss scaling protects against fp16 underflow during
backward.
""")

    code(cells, '''
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"CUDA mem    : {props.total_memory / 1e9:.2f} GB")
    print(f"CUDA SM     : {props.major}.{props.minor}")
    print(f"CUDA count  : {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("WARNING: no CUDA available, falling back to CPU (training will be very slow)")

DTYPE = torch.float16
print("torch       :", torch.__version__)
print("dtype       :", DTYPE)
''')

    # ------------------------------------------------------------------ #
    # 4. Configuration cell                                               #
    # ------------------------------------------------------------------ #
    md(cells, """
## 0d. Configuration

All knobs in one place. Edit and re-run from this cell to retune without
re-running phase 0.
""")

    code(cells, '''
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# --- Model -----------------------------------------------------------------
MODEL_NAME       = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LEN      = 1024
MAX_NEW_TOKENS   = 96
GENERATION_TEMP  = 0.7
GENERATION_TOP_P = 0.95

# --- LoRA ------------------------------------------------------------------
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# --- Phase 1: teacher data --------------------------------------------------
TEACHER_EPISODES_PER_TASK = {
    "easy":               40,
    "medium":             40,
    "hard":               30,
    "fragile_patient":    20,
    "refractory_patient": 20,
}
TEACHER_GRADER_MARGIN = 0.05

# --- Phase 2: SFT -----------------------------------------------------------
SFT_EPOCHS         = 1
SFT_LR             = 2e-5
SFT_BATCH_SIZE     = 4
SFT_GRAD_ACCUM     = 4
SFT_WARMUP_STEPS   = 50
SFT_WEIGHT_DECAY   = 0.0
SFT_LOG_EVERY      = 25

# --- Phase 3: GRPO ----------------------------------------------------------
@dataclass
class CurriculumStage:
    name: str
    task_mix: Dict[str, float]
    group_size: int
    n_grpo_steps: int

CURRICULUM: List[CurriculumStage] = [
    CurriculumStage("stage1_easy",
                    task_mix={"easy": 1.0},
                    group_size=8, n_grpo_steps=60),
    CurriculumStage("stage2_easy_medium",
                    task_mix={"easy": 0.3, "medium": 0.7},
                    group_size=6, n_grpo_steps=80),
    CurriculumStage("stage3_full",
                    task_mix={"easy": 0.2, "medium": 0.3, "hard": 0.5},
                    group_size=4, n_grpo_steps=100),
]

GRPO_LR              = 5e-6
GRPO_GRAD_CLIP       = 1.0
GRPO_FORMAT_BONUS    = 0.05
GRPO_PARSE_PENALTY   = 0.20
GRPO_ADV_EPS         = 1e-6
GRPO_LOG_EVERY       = 5
GRPO_CHECKPOINT_EVERY = 25

# --- Phase 4: held-out comparison -------------------------------------------
HELDOUT_TASKS        = ["hard",
                        "fragile_patient",
                        "refractory_patient",
                        "personalization_generalization"]
HELDOUT_EPISODES     = 4

# --- Determinism ------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)
random.seed(SEED)

print("Model        :", MODEL_NAME)
print("Curriculum   :", [s.name for s in CURRICULUM])
print("Heldout tasks:", HELDOUT_TASKS)
''')

    # ------------------------------------------------------------------ #
    # 5. Start the OpenEnv server in-process                              #
    # ------------------------------------------------------------------ #
    md(cells, """
## 0e. Start the OpenEnv server

The Parkinson's environment is pure NumPy -- no GPU needed. We run the FastAPI
server as a background subprocess on the host CPU and talk to it over
`http://127.0.0.1:8000`. Idempotent: skip if already running.
""")

    code(cells, '''
import atexit
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request


SERVER_URL = "http://127.0.0.1:8000"
SERVER_LOG = pathlib.Path("outputs") / "server.log"


def _server_alive(url: str = SERVER_URL, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/docs", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def _wait_for_server(url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _server_alive(url):
            return True
        time.sleep(0.5)
    return False


server_proc = globals().get("server_proc")
if server_proc is None or server_proc.poll() is not None:
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    server_log = open(SERVER_LOG, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "parkinsons_Motor.server.app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(REPO_ROOT),
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    def _shutdown_server():
        proc = globals().get("server_proc")
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    atexit.register(_shutdown_server)

if not _wait_for_server(SERVER_URL):
    raise RuntimeError(f"OpenEnv server failed to start within 60s. See {SERVER_LOG}.")

print("Server ready at", SERVER_URL)
''')

    # ------------------------------------------------------------------ #
    # 6. Async helper (Jupyter)                                           #
    # ------------------------------------------------------------------ #
    code(cells, '''
import asyncio

import nest_asyncio
nest_asyncio.apply()


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
        return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)
''')

    # ------------------------------------------------------------------ #
    # 7. Prompt templates                                                 #
    # ------------------------------------------------------------------ #
    md(cells, """
## 0f. Prompt template

We share one prompt format across SFT data generation, SFT training, and GRPO
rollouts. Format parity is essential -- the GRPO reward depends on the model
emitting parseable JSON, and we want SFT to teach exactly the format the env
expects at inference.
""")

    code(cells, '''
import textwrap


SYSTEM_PROMPT = textwrap.dedent("""
You are an expert closed-loop DBS controller managing Parkinsonian motor symptoms in real time.
Every step is a short clinical control decision: suppress pathological activity, preserve movement,
avoid overstimulation, and keep enough safety budget for the rest of the episode.

Return JSON only:
{"dbs_amplitude": X, "dbs_pulse_width": X, "dbs_frequency": X}

Important:
- Do not output prose, markdown, or explanation.
- Do not include motor_command. It is handled separately.
- Use pulse_width = 0.13 and frequency = 130 unless there is a strong safety reason to deviate.
- Prefer smooth amplitude changes. Avoid jumps larger than about 0.3 mA per step.

Clinical meaning of observations:
- beta_arv, tremor_arv: pathological. Lower is better.
- force_preserved: retained motor force. Higher is better.
- side_effect_load: cumulative stimulation burden. Stay below the task budget.
- beta_trend / tremor_trend: positive means worsening, negative means improving.
- side_effect_rate: positive means burden is still rising.
- gamma_arv: overstimulation warning. If high, back off quickly.

Decision policy:
- If gamma_arv > 0.55 or side_effect_load is near budget, reduce amplitude.
- If tremor_arv > 0.55 or beta_arv > 0.60, treat actively (>= 1.2 mA unless unsafe).
- If symptoms worsen and safety is acceptable, increase by ~0.10 - 0.15 mA.
- If symptoms improve and side effects rise, hold or reduce slightly.
- If stable, taper toward the lowest effective dose.
""").strip()


_TASK_CONTEXT = {
    "easy": ("EASY / Calm Start. Responsive patient, mild symptoms. "
             "Ceiling 1.5 mA, side-effect budget 0.55."),
    "medium": ("MEDIUM / Rescue Phase. Symptoms escalating, second deterioration possible. "
               "Ceiling 1.8 mA, side-effect budget 0.60."),
    "hard": ("HARD / Full Episode. Refractory patient, long-horizon, multiple crises. "
             "Ceiling 2.4 mA, side-effect budget 0.40."),
    "fragile_patient": ("FRAGILE. Tight safety budget. "
                        "Ceiling 1.4 mA, side-effect budget 0.26."),
    "refractory_patient": ("REFRACTORY. Drug-resistant, brute-force amplitude does not help. "
                           "Ceiling 2.4 mA, side-effect budget 0.48."),
    "personalization_generalization": ("MIXED PROFILES. Read the profile, infer the window. "
                                       "Variable ceiling per episode."),
    "exercise_bout": ("EXERCISE BOUT. Motor surge in first 30%, dyskinesia risk later."),
    "medication_interaction": ("MEDICATION INTERACTION. Off-med crisis mid-episode."),
    "nocturnal_transition": ("NOCTURNAL. Sleep targets tighten in the second half."),
    "surgical_followup": ("SURGICAL FOLLOWUP. Microlesion ceiling for first 25%."),
}


def build_user_prompt(step: int, obs: dict, task_id: str, history: list) -> str:
    recent = "\\n".join(history[-3:]) if history else "(first step)"
    ctx = _TASK_CONTEXT.get(task_id, "")

    return textwrap.dedent(f"""
    TASK: {task_id} | STEP: {step}
    {ctx}

    STATE:
      beta_arv         = {obs.get('beta_arv', 0.0):.4f}
      tremor_arv       = {obs.get('tremor_arv', 0.0):.4f}
      force_preserved  = {obs.get('force_preserved', 0.0):.4f}
      side_effect_load = {obs.get('side_effect_load', 0.0):.4f}
      beta_trend       = {obs.get('beta_trend', 0.0):+.4f}
      tremor_trend     = {obs.get('tremor_trend', 0.0):+.4f}
      side_effect_rate = {obs.get('side_effect_rate', 0.0):+.4f}
      gamma_arv        = {obs.get('gamma_arv', 0.0):.4f}
      dbs_entrainment  = {obs.get('dbs_entrainment', 0.0):.4f}
      stim_washout     = {obs.get('stim_washout', 0.0):.4f}
      tracking_accuracy= {obs.get('tracking_accuracy', 0.0):.4f}
      target_output    = {obs.get('target_output', 0.0):.4f}
      medication_phase = {obs.get('medication_phase', 0.5):.4f}

    RECENT:
    {recent}

    Output JSON only now.
    """).strip()


def parse_action_json(text: str) -> dict | None:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(text[s:e])
    except json.JSONDecodeError:
        return None


def clamp_action(d: dict | None, target_output: float = 0.0) -> dict:
    """Apply env-side ranges. Falls back to safe defaults if `d` is None."""
    if not d:
        d = {"dbs_amplitude": 1.0, "dbs_pulse_width": 0.13, "dbs_frequency": 130.0}
    return {
        "motor_command":   float(max(-1.0, min(1.0, target_output))),
        "dbs_amplitude":   float(max(0.0,  min(5.0,   d.get("dbs_amplitude", 1.0)))),
        "dbs_pulse_width": float(max(0.06, min(0.20,  d.get("dbs_pulse_width", 0.13)))),
        "dbs_frequency":   float(max(60.0, min(185.0, d.get("dbs_frequency", 130.0)))),
    }
''')

    # ------------------------------------------------------------------ #
    # 8. Phase 0 / 1 helpers                                              #
    # ------------------------------------------------------------------ #
    md(cells, """
## Phase 0 - Rule-based baselines (the "before" reference)

Two teacher policies, both pure NumPy reactive controllers -- no learning. They
serve double duty:
1. **Phase 0 baselines** the trained model will be compared against.
2. **Phase 1 teachers** for SFT cold-start data.

`policy_safety_aware` is the one already in the repo (defensive, rule-of-thumb).
`policy_phase_aware` is new -- uses `beta_trend`, `gamma_arv`, `medication_phase`,
and `stim_washout` for the dynamics that the easy rules miss (especially on
hard / refractory tasks).
""")

    code(cells, '''
def policy_safety_aware(obs: dict) -> dict:
    """Repo's existing rule-based controller."""
    beta = float(obs.get("beta_arv", 0.0))
    se   = float(obs.get("side_effect_load", 0.0))
    if beta > 0.6 and se < 0.5:
        amp, pw = 2.0, 0.13
    elif se >= 0.5:
        amp, pw = 0.5, 0.08
    else:
        amp, pw = 1.5, 0.11
    return {"dbs_amplitude": amp, "dbs_pulse_width": pw, "dbs_frequency": 130.0}


def policy_phase_aware(obs: dict) -> dict:
    """Phase-aware reactive controller. Better than safety_aware on hard/refractory."""
    beta   = float(obs.get("beta_arv", 0.0))
    tremor = float(obs.get("tremor_arv", 0.0))
    se     = float(obs.get("side_effect_load", 0.0))
    gamma  = float(obs.get("gamma_arv", 0.0))
    bt     = float(obs.get("beta_trend", 0.0))
    tt     = float(obs.get("tremor_trend", 0.0))
    sr     = float(obs.get("side_effect_rate", 0.0))
    med    = float(obs.get("medication_phase", 0.5))
    wash   = float(obs.get("stim_washout", 0.0))

    severity = 0.55 * tremor + 0.45 * beta
    amp = 0.5 + severity * 1.6
    amp *= (1.4 - 0.4 * med)
    amp *= 1.0 + max(0.0, bt) * 1.2
    amp *= 1.0 + max(0.0, tt) * 0.6
    if gamma > 0.55:
        amp *= 0.55
    if se > 0.45:
        amp *= 0.65
    if sr > 0.02:
        amp *= 0.85
    amp *= 1.0 - 0.3 * wash

    amp = max(0.3, min(amp, 2.4))
    freq = 130.0 if beta > tremor else 110.0
    pw   = 0.13
    return {"dbs_amplitude": amp, "dbs_pulse_width": pw, "dbs_frequency": freq}
''')

    code(cells, '''
from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv  # noqa: E402


def _obs_to_dict(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return {k: getattr(o, k) for k in dir(o) if not k.startswith("_")}


async def _rollout_policy(policy_fn, task_id: str, max_steps: int | None = None):
    """Run one episode with a Python policy_fn(obs_dict) -> action_dict."""
    env = ParkinsonsMotorEnv(base_url=SERVER_URL)
    await env.__aenter__()
    prompts, responses, per_step_obs, actions, rewards = [], [], [], [], []
    grader_score = -1.0
    success = False
    step_idx = 0
    try:
        r = await env.reset(task_id=task_id)
        obs = r.observation
        obs_d = _obs_to_dict(obs)
        done = False
        while not done:
            if max_steps is not None and step_idx >= max_steps:
                break
            target = float(obs_d.get("target_output", 0.0))
            action_d = clamp_action(policy_fn(obs_d), target_output=target)
            sr = await env.step(ParkinsonsMotorAction(
                motor_command=action_d["motor_command"],
                dbs_amplitude=action_d["dbs_amplitude"],
                dbs_pulse_width=action_d["dbs_pulse_width"],
                dbs_frequency=action_d["dbs_frequency"],
                task_id=task_id,
            ))
            per_step_obs.append(obs_d)
            actions.append(action_d)
            rewards.append(float(sr.reward or 0.0))
            prompts.append(build_user_prompt(step_idx, obs_d, task_id, []))
            responses.append(json.dumps({
                "dbs_amplitude":   round(action_d["dbs_amplitude"], 3),
                "dbs_pulse_width": round(action_d["dbs_pulse_width"], 3),
                "dbs_frequency":   round(action_d["dbs_frequency"], 1),
            }))
            obs = sr.observation
            obs_d = _obs_to_dict(obs)
            done = bool(sr.done)
            step_idx += 1
            if done:
                grader_score = float(obs_d.get("grader_score", -1.0) or -1.0)
                success = bool(obs_d.get("episode_success", False))
    finally:
        await env.__aexit__(None, None, None)

    return {
        "task_id":      task_id,
        "prompts":      prompts,
        "responses":    responses,
        "per_step_obs": per_step_obs,
        "actions":      actions,
        "rewards":      rewards,
        "total_reward": float(sum(rewards)),
        "grader_score": grader_score,
        "success":      success,
        "steps":        step_idx,
    }
''')

    # ------------------------------------------------------------------ #
    # 9. Phase 0 execution                                                #
    # ------------------------------------------------------------------ #
    code(cells, '''
PHASE0_PATH = REPO_ROOT / "outputs" / "runs" / "baseline_rule_based.json"
TASKS_FOR_BASELINE = ["easy", "medium", "hard",
                      "fragile_patient", "refractory_patient",
                      "personalization_generalization"]
EPISODES_PER_TASK_BASELINE = 3


def run_phase0() -> dict:
    if PHASE0_PATH.exists():
        print(f"[skip] phase 0 already done -> {PHASE0_PATH}")
        return json.loads(PHASE0_PATH.read_text())

    results = {p: {} for p in ["safety_aware", "phase_aware"]}
    for policy_name, fn in [("safety_aware", policy_safety_aware),
                            ("phase_aware",  policy_phase_aware)]:
        for task_id in TASKS_FOR_BASELINE:
            scores, success = [], []
            for ep in range(EPISODES_PER_TASK_BASELINE):
                ep_data = run_async(_rollout_policy(fn, task_id))
                scores.append(ep_data["grader_score"])
                success.append(ep_data["success"])
            mean_score = float(np.mean(scores))
            success_rate = float(np.mean(success))
            results[policy_name][task_id] = {
                "mean_grader_score": mean_score,
                "success_rate":      success_rate,
                "n_episodes":        len(scores),
                "scores":            scores,
            }
            print(f"  {policy_name:<14} {task_id:<32} grader={mean_score:.3f} pass={success_rate:.2f}")

    PHASE0_PATH.write_text(json.dumps(results, indent=2))
    print(f"Phase 0 complete -> {PHASE0_PATH}")
    return results


phase0_results = run_phase0()
''')

    # ------------------------------------------------------------------ #
    # 10. Phase 1: Teacher data generation                                #
    # ------------------------------------------------------------------ #
    md(cells, """
## Phase 1 - SFT teacher trajectories

Run rule-based teachers on each task, keep only episodes where
`grader_score >= success_threshold + 0.05`, format each step as a
`(system + user_prompt) -> JSON_action` SFT pair.

Output: `outputs/sft_data/teacher_traces.jsonl` -- one JSON object per line.
""")

    code(cells, '''
from parkinsons_Motor.tasks import get_task  # noqa: E402


SFT_DATA_PATH = REPO_ROOT / "outputs" / "sft_data" / "teacher_traces.jsonl"
SFT_STATS_PATH = REPO_ROOT / "outputs" / "sft_data" / "stats.json"


def _format_sft_pair(prompt_user: str, action_dict: dict) -> tuple[str, str]:
    response = json.dumps({
        "dbs_amplitude":   round(action_dict["dbs_amplitude"], 3),
        "dbs_pulse_width": round(action_dict["dbs_pulse_width"], 3),
        "dbs_frequency":   round(action_dict["dbs_frequency"], 1),
    })
    return prompt_user, response


def run_phase1() -> dict:
    if SFT_DATA_PATH.exists() and SFT_STATS_PATH.exists():
        stats = json.loads(SFT_STATS_PATH.read_text())
        print(f"[skip] phase 1 already done: {stats['kept_pairs']} pairs at {SFT_DATA_PATH}")
        return stats

    teachers = [policy_safety_aware, policy_phase_aware]

    kept_pairs = 0
    rejected_pairs = 0
    per_task_kept: Dict[str, int] = {}
    per_task_episodes_passed: Dict[str, int] = {}

    with SFT_DATA_PATH.open("w", encoding="utf-8") as out_f:
        for task_id, n_episodes in TEACHER_EPISODES_PER_TASK.items():
            try:
                threshold = get_task(task_id).success_threshold + TEACHER_GRADER_MARGIN
            except Exception:
                threshold = 0.5
            print(f"  generating {n_episodes} eps x {len(teachers)} teachers for {task_id} "
                  f"(keep >= {threshold:.2f})")
            for teacher in teachers:
                for _ in range(n_episodes):
                    ep = run_async(_rollout_policy(teacher, task_id))
                    if ep["grader_score"] < threshold:
                        rejected_pairs += len(ep["prompts"])
                        continue
                    per_task_episodes_passed[task_id] = per_task_episodes_passed.get(task_id, 0) + 1
                    history: list[str] = []
                    for step_i, (obs_d, action_d) in enumerate(zip(ep["per_step_obs"], ep["actions"])):
                        prompt_user = build_user_prompt(step_i, obs_d, task_id, history)
                        prompt_user_full = f"{SYSTEM_PROMPT}\\n\\n{prompt_user}"
                        _, response = _format_sft_pair(prompt_user, action_d)
                        out_f.write(json.dumps({
                            "prompt":        prompt_user_full,
                            "response":      response,
                            "task_id":       task_id,
                            "grader_score":  ep["grader_score"],
                            "step":          step_i,
                            "teacher":       teacher.__name__,
                        }) + "\\n")
                        kept_pairs += 1
                        per_task_kept[task_id] = per_task_kept.get(task_id, 0) + 1
                        history.append(
                            f"step={step_i} amp={action_d['dbs_amplitude']:.2f} "
                            f"pw={action_d['dbs_pulse_width']:.3f} "
                            f"freq={action_d['dbs_frequency']:.0f}"
                        )

    stats = {
        "path":                     str(SFT_DATA_PATH),
        "kept_pairs":               kept_pairs,
        "rejected_pairs":           rejected_pairs,
        "per_task_kept":            per_task_kept,
        "per_task_episodes_passed": per_task_episodes_passed,
        "model":                    MODEL_NAME,
    }
    SFT_STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Phase 1 complete: {kept_pairs} pairs kept, {rejected_pairs} rejected -> {SFT_DATA_PATH}")
    return stats


phase1_stats = run_phase1()
print("Per-task kept pairs:", phase1_stats["per_task_kept"])
''')

    # ------------------------------------------------------------------ #
    # 11. Phase 2: SFT setup - load model + tokenizer                     #
    # ------------------------------------------------------------------ #
    md(cells, """
## Phase 2 - SFT cold-start

Load Qwen-Coder in fp16, attach a LoRA adapter (PEFT promotes LoRA params to
fp32 for stability), train on the teacher pairs for 1 epoch with mixed-precision
autocast + GradScaler. Saves the adapter so phase 3 can pick it up.

**For Qwen-7B on T4:** uncomment the `BitsAndBytesConfig` block to load in 4-bit
(qlora). The base stays quantized through training; only LoRA params are trained.
For phase 5 final-merge you would re-load the fp16 base separately and then
attach the adapter.
""")

    code(cells, '''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


def _load_base_model_and_tokenizer():
    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- Optional 4-bit qlora path (uncomment for 7B) ---
    # from transformers import BitsAndBytesConfig
    # bnb_cfg = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # )
    # mdl = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME, quantization_config=bnb_cfg, device_map={"": 0},
    #     trust_remote_code=True,
    # )
    # ----------------------------------------------------

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(device)
    mdl.config.use_cache = False
    return mdl, tok


tokenizer = None
model = None


def _ensure_model_loaded():
    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = _load_base_model_and_tokenizer()
    return model, tokenizer
''')

    # ------------------------------------------------------------------ #
    # 12. Phase 2: Tokenize SFT data                                      #
    # ------------------------------------------------------------------ #
    code(cells, '''
from torch.utils.data import Dataset, DataLoader


class SFTPairsDataset(Dataset):
    """Each sample is a (prompt, response) pair packed into one token sequence
    where prompt tokens have label=-100 and response tokens carry the LM target."""

    def __init__(self, jsonl_path: str, tok, max_length: int = MAX_SEQ_LEN):
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append((obj["prompt"], obj["response"]))
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, response = self.examples[idx]
        full = prompt + "\\n" + response + self.tok.eos_token
        enc = self.tok(
            full,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)

        prompt_only = self.tok(
            prompt + "\\n",
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_only.input_ids.shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def _make_sft_loader():
    _ensure_model_loaded()
    ds = SFTPairsDataset(str(SFT_DATA_PATH), tokenizer, MAX_SEQ_LEN)
    print(f"SFT dataset size: {len(ds)}")
    return DataLoader(
        ds,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
''')

    # ------------------------------------------------------------------ #
    # 13. Phase 2: SFT training loop                                      #
    # ------------------------------------------------------------------ #
    code(cells, '''
import math
import time

from transformers import get_cosine_schedule_with_warmup


SFT_ADAPTER_DIR = REPO_ROOT / "outputs" / "sft_adapter"


def _attach_lora(model_):
    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    return get_peft_model(model_, cfg)


def run_phase2() -> Path:
    if (SFT_ADAPTER_DIR / "adapter_config.json").exists():
        print(f"[skip] SFT adapter already exists at {SFT_ADAPTER_DIR}")
        return SFT_ADAPTER_DIR

    global model
    _ensure_model_loaded()
    model = _attach_lora(model)
    model.print_trainable_parameters()

    loader = _make_sft_loader()
    n_steps = max(1, math.ceil(len(loader) / SFT_GRAD_ACCUM)) * SFT_EPOCHS
    print(f"SFT optim steps: {n_steps}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=SFT_LR,
        weight_decay=SFT_WEIGHT_DECAY,
    )
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=SFT_WARMUP_STEPS, num_training_steps=n_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model.train(True)
    micro = 0
    optim_step = 0
    losses = []
    t0 = time.time()
    for epoch in range(SFT_EPOCHS):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(dtype=DTYPE, enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out.loss / SFT_GRAD_ACCUM
            scaler.scale(loss).backward()
            micro += 1
            losses.append(float(out.loss.detach().to("cpu")))
            if micro % SFT_GRAD_ACCUM == 0:
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                optim_step += 1
                if optim_step % SFT_LOG_EVERY == 0:
                    avg = float(np.mean(losses[-SFT_LOG_EVERY * SFT_GRAD_ACCUM:]))
                    elapsed = time.time() - t0
                    print(f"  [SFT] step {optim_step:>4}/{n_steps}  loss={avg:.4f}  "
                          f"lr={sched.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s")

    SFT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(SFT_ADAPTER_DIR))
    tokenizer.save_pretrained(str(SFT_ADAPTER_DIR))

    loss_path = REPO_ROOT / "outputs" / "plots" / "sft_loss.json"
    loss_path.write_text(json.dumps({"per_micro_loss": losses}, indent=2))
    print(f"SFT done -> {SFT_ADAPTER_DIR}")
    return SFT_ADAPTER_DIR


sft_adapter_path = run_phase2()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
''')

    # ------------------------------------------------------------------ #
    # 14. Phase 2 sanity check                                            #
    # ------------------------------------------------------------------ #
    md(cells, """
### Phase 2 sanity check

Run 2 episodes per public task with the SFT'd model and print grader scores +
JSON-parse rate. Should see format-compliance ~= 100% and grader >= rule-based
on `easy`.
""")

    code(cells, '''
from transformers import GenerationConfig


@torch.no_grad()
def llm_action(model_, tok_, step: int, obs: dict, task_id: str, history: list):
    user = build_user_prompt(step, obs, task_id, history)
    full = SYSTEM_PROMPT + "\\n\\n" + user + "\\n"
    enc = tok_(full, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GENERATION_TEMP,
        top_p=GENERATION_TOP_P,
        pad_token_id=tok_.eos_token_id,
    )
    out = model_.generate(**enc, generation_config=gen_cfg)
    resp = tok_.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    parsed = parse_action_json(resp)
    return resp, parsed


async def _rollout_llm(model_, tok_, task_id: str, max_steps: int | None = None):
    env = ParkinsonsMotorEnv(base_url=SERVER_URL)
    await env.__aenter__()
    prompts, responses, rewards, actions = [], [], [], []
    parse_ok = 0
    parse_total = 0
    grader_score = -1.0
    success = False
    step_i = 0
    history: list[str] = []
    try:
        r = await env.reset(task_id=task_id)
        obs = r.observation
        obs_d = _obs_to_dict(obs)
        done = False
        while not done:
            if max_steps is not None and step_i >= max_steps:
                break
            target = float(obs_d.get("target_output", 0.0))
            resp, parsed = llm_action(model_, tok_, step_i, obs_d, task_id, history)
            parse_total += 1
            if parsed is not None:
                parse_ok += 1
            action_d = clamp_action(parsed, target_output=target)
            sr = await env.step(ParkinsonsMotorAction(
                motor_command=action_d["motor_command"],
                dbs_amplitude=action_d["dbs_amplitude"],
                dbs_pulse_width=action_d["dbs_pulse_width"],
                dbs_frequency=action_d["dbs_frequency"],
                task_id=task_id,
            ))
            prompts.append(build_user_prompt(step_i, obs_d, task_id, history))
            responses.append(resp)
            rewards.append(float(sr.reward or 0.0))
            actions.append(action_d)
            history.append(
                f"step={step_i} amp={action_d['dbs_amplitude']:.2f} "
                f"pw={action_d['dbs_pulse_width']:.3f} freq={action_d['dbs_frequency']:.0f}"
            )
            obs = sr.observation
            obs_d = _obs_to_dict(obs)
            done = bool(sr.done)
            step_i += 1
            if done:
                grader_score = float(obs_d.get("grader_score", -1.0) or -1.0)
                success = bool(obs_d.get("episode_success", False))
    finally:
        await env.__aexit__(None, None, None)
    return {
        "task_id":       task_id,
        "prompts":       prompts,
        "responses":     responses,
        "rewards":       rewards,
        "actions":       actions,
        "total_reward":  float(sum(rewards)),
        "grader_score":  grader_score,
        "success":       success,
        "parse_ok":      parse_ok,
        "parse_total":   parse_total,
        "steps":         step_i,
    }


print("Sanity: SFT model, 2 episodes per public task")
model.train(False)
for task in ["easy", "medium"]:
    eps = [run_async(_rollout_llm(model, tokenizer, task)) for _ in range(2)]
    g  = float(np.mean([e["grader_score"] for e in eps]))
    p  = sum(e["parse_ok"] for e in eps) / max(1, sum(e["parse_total"] for e in eps))
    print(f"  {task:<8} grader={g:.3f}  parse_rate={p:.2%}")
''')

    # ------------------------------------------------------------------ #
    # 15. Phase 3: GRPO setup                                             #
    # ------------------------------------------------------------------ #
    md(cells, """
## Phase 3 - Curriculum GRPO

Manual GRPO loop (the env reward is sequential, so TRL's `GRPOTrainer` is not a
good fit out of the box). Per training step:

1. Sample a task from the current curriculum stage's `task_mix`.
2. Roll out `group_size` episodes with the current LoRA adapter.
3. Reward each episode: `grader_score + format_bonus * frac_valid_json
   - parse_penalty * frac_parse_failures`.
4. Compute z-score within the group -> per-episode advantage.
5. Loss = `-advantage * mean_response_logprob_per_episode`, summed over the
   group, with grad accumulation across episodes (autocast forward + scaled
   backward like SFT).
""")

    code(cells, '''
from torch.nn.utils import clip_grad_norm_


GRPO_ADAPTER_DIR = REPO_ROOT / "outputs" / "grpo_adapter"
GRPO_LOG_PATH = REPO_ROOT / "outputs" / "plots" / "grpo_log.json"


def _sample_task(stage: CurriculumStage) -> str:
    keys = list(stage.task_mix.keys())
    weights = np.array([stage.task_mix[k] for k in keys], dtype=np.float64)
    weights /= weights.sum()
    return str(np.random.choice(keys, p=weights))


def _episode_format_score(ep: dict) -> float:
    if ep["parse_total"] == 0:
        return 0.0
    valid_frac = ep["parse_ok"] / ep["parse_total"]
    return GRPO_FORMAT_BONUS * valid_frac - GRPO_PARSE_PENALTY * (1.0 - valid_frac)


def _episode_reward(ep: dict) -> float:
    base = max(0.0, ep["grader_score"]) if ep["grader_score"] >= 0 else 0.0
    return base + _episode_format_score(ep)


def _tokenize_for_grpo(prompt_user: str, response: str):
    """Return (full_input_ids, prompt_len) or None if response would be truncated."""
    prompt_full = SYSTEM_PROMPT + "\\n\\n" + prompt_user + "\\n"
    p = tokenizer(prompt_full, truncation=True, max_length=MAX_SEQ_LEN,
                  return_tensors="pt", add_special_tokens=True)
    r = tokenizer(response + tokenizer.eos_token, truncation=True,
                  max_length=MAX_NEW_TOKENS, return_tensors="pt",
                  add_special_tokens=False)
    p_ids = p.input_ids[0]
    r_ids = r.input_ids[0]
    full = torch.cat([p_ids, r_ids], dim=0)
    if full.shape[0] > MAX_SEQ_LEN:
        keep_prompt = MAX_SEQ_LEN - r_ids.shape[0]
        if keep_prompt < 8:
            return None
        full = torch.cat([p_ids[:keep_prompt], r_ids], dim=0)
        prompt_len = keep_prompt
    else:
        prompt_len = p_ids.shape[0]
    return full, prompt_len


def _episode_logprob_loss(ep: dict, advantage: float):
    """Scalar loss = -advantage * mean_NLL over response tokens, averaged across steps."""
    losses = []
    for prompt_user, response in zip(ep["prompts"], ep["responses"]):
        tok_out = _tokenize_for_grpo(prompt_user, response)
        if tok_out is None:
            continue
        full, prompt_len = tok_out
        ids = full.unsqueeze(0).to(device)
        labels = full.clone()
        labels[:prompt_len] = -100
        labels = labels.unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(dtype=DTYPE, enabled=torch.cuda.is_available()):
            out = model(input_ids=ids, labels=labels)
            losses.append(out.loss * (-advantage))
    if not losses:
        return None
    return torch.stack(losses).mean()


def run_phase3():
    if (GRPO_ADAPTER_DIR / "adapter_config.json").exists():
        print(f"[skip] GRPO adapter already exists at {GRPO_ADAPTER_DIR}")
        return GRPO_ADAPTER_DIR

    log_records: list[dict] = []
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=GRPO_LR,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    global_step = 0
    for stage in CURRICULUM:
        print(f"\\n===== {stage.name}: tasks={stage.task_mix} group={stage.group_size} steps={stage.n_grpo_steps} =====")
        for gstep in range(stage.n_grpo_steps):
            task_id = _sample_task(stage)

            model.train(False)
            group = [run_async(_rollout_llm(model, tokenizer, task_id)) for _ in range(stage.group_size)]
            rewards = np.array([_episode_reward(ep) for ep in group], dtype=np.float64)
            mu = rewards.mean()
            sd = rewards.std() + GRPO_ADV_EPS
            advs = (rewards - mu) / sd

            model.train(True)
            optim.zero_grad(set_to_none=True)
            total_loss = 0.0
            n_used = 0
            for ep, adv in zip(group, advs):
                loss = _episode_logprob_loss(ep, float(adv))
                if loss is None:
                    continue
                scaler.scale(loss / max(1, len(group))).backward()
                total_loss += float(loss.detach().to("cpu"))
                n_used += 1
            if n_used > 0:
                scaler.unscale_(optim)
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRPO_GRAD_CLIP)
                scaler.step(optim)
                scaler.update()

            global_step += 1
            mean_grader = float(np.mean([ep["grader_score"] for ep in group]))
            mean_parse  = float(np.mean([ep["parse_ok"] / max(1, ep["parse_total"]) for ep in group]))

            rec = {
                "stage":        stage.name,
                "global_step":  global_step,
                "task_id":      task_id,
                "mean_reward":  float(rewards.mean()),
                "mean_grader":  mean_grader,
                "parse_rate":   mean_parse,
                "loss":         total_loss / max(1, n_used),
            }
            log_records.append(rec)
            if global_step % GRPO_LOG_EVERY == 0:
                print(f"  [{stage.name} step {gstep:>3}/{stage.n_grpo_steps}] "
                      f"task={task_id:<32} reward={rec['mean_reward']:+.3f} "
                      f"grader={mean_grader:.3f} parse={mean_parse:.2%} loss={rec['loss']:+.4f}")

            if global_step % GRPO_CHECKPOINT_EVERY == 0:
                ckpt = REPO_ROOT / "outputs" / "checkpoints" / f"grpo_step{global_step}"
                model.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))

        stage_dir = REPO_ROOT / "outputs" / "checkpoints" / f"grpo_{stage.name}"
        model.save_pretrained(str(stage_dir))
        tokenizer.save_pretrained(str(stage_dir))
        GRPO_LOG_PATH.write_text(json.dumps(log_records, indent=2))

    GRPO_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(GRPO_ADAPTER_DIR))
    tokenizer.save_pretrained(str(GRPO_ADAPTER_DIR))
    GRPO_LOG_PATH.write_text(json.dumps(log_records, indent=2))
    print(f"\\nPhase 3 complete -> {GRPO_ADAPTER_DIR}")
    return GRPO_ADAPTER_DIR


grpo_adapter_path = run_phase3()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
''')

    # ------------------------------------------------------------------ #
    # 16. Phase 4: Held-out comparison                                    #
    # ------------------------------------------------------------------ #
    md(cells, """
## Phase 4 - Held-out comparison

Compares the trained model on `hard` (was trained on but small slice) and three
**never-seen** expert tasks: `fragile_patient`, `refractory_patient`,
`personalization_generalization`.

Three baselines for the comparison plot:
- `safety_aware` (rule-based, weak)
- `phase_aware` (rule-based, stronger, our SFT teacher)
- `trained` -- the LoRA-merged Qwen-Coder

If the trained model beats `phase_aware` on at least one held-out task, that's
the "generalization" headline for the pitch.
""")

    code(cells, '''
HELDOUT_EVAL_PATH = REPO_ROOT / "outputs" / "runs" / "post_train_eval.json"
HELDOUT_EPISODES_PER_TASK = HELDOUT_EPISODES


async def _score_policy_on_task(policy_kind: str, task_id: str, n_episodes: int):
    scores, parse_rates = [], []
    for _ in range(n_episodes):
        if policy_kind in ("safety_aware", "phase_aware"):
            fn = policy_safety_aware if policy_kind == "safety_aware" else policy_phase_aware
            ep = await _rollout_policy(fn, task_id)
            scores.append(ep["grader_score"])
            parse_rates.append(1.0)
        elif policy_kind == "trained":
            ep = await _rollout_llm(model, tokenizer, task_id)
            scores.append(ep["grader_score"])
            parse_rates.append(ep["parse_ok"] / max(1, ep["parse_total"]))
        else:
            raise ValueError(policy_kind)
    return {
        "mean_grader": float(np.mean(scores)),
        "parse_rate":  float(np.mean(parse_rates)),
        "scores":      scores,
    }


def run_phase4() -> dict:
    results: dict = {"trained": {}, "phase_aware": {}, "safety_aware": {}}
    model.train(False)
    for task_id in HELDOUT_TASKS:
        for kind in ["trained", "phase_aware", "safety_aware"]:
            results[kind][task_id] = run_async(
                _score_policy_on_task(kind, task_id, HELDOUT_EPISODES_PER_TASK)
            )
            print(f"  {kind:<14} {task_id:<32} grader={results[kind][task_id]['mean_grader']:.3f}")
    HELDOUT_EVAL_PATH.write_text(json.dumps(results, indent=2))
    print(f"Phase 4 complete -> {HELDOUT_EVAL_PATH}")
    return results


phase4_results = run_phase4()
''')

    # ------------------------------------------------------------------ #
    # 17. Phase 4 plots                                                   #
    # ------------------------------------------------------------------ #
    code(cells, '''
import matplotlib.pyplot as plt


PLOTS_DIR = REPO_ROOT / "outputs" / "plots"


def _plot_grpo_curves():
    if not GRPO_LOG_PATH.exists():
        return
    log = json.loads(GRPO_LOG_PATH.read_text())
    if not log:
        return
    steps   = [r["global_step"] for r in log]
    rewards = [r["mean_reward"] for r in log]
    graders = [r["mean_grader"] for r in log]
    parses  = [r["parse_rate"] for r in log]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, rewards); axes[0].set_title("GRPO group mean reward")
    axes[0].set_xlabel("global step"); axes[0].set_ylabel("reward"); axes[0].grid(alpha=0.3)
    axes[1].plot(steps, graders, color="tab:green"); axes[1].set_title("Grader score")
    axes[1].set_xlabel("global step"); axes[1].set_ylabel("grader"); axes[1].grid(alpha=0.3)
    axes[2].plot(steps, parses, color="tab:red"); axes[2].set_title("JSON parse rate")
    axes[2].set_xlabel("global step"); axes[2].set_ylabel("parse %"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "grpo_curves.png"
    plt.savefig(out, dpi=140)
    plt.show()
    print("Saved", out)


def _plot_heldout_comparison():
    if not HELDOUT_EVAL_PATH.exists():
        return
    res = json.loads(HELDOUT_EVAL_PATH.read_text())
    tasks = HELDOUT_TASKS
    kinds = ["safety_aware", "phase_aware", "trained"]
    x = np.arange(len(tasks))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(kinds):
        ys = [res[k][t]["mean_grader"] for t in tasks]
        ax.bar(x + (i - 1) * w, ys, w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel("mean grader_score (higher is better)")
    ax.set_title("Held-out comparison: trained LLM vs rule-based baselines")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = PLOTS_DIR / "heldout_comparison.png"
    plt.savefig(out, dpi=140)
    plt.show()
    print("Saved", out)


_plot_grpo_curves()
_plot_heldout_comparison()


print("\\n=== Held-out summary ===")
res = json.loads(HELDOUT_EVAL_PATH.read_text())
print(f"{'task':<34} {'safety_aware':>14} {'phase_aware':>14} {'trained':>10}")
for t in HELDOUT_TASKS:
    print(f"{t:<34} "
          f"{res['safety_aware'][t]['mean_grader']:>14.3f} "
          f"{res['phase_aware'][t]['mean_grader']:>14.3f} "
          f"{res['trained'][t]['mean_grader']:>10.3f}")
''')

    # ------------------------------------------------------------------ #
    # 18. Phase 5: Save merged final model                                #
    # ------------------------------------------------------------------ #
    md(cells, """
## Phase 5 - Save merged final model

Merge LoRA into the fp16 base, save to `outputs/final_model/`. Optionally push
to HF Hub.

Note: if you swapped to 4-bit qlora training in phase 2, do NOT call
`merge_and_unload()` directly on the quantized base (the helpguide warns about
this). Instead, re-load the base in fp16, attach the trained adapter, and merge
that. The default fp16 path here just works.
""")

    code(cells, '''
FINAL_MODEL_DIR = REPO_ROOT / "outputs" / "final_model"


def run_phase5():
    print("Merging LoRA into base ...")
    merged = model.merge_and_unload()
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    print(f"Saved merged model -> {FINAL_MODEL_DIR}")


run_phase5()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
''')

    code(cells, '''
# Optional: push to Hugging Face Hub. Set HF_TOKEN as a Kaggle Secret first.
# Uncomment the lines below to push.

# from huggingface_hub import login
# login(os.environ.get("HF_TOKEN"))
# REPO_ID = "your-username/dbs-llm-controller"
# from transformers import AutoModelForCausalLM, AutoTokenizer
# m = AutoModelForCausalLM.from_pretrained(str(FINAL_MODEL_DIR))
# t = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
# m.push_to_hub(REPO_ID); t.push_to_hub(REPO_ID)
print("Phase 5 done. To publish, fill in REPO_ID and uncomment the push block.")
''')

    # ------------------------------------------------------------------ #
    # 19. Cleanup                                                         #
    # ------------------------------------------------------------------ #
    md(cells, """
## Cleanup

Stop the in-process OpenEnv server. Safe to skip if you want to keep running
ad-hoc scoring after training.
""")

    code(cells, '''
try:
    server_proc.terminate()
    server_proc.wait(timeout=5)
    print("Server stopped.")
except Exception as exc:
    print("Server shutdown note:", exc)
''')

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {
                "name":             "python",
                "pygments_lexer":   "ipython3",
                "codemirror_mode":  {"name": "ipython", "version": 3},
                "mimetype":         "text/x-python",
                "file_extension":   ".py",
                "nbconvert_exporter": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def verify(nb: Dict[str, Any]) -> None:
    """Compile every code cell as Python to catch syntax errors before shipping.

    Also flags lingering TPU references that suggest an incomplete migration.
    """
    errors = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src, filename=f"cell[{i}]")
        except SyntaxError as exc:
            errors += 1
            print(f"SYNTAX ERROR in cell {i}: {exc}")

    forbidden = ["torch_xla", "xm.xla_device", "xm.optimizer_step",
                 "xm.mark_step", "TPU"]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for tok in forbidden:
            if tok in src:
                errors += 1
                print(f"LEFTOVER {tok!r} in cell {i}")

    if errors:
        raise SystemExit(f"verify() found {errors} problems")
    print("verify(): all code cells parse, no TPU leftovers.")


def main() -> None:
    nb = build_notebook()
    verify(nb)
    OUTPUT_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_cells = len(nb["cells"])
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md   = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"Wrote {OUTPUT_PATH} ({n_cells} cells: {n_code} code, {n_md} markdown)")


if __name__ == "__main__":
    main()
