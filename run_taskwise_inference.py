"""
Run local inference one task at a time with long pauses between runs.

This helps when you want to avoid provider rate limits or credit spikes by
splitting easy, medium, and hard evaluation into separate runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "runs"
DEFAULT_TASKS = ["easy", "medium", "hard"]
TASKS = [
    task.strip()
    for task in os.getenv("INFERENCE_TASKS", ",".join(DEFAULT_TASKS)).split(",")
    if task.strip()
]
TASKWISE_SLEEP_SECONDS = float(os.getenv("TASKWISE_SLEEP_SECONDS", "20"))
REQUEST_SLEEP_SECONDS = os.getenv("INFERENCE_SLEEP_SECONDS", "2.0")
INTER_TASK_SLEEP_SECONDS = os.getenv("INFERENCE_TASK_SLEEP_SECONDS", "5.0")

TASK_LABELS = {
    "easy": "Easy / Calm Start",
    "medium": "Medium / Rescue Phase",
    "hard": "Hard / Full Episode",
    "beta_suppression": "Easy / Calm Start",
    "tremor_correction": "Medium / Rescue Phase",
    "full_episode": "Hard / Full Episode",
    "fragile_patient": "Extension / Fragile Patient",
    "refractory_patient": "Extension / Refractory Patient",
    "personalization_generalization": "Extension / Generalization",
}


def run_single_task(task_id: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["INFERENCE_TASKS"] = task_id
    env["INFERENCE_SLEEP_SECONDS"] = REQUEST_SLEEP_SECONDS
    env["INFERENCE_TASK_SLEEP_SECONDS"] = INTER_TASK_SLEEP_SECONDS
    env["INFERENCE_OUTPUT_BASENAME"] = f"inference_{task_id}"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_local_inference.py")],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"inference_{task_id}.stdout.log").write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        (OUTPUT_DIR / f"inference_{task_id}.stderr.log").write_text(result.stderr, encoding="utf-8")

    report_path = OUTPUT_DIR / f"inference_{task_id}.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    return {
        "tasks": [task_id],
        "task_results": [{"task_id": task_id, "score": 0.0, "success": False}],
        "mean_score": 0.0,
        "subprocess_returncode": result.returncode,
    }


def write_summary(reports: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = OUTPUT_DIR / "taskwise_inference_summary.json"
    summary_md = OUTPUT_DIR / "taskwise_inference_summary.md"

    task_results: List[Dict[str, Any]] = []
    for report in reports:
        task_results.extend(report.get("task_results", []))

    mean_score = (
        sum(item.get("score", 0.0) for item in task_results) / len(task_results)
        if task_results
        else 0.0
    )

    payload = {
        "tasks": TASKS,
        "taskwise_sleep_seconds": TASKWISE_SLEEP_SECONDS,
        "request_sleep_seconds": REQUEST_SLEEP_SECONDS,
        "inter_task_sleep_seconds": INTER_TASK_SLEEP_SECONDS,
        "mean_score": mean_score,
        "task_results": task_results,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Taskwise Inference Summary",
        "",
        "This run executes one task at a time to reduce provider load and make failures easier to inspect.",
        "",
        f"- Tasks: `{', '.join(TASKS)}`",
        f"- Between task runs: `{TASKWISE_SLEEP_SECONDS}` s",
        f"- Between model requests: `{REQUEST_SLEEP_SECONDS}` s",
        f"- Between tasks inside each run: `{INTER_TASK_SLEEP_SECONDS}` s",
        f"- Mean score: `{mean_score:.4f}`",
        "",
        "| Task ID | Friendly Name | Score | Success |",
        "|---|---|---:|---:|",
    ]
    for item in task_results:
        task_id = item["task_id"]
        lines.append(
            f"| `{task_id}` | {TASK_LABELS.get(task_id, task_id)} | {item.get('score', 0.0):.4f} | {'PASS' if item.get('success', False) else 'FAIL'} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    reports: List[Dict[str, Any]] = []

    print("=" * 60, flush=True)
    print("Taskwise Inference Runner", flush=True)
    print(f"Tasks: {TASKS}", flush=True)
    print(f"Pause between task runs: {TASKWISE_SLEEP_SECONDS}s", flush=True)
    print("=" * 60, flush=True)

    for index, task_id in enumerate(TASKS):
        print(f"\nRunning {task_id} ({TASK_LABELS.get(task_id, task_id)})", flush=True)
        report = run_single_task(task_id)
        reports.append(report)

        task_result = report.get("task_results", [{}])[0]
        print(
            f"  -> score={task_result.get('score', 0.0):.4f} success={task_result.get('success', False)}",
            flush=True,
        )

        if index != len(TASKS) - 1 and TASKWISE_SLEEP_SECONDS > 0:
            print(f"  -> sleeping {TASKWISE_SLEEP_SECONDS:.1f}s before the next task run", flush=True)
            time.sleep(TASKWISE_SLEEP_SECONDS)

    write_summary(reports)
    print("\nSaved taskwise summary to outputs/runs/taskwise_inference_summary.json", flush=True)
    print("Saved taskwise summary to outputs/runs/taskwise_inference_summary.md", flush=True)


if __name__ == "__main__":
    main()
