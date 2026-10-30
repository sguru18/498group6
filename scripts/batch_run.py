"""Batch runner for SWE-bench Lite agent sessions.

Runs multiple SWE-bench Lite tasks sequentially using the real agent harness
(Docker sandbox + vLLM endpoint). Writes per-task JSONL logs and a final
batch_summary.json to the output directory.

Usage::

    python scripts/batch_run.py --n 20 --output-dir data/sessions/ [--seed 42]
    python scripts/batch_run.py --task-ids django__django-11049 astropy__astropy-1234 \\
        --output-dir data/sessions/

Environment variables:
    VLLM_BASE_URL  Base URL of the OpenAI-compatible vLLM endpoint.
                   Default: http://localhost:8000/v1
    VLLM_MODEL     Model name to pass to the API.  Default: /models
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow ``python scripts/batch_run.py`` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_tasks(
    task_ids: list[str] | None,
    n: int,
    seed: int,
    repos: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the list of task dicts to run.

    If *task_ids* is given, load each by ID; otherwise sample *n* tasks from
    SWE-bench Lite using *seed* for reproducibility.

    Args:
        task_ids: Explicit list of SWE-bench instance IDs, or None.
        n: Number of tasks to sample when *task_ids* is None.
        seed: RNG seed for reproducible sampling.

    Returns:
        List of task dicts.
    """
    from harness.swebench_tasks import load_lite_tasks, get_task

    if task_ids:
        tasks: list[dict[str, Any]] = []
        for tid in task_ids:
            try:
                tasks.append(get_task(tid))
            except KeyError as exc:
                print(f"[WARNING] Task '{tid}' not found in SWE-bench Lite: {exc}")
        return tasks
    return load_lite_tasks(n=n, seed=seed, repos=repos)


def _completed(path: Path) -> bool:
    """True if the file ends with a current-harness summary that isn't a crash."""
    from harness.run_session import HARNESS_VERSION

    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        return False
    summary = json.loads(lines[-1])
    return (summary.get("__summary__") is True
            and summary.get("harness_version") == HARNESS_VERSION
            and summary.get("stop_reason") != "error")


def run_batch(
    tasks: list[dict[str, Any]],
    output_dir: str,
    max_steps: int = 45,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run all *tasks* (``workers`` at a time), writing logs to *output_dir*.

    Errors on individual tasks are caught and logged; the batch continues.

    Args:
        tasks: List of task dicts from swebench_tasks.
        output_dir: Directory for per-task JSONL files and batch_summary.json.

    Returns:
        List of per-task result dicts:
        ``{"task_id": str, "status": "ok"|"error", "error_msg": str|None}``.
    """
    from harness.run_session import run_real_session

    total = len(tasks)

    def _one(idx: int, task: dict[str, Any]) -> dict[str, Any]:
        task_id: str = task["task_id"]
        existing = Path(output_dir) / f"{task_id}.jsonl"
        if existing.exists() and _completed(existing):
            print(f"[{idx}/{total}] Skipping {task_id} (already completed)")
            return {"task_id": task_id, "status": "ok", "error_msg": None}
        print(f"[{idx}/{total}] Running {task_id} …")
        try:
            out_path = run_real_session(task_id, output_dir, max_steps=max_steps)
            print(f"  ✓  Done → {out_path}")
            return {"task_id": task_id, "status": "ok", "error_msg": None}
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  ✗  Error for {task_id}: {msg}")
            traceback.print_exc()
            return {"task_id": task_id, "status": "error", "error_msg": msg}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda it: _one(*it), enumerate(tasks, start=1)))
    errored = sum(r["status"] == "error" for r in results)

    # ------------------------------------------------------------------
    # Write batch summary
    # ------------------------------------------------------------------
    summary_path = Path(output_dir) / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    batch_meta: dict[str, Any] = {
        "total": total,
        "completed": total - errored,
        "errored": errored,
        "ts": _now_iso(),
        "tasks": results,
    }
    summary_path.write_text(json.dumps(batch_meta, indent=2))

    ok_count = total - errored
    print(
        f"\nCompleted {ok_count}/{total} tasks ({errored} errored). "
        f"Session logs in {output_dir}"
    )
    print(f"Batch summary written to {summary_path}")
    return results


def main() -> None:
    """Entry point for the batch runner CLI."""
    parser = argparse.ArgumentParser(
        description="Run multiple SWE-bench Lite tasks with the real agent harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of tasks to sample from SWE-bench Lite (ignored if --task-ids given).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible task sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sessions/",
        help="Directory to write session JSONL files and batch_summary.json.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        metavar="TASK_ID",
        default=None,
        help="Explicit list of SWE-bench task IDs to run (overrides --n and --seed).",
    )
    parser.add_argument("--repos", nargs="+", default=None,
                        help="Only sample tasks from these repos, e.g. django/django sympy/sympy")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--workers", type=int, default=1,
                        help="Sessions to run concurrently (each uses its own container)")
    parser.add_argument("--task-file", default=None,
                        help="File with one task id per line (e.g. from scripts/prescreen.py)")
    args = parser.parse_args()
    if args.task_file:
        args.task_ids = [l.strip() for l in Path(args.task_file).read_text().splitlines() if l.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[batch_run] Loading tasks …")
    tasks = _resolve_tasks(
        task_ids=args.task_ids,
        n=args.n,
        seed=args.seed,
        repos=args.repos,
    )
    if not tasks:
        print("No tasks to run. Exiting.")
        sys.exit(1)

    print(f"[batch_run] Running {len(tasks)} task(s) → {args.output_dir}")
    run_batch(tasks, args.output_dir, max_steps=args.max_steps, workers=args.workers)


if __name__ == "__main__":
    main()
