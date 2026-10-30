"""Prescreen SWE-bench Lite tasks without the LLM and pick an error-diverse batch.

For each candidate task this applies the test patch and runs the baseline tests
(Docker only, no GPU), recording whether the setup is usable and which error
types the reproduction tests fail with. It then selects a batch balanced across
error categories (logic / runtime / environment / syntax) and writes the task
ids to a file that scripts/batch_run.py can consume with --task-file.

Usage:
    python scripts/prescreen.py --n 60 --select 25 --workers 3 \\
        --output data/prescreen.json --selected data/selected_tasks.txt [--remove-images]
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import test_runner


def _local_images() -> set[str]:
    out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                         capture_output=True, text=True).stdout
    return set(out.split())


def screen(task: dict[str, Any], remove_image: bool) -> dict[str, Any]:
    from harness.docker_sandbox import DockerSandbox, _instance_to_image
    from harness.run_session import build_stages, prepare_sandbox

    image = _instance_to_image(task["task_id"])
    had_image = image in _local_images()
    t0 = time.time()
    try:
        with DockerSandbox(task, timeout=120) as sb:
            base = prepare_sandbox(task, sb)
        stages = build_stages(task, base["statuses"]) if base["valid"] else []
        rec = {
            "task_id": task["task_id"],
            "repo": task["repo"],
            "valid": base["valid"] and base["patch_applied"],
            "crashed": base["crashed"],
            "error_types": base["error_types"],
            "error_category": test_runner.error_category(base["error_types"]) or "logic",
            "f2p_failing": base["f2p_failing"],
            "f2p_total": base["f2p_total"],
            "p2p_passing": base["p2p_passing"],
            "n_stages": len(stages),
            "baseline_seconds": round(time.time() - t0),
            "summary_line": base["summary_line"],
        }
    except Exception as exc:
        rec = {"task_id": task["task_id"], "repo": task["repo"], "valid": False,
               "error": f"{type(exc).__name__}: {exc}"}
    if remove_image and not had_image:
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
    return rec


def select_balanced(records: list[dict[str, Any]], k: int, max_seconds: int) -> list[str]:
    """Round-robin over error categories (rarest first), preferring multi-stage tasks."""
    pool: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("valid") and r.get("baseline_seconds", 0) <= max_seconds:
            pool.setdefault(r["error_category"], []).append(r)
    for rs in pool.values():
        rs.sort(key=lambda r: (-r["n_stages"], r["baseline_seconds"]))
    order = sorted(pool, key=lambda c: len(pool[c]))
    chosen: list[str] = []
    while len(chosen) < k and any(pool.values()):
        for cat in order:
            if pool[cat] and len(chosen) < k:
                chosen.append(pool[cat].pop(0)["task_id"])
    return chosen


def main() -> None:
    from harness.swebench_tasks import load_lite_tasks

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=60, help="Candidate tasks to screen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repos", nargs="+", default=None)
    ap.add_argument("--select", type=int, default=25, help="Size of the balanced batch")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-baseline-seconds", type=int, default=300,
                    help="Skip tasks whose test files take longer than this to run")
    ap.add_argument("--output", default="data/prescreen.json")
    ap.add_argument("--selected", default="data/selected_tasks.txt")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="Task ids to skip (e.g. ones with a session currently running)")
    ap.add_argument("--remove-images", action="store_true",
                    help="Delete images that were pulled only for screening (saves disk)")
    args = ap.parse_args()

    out_path = Path(args.output)
    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        done = {r["task_id"]: r for r in json.loads(out_path.read_text())}

    tasks = [t for t in load_lite_tasks(n=args.n, seed=args.seed, repos=args.repos)
             if t["task_id"] not in args.exclude
             and (t["task_id"] not in done or "error" in done[t["task_id"]])]
    print(f"[prescreen] {len(done)} cached, screening {len(tasks)} tasks with {args.workers} workers")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(screen, t, args.remove_images): t for t in tasks}
        for fut in as_completed(futures):
            rec = fut.result()
            done[rec["task_id"]] = rec
            print(f"  {rec['task_id']:40} valid={rec.get('valid')} "
                  f"cat={rec.get('error_category')} errs={rec.get('error_types')} "
                  f"stages={rec.get('n_stages')} {rec.get('baseline_seconds', '')}s {rec.get('error', '')}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(list(done.values()), indent=2))

    records = list(done.values())
    chosen = select_balanced(records, args.select, args.max_baseline_seconds)
    Path(args.selected).write_text("\n".join(chosen) + "\n")
    cats: dict[str, int] = {}
    for r in records:
        if r["task_id"] in chosen:
            cats[r["error_category"]] = cats.get(r["error_category"], 0) + 1
    print(f"\n[prescreen] {sum(bool(r.get('valid')) for r in records)}/{len(records)} valid; "
          f"selected {len(chosen)} -> {args.selected}  by category: {cats}")


if __name__ == "__main__":
    main()
