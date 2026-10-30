"""Extract X* data points (failure -> next edit) and report log quality per session.

An X* data point is a test failure followed by the next code edit, with no other
test run in between. For each one we record how many tokens the model generated
between the failure and the edit (``gen_tokens``, from vLLM usage — harness v2
only), how much the context grew (``ctx_tokens``), the error category, and
whether the next test run after the edit passed.

Usage:
    python scripts/analyze_sessions.py --sessions-dir data/sessions/ [--output data/labeled/xstar.json]
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import test_runner

_FAIL_EVENTS = {"test_fail", "test_error"}
_SYMBOL = {"test_fail": "F", "test_error": "X", "test_pass": "P", "edit": "E", "edit_failed": "e",
           "blocked": "B", "explore": ".", "bash_error": "!", "reasoning": "", "final_eval": "|"}


def _v1_event(step: dict[str, Any]) -> str:
    """Heuristic event label for logs written before the harness emitted `event`."""
    if step.get("role") != "tool":
        return "reasoning"
    out = step.get("tool_output") or ""
    if step.get("tool_name") == "str_replace":
        return "edit" if out.startswith("Success") else "edit_failed"
    if out.startswith("[BLOCKED"):
        return "blocked"
    cmd = (step.get("tool_args") or {}).get("command", "")
    if re.search(r"sed -i|cat\s*>|perl -pi", cmd):
        return "edit"
    kind = test_runner.classify_bash_output(out)["kind"]
    return {"test_pass": "test_pass", "test_fail": "test_fail", "error": "bash_error"}.get(kind, "explore")


def load(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    steps, summary = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("__summary__"):
            summary = obj
        else:
            steps.append(obj)
    for s in steps:
        if "event" not in s:
            s["event"] = _v1_event(s)
            if s["event"] == "test_fail":
                cls = test_runner.classify_bash_output(s.get("tool_output") or "")
                s["error_category"] = cls["error_category"]
                s["error_types"] = cls["error_types"]
    return steps, summary


def xstar_points(task_id: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for i, s in enumerate(steps):
        if s["event"] not in _FAIL_EVENTS:
            continue
        for j in range(i + 1, len(steps)):
            ev = steps[j]["event"]
            if ev in _FAIL_EVENTS or ev in ("test_pass", "final_eval"):
                break
            if ev != "edit":
                continue
            nxt = next((steps[k] for k in range(j + 1, len(steps))
                        if steps[k]["event"] in _FAIL_EVENTS | {"test_pass"}), None)
            g0, g1 = s.get("gen_tokens_so_far"), steps[j].get("gen_tokens_so_far")
            points.append({
                "task_id": task_id,
                "fail_step": s["step_id"],
                "edit_step": steps[j]["step_id"],
                "steps_between": j - i,
                "gen_tokens": (g1 - g0) if g0 is not None and g1 is not None else None,
                "ctx_tokens": steps[j]["tokens_so_far"] - s["tokens_so_far"],
                "error_category": s.get("error_category"),
                "error_types": s.get("error_types", []),
                "stage": s.get("stage"),
                "fail_source": s.get("source", "agent"),
                "next_test": nxt["event"] if nxt else None,
            })
            break
    return points


def pattern(steps: list[dict[str, Any]]) -> str:
    return "".join(_SYMBOL.get(s["event"], "?") for s in steps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions-dir", default="data/sessions/")
    ap.add_argument("--output", default=None, help="Write all X* points as JSON")
    args = ap.parse_args()

    all_points: list[dict[str, Any]] = []
    print(f"{'task':38} {'res':4} {'stop':16} {'stg':5} {'F':>3} {'E':>3} {'X*':>3}  pattern")
    for path in sorted(Path(args.sessions_dir).glob("*.jsonl")):
        steps, summary = load(path)
        pts = xstar_points(path.stem, steps)
        all_points += pts
        events = Counter(s["event"] for s in steps)
        stg = (f"{summary.get('stages_passed', '?')}/{summary.get('stages_total', '?')}"
               if "stages_total" in summary else "-")
        print(f"{path.stem:38} {str(summary.get('resolved', '?'))[:4]:4} "
              f"{str(summary.get('stop_reason') or ('v1' if 'harness_version' not in summary else '?')):16} "
              f"{stg:5} {events['test_fail'] + events['test_error']:3d} {events['edit']:3d} {len(pts):3d}  "
              f"{pattern(steps)[:90]}")

    print(f"\n{len(all_points)} X* points from {len({p['task_id'] for p in all_points})} sessions")
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for p in all_points:
        by_cat.setdefault(p["error_category"] or "unknown", []).append(p)
    for cat, pts in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        gen = [p["gen_tokens"] for p in pts if p["gen_tokens"] is not None]
        ctx = [p["ctx_tokens"] for p in pts]
        fixed = sum(p["next_test"] == "test_pass" for p in pts)
        print(f"  {cat:12} n={len(pts):3d}  median gen_tokens={statistics.median(gen) if gen else '-':>6}  "
              f"median ctx_tokens={statistics.median(ctx):>6}  next test passed: {fixed}/{len(pts)}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(all_points, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
