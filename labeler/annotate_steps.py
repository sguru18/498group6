"""Annotate each tool-call step in session JSONL files with error metadata.

Adds two fields to every bash tool-call step:
  - "is_error": bool   — True if the tool output contains an error signal
  - "error_type": str | None — one of "syntax", "import", "assertion",
                               "runtime", "unknown", or None (no error)

Rewrites each JSONL file in-place (or writes to --output-dir if given).

Usage:
    python labeler/annotate_steps.py --sessions-dir data/sessions/
    python labeler/annotate_steps.py --sessions-dir data/sessions/ --output-dir data/annotated/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from labeler.error_taxonomy import classify_error

def _is_error(tool_output: str) -> bool:
    """An error is detected if classify_error returns a non-None stratum."""
    return classify_error(tool_output) is not None


def annotate_session(path: Path, out_path: Path) -> int:
    """Annotate all steps in one session file. Returns number of steps annotated."""
    lines = path.read_text().splitlines()
    annotated = []
    count = 0

    for line in lines:
        if not line.strip():
            continue
        step = json.loads(line)

        # Only annotate bash tool calls; leave summary + other steps untouched
        if step.get("tool_name") == "bash":
            output = step.get("tool_output") or ""
            error = _is_error(output)
            step["is_error"] = error
            step["error_type"] = classify_error(output) if error else None
            count += 1
        else:
            # Non-bash steps: still add the fields as null for schema consistency
            if "__summary__" not in step:
                step["is_error"] = False
                step["error_type"] = None

        annotated.append(json.dumps(step))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(annotated) + "\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate session JSONL steps with error labels.")
    parser.add_argument("--sessions-dir", required=True, help="Directory containing session JSONL files.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: overwrite in-place).")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    files = sorted(sessions_dir.glob("*.jsonl"))

    if not files:
        print(f"No .jsonl files found in {sessions_dir}")
        sys.exit(1)

    for f in files:
        if f.name == "batch_summary.json":
            continue
        out = Path(args.output_dir) / f.name if args.output_dir else f
        n = annotate_session(f, out)
        print(f"  {f.name}: annotated {n} bash steps → {out}")

    print(f"\nDone. {len(files)} session(s) annotated.")


if __name__ == "__main__":
    main()
