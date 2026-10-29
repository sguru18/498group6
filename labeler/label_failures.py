"""CLI: reads a sessions directory and outputs a labeled_sessions.json file.

For each .jsonl file in the sessions directory, this script labels the session
by segmenting it into attempts and computing failure metrics per Contract B.

Usage:
    python labeler/label_failures.py --sessions-dir data/sessions/ \
        --output data/labeled/labeled_sessions.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.segment import label_session


def main() -> None:
    """Entry point for the label_failures CLI."""
    parser = argparse.ArgumentParser(
        description="Label all session JSONL files in a directory."
    )
    parser.add_argument(
        "--sessions-dir",
        required=True,
        help="Directory containing .jsonl session files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the labeled_sessions.json file",
    )
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    jsonl_files = sorted(sessions_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {sessions_dir}", file=sys.stderr)
        sys.exit(1)

    labeled: list[dict] = []
    for jsonl_path in jsonl_files:
        try:
            result = label_session(jsonl_path)
            labeled.append(result)
            print(f"  Labeled {jsonl_path.name}: {len(result['attempts'])} attempts, "
                  f"resolved={result['resolved']}, "
                  f"failed_fraction={result['failed_token_fraction']:.4f}")
        except Exception as e:
            print(f"  ERROR labeling {jsonl_path.name}: {e}", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(labeled, f, indent=2)

    print(f"\nLabeled {len(labeled)} sessions → {out_path}")


if __name__ == "__main__":
    main()
