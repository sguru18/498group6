"""Attempt segmentation logic for agent session logs.

Reads a parsed session (list of step dicts from a JSONL file) and segments it
into attempts. An attempt boundary is detected at each bash test-runner call.
The outcome of each attempt is determined by scanning the tool_output for
pass/fail signals.

Conforms to Contract B (Labeled Sessions Schema).
"""

import json
from pathlib import Path
from typing import Any

from labeler.error_taxonomy import classify_error

# Test runner keywords that signal a test execution step
TEST_RUNNER_KEYWORDS = [
    "pytest",
    "python -m pytest",
    "make test",
    "npm test",
    "yarn test",
    "go test",
    "cargo test",
    "mvn test",
    "gradlew test",
    "rspec",
    "jest",
    "mocha",
    "nosetests",
    "unittest",
    "runtests.py",
    "bin/test",
]

# Signals that indicate test failure (case-insensitive)
FAILURE_SIGNALS = [
    "FAILED",
    "ERROR",
    "error:",
    "Traceback",
    "AssertionError",
    "SyntaxError",
    "ImportError",
    "ModuleNotFoundError",
    "Exception",
]

# Signals that indicate test pass (case-insensitive)
PASS_SIGNALS = [
    "passed",
    "ok",
    "PASSED",
    "0 failed",
    "0 errors",
]


def _is_test_run_step(step: dict[str, Any]) -> bool:
    """Return True if the step is a bash test-runner call.

    Args:
        step: A step dict from the session log.

    Returns:
        True if this step triggered a test run.
    """
    if step.get("test_status"):
        return True
    if step.get("tool_name") != "bash":
        return False
    cmd = ""
    if step.get("tool_args"):
        cmd = step["tool_args"].get("command", "")
    return any(kw in cmd for kw in TEST_RUNNER_KEYWORDS)


def _determine_outcome(tool_output: str | None) -> str:
    """Determine pass/fail outcome from test runner output.

    If both failure and pass signals are present, failure takes precedence.
    If neither is found, returns "unknown".

    Args:
        tool_output: The stdout/stderr string from the test runner.

    Returns:
        One of "pass", "fail", or "unknown".
    """
    if not tool_output:
        return "unknown"

    has_fail = any(sig in tool_output for sig in FAILURE_SIGNALS)
    has_pass = any(sig.lower() in tool_output.lower() for sig in PASS_SIGNALS)

    if has_fail:
        return "fail"
    if has_pass:
        return "pass"
    return "unknown"


def load_session(jsonl_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a session JSONL file into steps and summary.

    Args:
        jsonl_path: Path to the .jsonl session file.

    Returns:
        A tuple of (steps, summary) where steps is the list of step dicts
        and summary is the final __summary__ line dict.

    Raises:
        ValueError: If the file has no summary line or is malformed.
    """
    steps: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("__summary__"):
                summary = obj
            else:
                steps.append(obj)

    if not summary:
        raise ValueError(f"No __summary__ line found in {jsonl_path}")

    return steps, summary


def segment_attempts(
    steps: list[dict[str, Any]],
    total_tokens: int,
) -> list[dict[str, Any]]:
    """Segment a session's steps into attempts.

    An attempt spans from the step after the previous test boundary (or step 0)
    to the step where a test run command is executed. The last attempt ends at
    the final step regardless of whether a test run occurred.

    Attempts are non-overlapping, contiguous, and cover all steps 0..N-1.

    Args:
        steps: List of step dicts from the session.
        total_tokens: Total tokens for the session (from summary).

    Returns:
        List of attempt dicts conforming to Contract B attempt schema.
    """
    if not steps:
        return []

    # Find the step indices where test runs occur
    test_run_indices: list[int] = [
        i for i, s in enumerate(steps) if _is_test_run_step(s)
    ]

    # Build attempt boundaries: each test run step ends an attempt
    # If no test run found, the whole session is one attempt
    boundaries: list[int] = test_run_indices if test_run_indices else [len(steps) - 1]

    attempts: list[dict[str, Any]] = []
    attempt_start = 0

    for boundary_idx, end_step_idx in enumerate(boundaries):
        attempt_end = end_step_idx
        test_step = steps[end_step_idx]
        tool_output = test_step.get("tool_output") or ""
        # Harness v2 steps carry a graded test_status; prefer it over text heuristics.
        structured = test_step.get("test_status")
        outcome = ({"pass": "pass", "fail": "fail", "error": "fail"}[structured]
                   if structured else _determine_outcome(tool_output))

        # Token span: from start of attempt to end of attempt
        token_start = steps[attempt_start]["tokens_so_far"] - _step_token_count(steps[attempt_start])
        # token_start is the cumulative tokens BEFORE this attempt's first step
        # We compute it as tokens_so_far of the previous step (or 0 for first attempt)
        if attempt_start == 0:
            token_start = 0
        else:
            token_start = steps[attempt_start - 1]["tokens_so_far"]

        token_end = steps[attempt_end]["tokens_so_far"]
        token_span = token_end - token_start

        error_type: str | None = None
        if outcome == "fail" and test_step.get("error_category"):
            error_type = test_step["error_category"]
        elif outcome == "fail":
            error_type = classify_error(tool_output)
        elif outcome == "pass":
            error_type = None
        else:
            # unknown outcome — still try to classify
            error_type = classify_error(tool_output) if tool_output else "unknown"

        # Extract a short snippet of the error for the label
        error_snippet: str | None = None
        if tool_output and outcome != "pass":
            lines = [l.strip() for l in tool_output.splitlines() if l.strip()]
            error_snippet = lines[0] if lines else None
            if error_snippet and len(error_snippet) > 200:
                error_snippet = error_snippet[:200]

        attempts.append({
            "attempt_id": len(attempts),
            "start_step": attempt_start,
            "end_step": attempt_end,
            "token_start": token_start,
            "token_end": token_end,
            "token_span": token_span,
            "outcome": outcome,
            "error_type": error_type,
            "error_snippet": error_snippet,
        })

        attempt_start = attempt_end + 1

    # If there are steps after the last test run, add a final attempt
    if attempt_start <= len(steps) - 1:
        last_step = steps[-1]
        if attempt_start == 0:
            token_start = 0
        else:
            token_start = steps[attempt_start - 1]["tokens_so_far"]
        token_end = last_step["tokens_so_far"]
        token_span = token_end - token_start

        attempts.append({
            "attempt_id": len(attempts),
            "start_step": attempt_start,
            "end_step": len(steps) - 1,
            "token_start": token_start,
            "token_end": token_end,
            "token_span": token_span,
            "outcome": "unknown",
            "error_type": "unknown",
            "error_snippet": None,
        })

    return attempts


def _step_token_count(step: dict[str, Any]) -> int:
    """Estimate tokens contributed by a single step (not used in main path)."""
    return 0


def label_session(jsonl_path: str | Path) -> dict[str, Any]:
    """Label a single session JSONL file.

    Loads the session, segments it into attempts, and computes the
    failed_token_fraction. Returns a labeled session dict per Contract B.

    Args:
        jsonl_path: Path to the session JSONL file.

    Returns:
        A labeled session dict conforming to Contract B.
    """
    path = Path(jsonl_path)
    task_id = path.stem
    steps, summary = load_session(path)
    total_tokens = summary.get("total_tokens", 0)
    resolved = summary.get("resolved", False)

    attempts = segment_attempts(steps, total_tokens)

    failed_token_span = sum(
        a["token_span"] for a in attempts if a["outcome"] == "fail"
    )
    failed_token_fraction = (
        failed_token_span / total_tokens if total_tokens > 0 else 0.0
    )

    return {
        "task_id": task_id,
        "resolved": resolved,
        "total_tokens": total_tokens,
        "attempts": attempts,
        "failed_token_fraction": round(failed_token_fraction, 4),
    }
