"""SWE-bench Lite task loader using the HuggingFace datasets library.

Provides helpers to load and sample tasks from the princeton-nlp/SWE-bench_Lite
dataset. The dataset split used is "test" (the standard evaluation split).
"""

import random
from typing import Optional

_DATASET: Optional[object] = None  # cached after first load


def _get_dataset() -> object:
    """Load (and cache) the SWE-bench Lite test split from HuggingFace."""
    global _DATASET
    if _DATASET is None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets package required: pip install datasets"
            ) from e
        _DATASET = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return _DATASET


def _row_to_task(row: dict) -> dict:
    """Convert a dataset row to the canonical task dict format."""
    return {
        "task_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "problem_statement": row["problem_statement"],
        "test_patch": row.get("test_patch", ""),
        "hints_text": row.get("hints_text") or None,
    }


def load_lite_tasks(n: int = 20, seed: int = 42) -> list[dict]:
    """Load n tasks from SWE-bench Lite, deterministically sampled.

    Uses a seeded RNG so the same (n, seed) pair always returns the same tasks.

    Args:
        n: Number of tasks to return.
        seed: Random seed for reproducible sampling.

    Returns:
        List of task dicts, each containing:
            - task_id: str  (e.g. 'django__django-11049')
            - repo: str     (e.g. 'django/django')
            - base_commit: str
            - problem_statement: str
            - test_patch: str
            - hints_text: str | None
    """
    ds = _get_dataset()
    total = len(ds)
    rng = random.Random(seed)
    indices = rng.sample(range(total), min(n, total))
    return [_row_to_task(ds[i]) for i in sorted(indices)]


def get_task(task_id: str) -> dict:
    """Fetch a single task by ID from SWE-bench Lite.

    Args:
        task_id: The instance_id to look up (e.g. 'django__django-11049').

    Returns:
        Task dict with task_id, repo, base_commit, problem_statement,
        test_patch, hints_text.

    Raises:
        KeyError: If the task_id is not found in SWE-bench Lite.
    """
    ds = _get_dataset()
    for row in ds:
        if row["instance_id"] == task_id:
            return _row_to_task(row)
    raise KeyError(f"Task '{task_id}' not found in SWE-bench Lite")
