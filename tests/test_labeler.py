"""Tests for the labeler module (segment.py and error_taxonomy.py).

Validates Contract B compliance using the three synthetic fixture files.
"""

import json
from pathlib import Path

import pytest

from labeler.segment import label_session, load_session, segment_attempts
from labeler.error_taxonomy import classify_error

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIXTURE_RESOLVED = FIXTURES_DIR / "fixture_resolved.jsonl"
FIXTURE_TWO_FAILS = FIXTURES_DIR / "fixture_two_fails_then_pass.jsonl"
FIXTURE_ALL_FAILS = FIXTURES_DIR / "fixture_all_fails.jsonl"

ALL_FIXTURES = [FIXTURE_RESOLVED, FIXTURE_TWO_FAILS, FIXTURE_ALL_FAILS]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _label(path: Path) -> dict:
    return label_session(path)


# ---------------------------------------------------------------------------
# Test 1: fixture_resolved
# ---------------------------------------------------------------------------

def test_fixture_resolved() -> None:
    """fixture_resolved → 1 attempt, outcome=pass, error_type=None, failed_fraction=0.0"""
    result = _label(FIXTURE_RESOLVED)

    attempts = result["attempts"]
    assert len(attempts) == 1, f"Expected 1 attempt, got {len(attempts)}"

    attempt = attempts[0]
    assert attempt["outcome"] == "pass", f"Expected pass, got {attempt['outcome']}"
    assert attempt["error_type"] is None, f"Expected error_type=None, got {attempt['error_type']}"

    assert result["failed_token_fraction"] == 0.0, (
        f"Expected failed_token_fraction=0.0, got {result['failed_token_fraction']}"
    )


# ---------------------------------------------------------------------------
# Test 2: fixture_two_fails_then_pass
# ---------------------------------------------------------------------------

def test_fixture_two_fails_then_pass() -> None:
    """fixture_two_fails_then_pass → 3 attempts, correct outcomes and error_types."""
    result = _label(FIXTURE_TWO_FAILS)

    attempts = result["attempts"]
    assert len(attempts) == 3, f"Expected 3 attempts, got {len(attempts)}"

    # Outcomes
    assert attempts[0]["outcome"] == "fail"
    assert attempts[1]["outcome"] == "fail"
    assert attempts[2]["outcome"] == "pass"

    # Error types
    assert attempts[0]["error_type"] == "assertion", (
        f"Expected assertion, got {attempts[0]['error_type']}"
    )
    assert attempts[1]["error_type"] == "runtime", (
        f"Expected runtime, got {attempts[1]['error_type']}"
    )
    assert attempts[2]["error_type"] is None, (
        f"Expected None for passing attempt, got {attempts[2]['error_type']}"
    )

    # failed_token_fraction
    total_tokens = result["total_tokens"]
    expected_fraction = (attempts[0]["token_span"] + attempts[1]["token_span"]) / total_tokens
    assert abs(result["failed_token_fraction"] - round(expected_fraction, 4)) < 1e-6, (
        f"Expected {round(expected_fraction, 4)}, got {result['failed_token_fraction']}"
    )


# ---------------------------------------------------------------------------
# Test 3: fixture_all_fails
# ---------------------------------------------------------------------------

def test_fixture_all_fails() -> None:
    """fixture_all_fails → 2 attempts, both fail, fraction=1.0, resolved=False."""
    result = _label(FIXTURE_ALL_FAILS)

    attempts = result["attempts"]
    assert len(attempts) == 2, f"Expected 2 attempts, got {len(attempts)}"

    assert attempts[0]["outcome"] == "fail"
    assert attempts[1]["outcome"] == "fail"

    assert attempts[0]["error_type"] == "syntax", (
        f"Expected syntax, got {attempts[0]['error_type']}"
    )
    assert attempts[1]["error_type"] == "import", (
        f"Expected import, got {attempts[1]['error_type']}"
    )

    assert result["failed_token_fraction"] == 1.0, (
        f"Expected failed_token_fraction=1.0, got {result['failed_token_fraction']}"
    )

    assert result["resolved"] is False


# ---------------------------------------------------------------------------
# Test 4: failed_token_fraction bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_path", ALL_FIXTURES)
def test_failed_token_fraction_bounds(fixture_path: Path) -> None:
    """For all fixtures, 0.0 <= failed_token_fraction <= 1.0."""
    result = _label(fixture_path)
    ftf = result["failed_token_fraction"]
    assert 0.0 <= ftf <= 1.0, (
        f"{fixture_path.name}: failed_token_fraction={ftf} out of bounds"
    )


# ---------------------------------------------------------------------------
# Test 5: attempts are contiguous
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_path", ALL_FIXTURES)
def test_attempts_are_contiguous(fixture_path: Path) -> None:
    """For all fixtures, attempts cover steps 0..N-1 with no gaps or overlaps."""
    steps, summary = load_session(fixture_path)
    total_steps = len(steps)
    result = _label(fixture_path)
    attempts = result["attempts"]

    assert len(attempts) > 0, "No attempts found"

    # First attempt starts at step 0
    assert attempts[0]["start_step"] == 0, (
        f"First attempt should start at step 0, got {attempts[0]['start_step']}"
    )

    # Last attempt ends at total_steps - 1
    assert attempts[-1]["end_step"] == total_steps - 1, (
        f"Last attempt should end at step {total_steps - 1}, "
        f"got {attempts[-1]['end_step']}"
    )

    # Contiguous: each attempt starts immediately after the previous ends
    for i in range(1, len(attempts)):
        expected_start = attempts[i - 1]["end_step"] + 1
        actual_start = attempts[i]["start_step"]
        assert actual_start == expected_start, (
            f"{fixture_path.name}: attempt {i} starts at {actual_start}, "
            f"expected {expected_start} (gap or overlap)"
        )


# ---------------------------------------------------------------------------
# Test 6: token_span consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_path", ALL_FIXTURES)
def test_token_span_consistency(fixture_path: Path) -> None:
    """For all fixtures, token_span == token_end - token_start for every attempt."""
    result = _label(fixture_path)
    for attempt in result["attempts"]:
        expected = attempt["token_end"] - attempt["token_start"]
        assert attempt["token_span"] == expected, (
            f"{fixture_path.name} attempt {attempt['attempt_id']}: "
            f"token_span={attempt['token_span']} != "
            f"token_end-token_start={expected}"
        )
