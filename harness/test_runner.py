"""Per-repo test commands, test-output parsing, and error classification.

The harness runs the official SWE-bench reproduction tests (FAIL_TO_PASS) and
regression tests (PASS_TO_PASS) itself, so every failure the agent sees is a
real, parseable test result. This module knows how to:

  * build the test command for a task (Django runner, sympy bin/test, pytest)
  * parse per-test statuses out of the runner output
  * extract the failure sections and exception types for the agent / the logs
  * classify an arbitrary bash output as a test result, error, or neither
"""

import re
import shlex
from typing import Any

# Statuses that count as "not failing" for resolution purposes.
PASSING = {"PASSED", "SKIPPED", "XFAIL"}

_DJANGO_CMD = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
_SYMPY_CMD = "PYTHONWARNINGS=ignore bin/test -C --verbose"
_PYTEST_CMD = "python -m pytest -rA --tb=short -p no:cacheprovider -W ignore::DeprecationWarning"


# ---------------------------------------------------------------------------
# Test selection
# ---------------------------------------------------------------------------

def test_files_from_patch(test_patch: str) -> list[str]:
    """Return the repo-relative paths touched by a SWE-bench test_patch."""
    files: list[str] = []
    for line in test_patch.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m and m.group(2) not in files:
            files.append(m.group(2))
    return files


def _is_test_module(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.endswith(".py") and (name.startswith("test") or name == "tests.py")


def _django_label_for_file(path: str) -> str | None:
    if not path.startswith("tests/") or not _is_test_module(path):
        return None
    return path[len("tests/"):-len(".py")].replace("/", ".")


def _django_label_for_test(test_id: str) -> str | None:
    """'test_x (app.tests.Cls)' -> 'app.tests.Cls.test_x'. Docstring-named tests -> None."""
    m = re.match(r"^(\w+) \(([\w.]+)\)$", test_id.strip())
    if not m:
        return None
    name, where = m.groups()
    return where if where.endswith("." + name) else f"{where}.{name}"


def _sympy_name(test_id: str) -> str | None:
    return test_id if re.fullmatch(r"test_\w+", test_id) else None


def build_command(task: dict[str, Any], test_ids: list[str] | None) -> str:
    """Build the shell command that runs `test_ids` (or all test files if None).

    Falls back to running the whole test files when an individual test id
    can't be turned into a runner selector.
    """
    repo = task["repo"]
    files = [f for f in task.get("test_files", []) if _is_test_module(f)]

    if repo == "django/django":
        labels = [_django_label_for_test(t) for t in test_ids] if test_ids else []
        if not test_ids or any(l is None for l in labels):
            labels = [l for l in (_django_label_for_file(f) for f in files) if l]
        return f"{_DJANGO_CMD} {' '.join(labels)}"

    if repo == "sympy/sympy":
        cmd = f"{_SYMPY_CMD} {' '.join(files)}"
        if test_ids and len(test_ids) == 1 and _sympy_name(test_ids[0]):
            cmd += f" -k {test_ids[0]}"
        return cmd

    if not test_ids:
        # Repos like pytest name test modules without a test_ prefix (testing/python/fixtures.py),
        # so take the files from the test ids themselves.
        ids = task.get("fail_to_pass", []) + task.get("pass_to_pass", [])
        from_ids = list(dict.fromkeys(t.split("::")[0] for t in ids if "::" in t))
        files = from_ids or [f for f in task.get("test_files", []) if f.endswith(".py")]
    selectors = list(test_ids) if test_ids else files
    return f"{_PYTEST_CMD} {' '.join(shlex.quote(s) for s in selectors)}"


# ---------------------------------------------------------------------------
# Parsing runner output
# ---------------------------------------------------------------------------

_DJANGO_STATUS = {
    "ok": "PASSED", "FAIL": "FAILED", "ERROR": "ERROR",
    "skipped": "SKIPPED", "expected failure": "XFAIL", "unexpected success": "FAILED",
}
_DJANGO_LINE = re.compile(
    r"^(?P<test>\w+ \([\w.]+\))(?:\n(?P<doc>.*?))? \.\.\. "
    r"(?P<st>ok|FAIL|ERROR|skipped|expected failure|unexpected success)",
    re.MULTILINE,
)
_PYTEST_LINE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS) (\S.*?)(?: - .*)?$", re.MULTILINE)
_SYMPY_LINE = re.compile(r"^(test_\w+) (ok|F|E|f|s|Skipped|X)\b", re.MULTILINE)
_SYMPY_STATUS = {"ok": "PASSED", "F": "FAILED", "E": "ERROR", "f": "XFAIL",
                 "s": "SKIPPED", "Skipped": "SKIPPED", "X": "FAILED"}


def parse_statuses(repo: str, output: str) -> dict[str, str]:
    """Map test id (in SWE-bench FAIL_TO_PASS format) -> PASSED/FAILED/ERROR/SKIPPED/XFAIL."""
    statuses: dict[str, str] = {}
    if repo == "django/django":
        for m in _DJANGO_LINE.finditer(output):
            st = _DJANGO_STATUS[m.group("st")]
            statuses[m.group("test")] = st
            if m.group("doc"):
                statuses[m.group("doc").strip()] = st
        # Subtest failures print no inline status, only a 'FAIL: test_x (app.tests.Cls) (param=..)' header.
        for m in re.finditer(r"^(FAIL|ERROR): (\w+ \([\w.]+\))", output, re.MULTILINE):
            statuses[m.group(2)] = "FAILED" if m.group(1) == "FAIL" else "ERROR"
        # Python 3.11+ unittest prints 'test_x (app.tests.Cls.test_x)'; alias the short form too
        for tid, st in list(statuses.items()):
            m = re.match(r"^(\w+) \(([\w.]+)\.(\w+)\)$", tid)
            if m and m.group(1) == m.group(3):
                statuses.setdefault(f"{m.group(1)} ({m.group(2)})", st)
    elif repo == "sympy/sympy":
        for name, st in _SYMPY_LINE.findall(output):
            statuses[name] = _SYMPY_STATUS[st]
    else:
        for st, tid in _PYTEST_LINE.findall(output):
            statuses[tid.strip()] = "FAILED" if st == "XPASS" else st
    return statuses


_EXC_LINE = re.compile(
    r"^(?:E\s+)?(?:[a-z_][\w]*\.)*"
    r"([A-Z]\w*(?:Error|Exception|Exit|Interrupt|Iteration|Configured|DoesNotExist|Warning|Failure))"
    r"(?::|$)",
    re.MULTILINE,
)


def extract_error_types(text: str) -> list[str]:
    """Unique exception class names appearing as exception lines, in order."""
    seen: list[str] = []
    for m in _EXC_LINE.finditer(text):
        name = m.group(1)
        if name not in seen and not name.endswith("Warning"):
            seen.append(name)
    return seen


_ENV_ERRORS = {"ImportError", "ModuleNotFoundError", "FileNotFoundError", "ImproperlyConfigured"}
_SYNTAX_ERRORS = {"SyntaxError", "IndentationError", "TabError"}


def error_category(error_types: list[str], timed_out: bool = False) -> str | None:
    """Collapse exception types onto the labeler's four strata (see labeler/error_taxonomy.py)."""
    if timed_out:
        return "timeout"
    if not error_types:
        return None
    s = set(error_types)
    if s & _SYNTAX_ERRORS:
        return "syntax"
    if s & _ENV_ERRORS:
        return "environment"
    if s - {"AssertionError"}:
        return "runtime"
    return "logic"


def extract_failure_sections(repo: str, output: str) -> str:
    """Return just the failure/traceback blocks from runner output (no setup noise)."""
    if repo == "django/django":
        m = re.search(r"^={20,}\n(?:FAIL|ERROR):", output, re.MULTILINE)
        if not m:
            return _bare_traceback(output)
        end = re.search(r"^-{20,}\nRan \d+ tests?", output[m.start():], re.MULTILINE)
        return output[m.start(): m.start() + end.start()] if end else output[m.start():]
    if repo == "sympy/sympy":
        m = re.search(r"^_{20,}$", output, re.MULTILINE)
        if not m:
            return _bare_traceback(output)
        end = re.search(r"^=+ tests finished", output[m.start():], re.MULTILINE)
        return output[m.start(): m.start() + end.start()] if end else output[m.start():]
    m = re.search(r"^=+ (?:FAILURES|ERRORS) =+$", output, re.MULTILINE)
    if not m:
        return _bare_traceback(output)
    end = re.search(r"^=+ (?:short test summary info|PASSES|warnings summary) =+$",
                    output[m.end():], re.MULTILINE)
    return output[m.start(): m.end() + end.start()] if end else output[m.start():]


def _bare_traceback(output: str) -> str:
    """Tracebacks that happen before any test runs (import errors, collection errors)."""
    i = output.find("Traceback (most recent call last)")
    return output[i:] if i >= 0 else ""


def runner_summary_line(repo: str, output: str) -> str:
    if repo == "django/django":
        m = re.search(r"^Ran \d+ tests?.*\n\n(?:OK|FAILED).*$", output, re.MULTILINE)
    elif repo == "sympy/sympy":
        m = re.search(r"tests finished:.*?(?= =|$)", output, re.MULTILINE)
    else:
        m = re.search(r"^=+ (.*(?:passed|failed|error|no tests ran).*) =+$", output, re.MULTILINE)
    return " ".join(m.group(0).split()).strip("= ") if m else ""


def _trim(text: str, limit: int) -> str:
    """Keep the head and (more of) the tail: tracebacks end with the actual exception."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} chars omitted]...\n{text[-tail:]}"


# ---------------------------------------------------------------------------
# High-level result objects
# ---------------------------------------------------------------------------

def evaluate(
    task: dict[str, Any],
    output: str,
    targets: list[str],
    required_passing: list[str] | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    """Grade a test run against `targets` (must pass) and `required_passing` (must not regress).

    Returns a dict with status in {"pass", "fail", "error"}:
      pass  — every target and required test that appears in the output passes
      fail  — at least one of them fails/errors
      error — none of the targets appear (crash before tests ran, collection error, timeout)
    """
    repo = task["repo"]
    statuses = parse_statuses(repo, output)
    must = list(dict.fromkeys(targets + (required_passing or [])))
    seen = {t: statuses[t] for t in must if t in statuses}
    failing = [t for t, st in seen.items() if st not in PASSING]
    targets_seen = [t for t in targets if t in statuses]

    if timed_out or not targets_seen:
        status = "error"
    elif failing:
        status = "fail"
    else:
        status = "pass"

    failure_text = extract_failure_sections(repo, output)
    error_types = extract_error_types(failure_text or output) if status != "pass" else []
    if status == "fail" and not error_types:
        error_types = ["AssertionError"] if any(seen[t] == "FAILED" for t in failing) else []
    return {
        "status": status,
        "n_passed": sum(1 for st in seen.values() if st in PASSING),
        "n_failed": len(failing),
        "failing_tests": failing[:10],
        "missing_targets": [t for t in targets if t not in statuses][:10],
        "error_types": error_types,
        "error_category": error_category(error_types, timed_out) if status != "pass" else None,
        "summary_line": runner_summary_line(repo, output),
        "failure_text": failure_text,
    }


def format_for_agent(result: dict[str, Any], output: str, header: str, limit: int = 3500) -> str:
    """Condensed, model-facing rendering of a graded test run."""
    lines = [header]
    if result["status"] == "pass":
        lines.append(f"RESULT: PASS ({result['n_passed']} passed). {result['summary_line']}")
        return "\n".join(lines)
    if result["status"] == "error":
        lines.append("RESULT: ERROR — the target tests did not run to completion "
                     "(crash during import/collection, or timeout).")
    else:
        lines.append(f"RESULT: FAIL — {result['n_failed']} failing, {result['n_passed']} passing. "
                     f"{result['summary_line']}")
        lines.append("Failing tests:\n" + "\n".join(f"  - {t}" for t in result["failing_tests"]))
    body = result["failure_text"] or output
    lines.append("\n" + _trim(body.strip(), limit))
    return "\n".join(lines)


_GENERIC_SUMMARY = re.compile(
    r"(^Ran \d+ tests?)|(=+ .*(passed|failed|error).* =+$)|(tests finished:)", re.MULTILINE
)


def classify_bash_output(output: str) -> dict[str, Any]:
    """Classify the output of an arbitrary agent bash command.

    Returns {"kind": "test_pass"|"test_fail"|"error"|"none", "error_types", "error_category"}.
    """
    if not output:
        return {"kind": "none", "error_types": [], "error_category": None}
    timed_out = output.startswith("[TIMEOUT")
    error_types = extract_error_types(output)
    lo = output.lower()
    looks_like_tests = bool(_GENERIC_SUMMARY.search(output))
    if looks_like_tests and not re.search(r"ran 0 tests|no tests ran|collected 0 items", lo):
        failed = bool(re.search(r"\bfailed\b|\bfailures=|\berrors?=|\d+ errors?\b|, \d+ failed", lo))
        if failed:
            return {"kind": "test_fail", "error_types": error_types or ["AssertionError"],
                    "error_category": error_category(error_types or ["AssertionError"])}
        return {"kind": "test_pass", "error_types": [], "error_category": None}
    if timed_out or "Traceback (most recent call last)" in output or error_types:
        return {"kind": "error", "error_types": error_types,
                "error_category": error_category(error_types, timed_out)}
    return {"kind": "none", "error_types": [], "error_category": None}
