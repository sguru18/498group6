"""Agent session runner with logging.

This script runs a single agent session for a given task, writing each step
to a JSONL file under the output directory (one file per task). The final line
of the JSONL is a summary object per Contract A.

Usage:
    python harness/run_session.py --task-id <id> --output-dir data/sessions/ [--mock]
    python harness/run_session.py --task-id <id> --check-setup   # no LLM: verify the baseline fails

The --mock flag generates a synthetic 2-attempt session (attempt 1 fails with
AssertionError, attempt 2 passes) without calling any real LLM or vLLM endpoint.

Session structure (real runs)
-----------------------------
The harness applies the SWE-bench test_patch so the issue's FAIL_TO_PASS tests
genuinely fail, then walks the agent through *stages*:

    stage 1..k   one (group of) FAIL_TO_PASS test(s) at a time
    last stage   regression: FAIL_TO_PASS + PASS_TO_PASS in the touched test files

The agent's `run_tests` tool always runs the current stage. When a stage
passes, the harness immediately runs the next one and shows the agent its new
failure, so a session produces repeated fail -> edit -> pass -> next-fail arcs.

Every step carries an `event` label (test_fail, test_pass, test_error, edit,
edit_failed, blocked, explore, reasoning, ...) plus parsed error types, so
X* analysis does not have to re-derive them from raw text.
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import test_runner
from harness.token_utils import count_tokens

HARNESS_VERSION = "2"


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_step(
    step_id: int,
    role: str,
    content: str | None,
    tool_name: str | None,
    tool_args: dict[str, Any] | None,
    tool_output: str | None,
    tokens_so_far: int,
) -> dict[str, Any]:
    """Build a step dict conforming to Contract A."""
    return {
        "step_id": step_id,
        "role": role,
        "content": content,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_output": tool_output,
        "tokens_so_far": tokens_so_far,
        "ts": _now_iso(),
    }


def generate_mock_session(task_id: str) -> list[dict[str, Any]]:
    """Generate a synthetic 2-attempt session without calling any LLM.

    Attempt 0: steps 0-4, bash test at step 4 → AssertionError (fail)
    Attempt 1: steps 5-9, bash test at step 9 → 3 passed (pass)

    Args:
        task_id: The task identifier (used in messages for realism).

    Returns:
        A list of step dicts (no summary line — caller appends that).
    """
    steps: list[dict[str, Any]] = []
    tokens = 0

    # --- Attempt 0: fails with AssertionError ---
    messages_0 = [
        ("assistant", f"Starting task {task_id}. Let me read the failing test."),
        ("tool", None, "bash", {"command": f"cat tests/test_{task_id}.py"},
         f"def test_add():\n    assert add(1, 2) == 42  # expected 42"),
        ("assistant", "I see. The function returns the wrong value. Let me fix it."),
        ("tool", None, "bash", {"command": "sed -i 's/return x/return x + y/' src/solution.py"},
         ""),
        ("tool", None, "bash", {"command": "pytest tests/ -x"},
         "FAILED tests/test_solution.py::test_add - AssertionError: assert 3 == 42"),
    ]

    for item in messages_0:
        role = item[0]
        if role == "assistant":
            text = item[1]
            tokens += count_tokens(text)
            steps.append(_make_step(len(steps), "assistant", text, None, None, None, tokens))
        else:
            _, _, tname, targs, tout = item
            tokens += count_tokens(json.dumps(targs) + (tout or ""))
            steps.append(_make_step(len(steps), "tool", None, tname, targs, tout, tokens))

    # --- Attempt 1: passes ---
    messages_1 = [
        ("assistant", "The fix was incomplete. Let me look at the actual implementation."),
        ("tool", None, "bash", {"command": "cat src/solution.py"},
         "def add(x, y):\n    return x + y"),
        ("assistant", "The implementation looks correct now. The test expectation was wrong. Fixing test."),
        ("tool", None, "bash", {"command": "sed -i 's/== 42/== 3/' tests/test_solution.py"},
         ""),
        ("tool", None, "bash", {"command": "pytest tests/ -x"},
         "3 passed in 0.12s"),
    ]

    for item in messages_1:
        role = item[0]
        if role == "assistant":
            text = item[1]
            tokens += count_tokens(text)
            steps.append(_make_step(len(steps), "assistant", text, None, None, None, tokens))
        else:
            _, _, tname, targs, tout = item
            tokens += count_tokens(json.dumps(targs) + (tout or ""))
            steps.append(_make_step(len(steps), "tool", None, tname, targs, tout, tokens))

    return steps


def write_session(
    steps: list[dict[str, Any]],
    task_id: str,
    output_dir: str,
    resolved: bool,
) -> Path:
    """Write steps + summary to a JSONL file in output_dir."""
    out_path = Path(output_dir) / f"{task_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = steps[-1]["tokens_so_far"] if steps else 0
    summary = {
        "__summary__": True,
        "resolved": resolved,
        "total_steps": len(steps),
        "total_tokens": total_tokens,
    }

    with open(out_path, "w") as f:
        for step in steps:
            f.write(json.dumps(step) + "\n")
        f.write(json.dumps(summary) + "\n")

    return out_path


# ---------------------------------------------------------------------------
# Tool definitions for the real agent loop
# ---------------------------------------------------------------------------

RUN_TESTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": (
            "Run the target tests for the current stage of this task and report "
            "which tests fail, with tracebacks. Call this before your first edit "
            "and after every edit."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a bash command in /testbed to inspect code, e.g. "
            "`grep -rn 'def name' /testbed/pkg` or `sed -n '100,140p' /testbed/pkg/mod.py`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run",
                }
            },
            "required": ["command"],
        },
    },
}

STR_REPLACE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "str_replace",
        "description": (
            "Edit a source file by replacing one exact, unique occurrence of old_str with new_str. "
            "old_str must match the file character-for-character, including indentation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path, e.g. /testbed/pkg/mod.py"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

TOOLS = [RUN_TESTS_TOOL, BASH_TOOL, STR_REPLACE_TOOL]
_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM_PROMPT: str = """You are an expert software engineer fixing a GitHub issue in the repository at /testbed.
The Python environment is already installed and activated. There is no network access.

TOOLS
- run_tests: runs the target tests for the current stage and shows failures with tracebacks.
- bash: inspect the code, e.g. `grep -rn "def foo" /testbed/pkg` or `sed -n '120,160p' /testbed/pkg/mod.py`.
- str_replace: edit a SOURCE file. old_str must match the file exactly (including indentation) and be unique.

WORKFLOW — repeat until the harness says the task is complete:
  1. Call run_tests and read the failure: test name, exception type, file and line, expected vs actual.
  2. Find the responsible source code with bash (grep -n, sed -n). Read the failing test too if needed.
  3. Make ONE focused str_replace edit to the source code.
  4. Call run_tests again. If it still fails, read the NEW output carefully and go back to step 2.
When a stage passes, the harness gives you the next target (or reports tests your change broke). Keep going.

RULES
- Before every tool call, write 1-3 sentences: what the last output told you and what you will do next.
- Make exactly ONE tool call per message, through the tool-call interface.
- Never modify test files. The tests define the correct behaviour; fix the library code.
- Do not re-run tests without changing code first; the result will be identical.
- Do not pip install anything, do not run git commands, and do not use echo to report progress.
- Only stop when the harness tells you all stages pass.
"""

MAX_STEPS: int = 45

# Context-window budget (vLLM --max-model-len 16384).
_MODEL_CONTEXT: int = 16384
_MAX_COMPLETION_TOKENS: int = 1536
_TOOLS_OVERHEAD_TOKENS: int = 700
_CONTEXT_BUDGET: int = _MODEL_CONTEXT - _MAX_COMPLETION_TOKENS - 500

# Characters of a single tool output put into the conversation.
_MAX_OUTPUT_IN_CONTEXT: int = 3500
_MAX_PROBLEM_CHARS: int = 6000
_MAX_STAGES: int = 4
_TEST_TIMEOUT: int = 900
_BASH_TIMEOUT: int = 120
_MAX_CONSECUTIVE_BLOCKED: int = 6
_MAX_NUDGES: int = 3

_BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*echo\b[^|>;&]*$"),
     "echo does not do anything useful here. Inspect code with bash, edit with str_replace, "
     "and check with run_tests."),
    (re.compile(r"(^|[;&|]\s*)git\s+(add|commit|checkout|stash|reset|clean|restore|apply|am|"
                r"cherry-pick|rebase|merge|push|pull|fetch|clone|switch)\b"),
     "git commands that change repository state are disabled; nothing needs to be committed. "
     "Use `git diff` to review your changes if needed."),
    (re.compile(r"(^|[;&|]\s*)(python\s+-m\s+)?pip3?\s+install\b"),
     "The environment is already fully installed; pip install is disabled."),
    (re.compile(r"^\s*exit\b"), "Do not exit; keep working until the harness says the task is complete."),
]


def _blocked_reason(command: str) -> str | None:
    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def _is_test_path(path: str) -> bool:
    rel = path.removeprefix("/testbed/")
    parts = rel.split("/")
    name = parts[-1]
    return (
        any(p in ("tests", "testing") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _numbered(lines: list[str], start: int, end: int) -> str:
    start, end = max(0, start), min(len(lines), end)
    return "\n".join(f"{i + 1:6d}\t{lines[i]}" for i in range(start, end))


def _locate_near_miss(content: str, old_str: str) -> str:
    """Explain why old_str didn't match: indentation-only mismatch, or show the closest region."""
    file_lines = content.splitlines()
    old_lines = [l for l in old_str.splitlines()]
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    if not old_lines:
        return "old_str is empty."

    stripped_old = [l.strip() for l in old_lines]
    n = len(stripped_old)
    for i in range(len(file_lines) - n + 1):
        if [l.strip() for l in file_lines[i:i + n]] == stripped_old:
            return (
                f"old_str matches lines {i + 1}-{i + n} except for whitespace/indentation. "
                f"The exact text in the file is:\n{_numbered(file_lines, i, i + n)}\n"
                "Copy the indentation exactly (the line-number prefix is not part of the file)."
            )

    anchor = max(stripped_old, key=len)
    best_i, best_ratio = 0, 0.0
    for i, line in enumerate(file_lines):
        ratio = difflib.SequenceMatcher(None, line.strip(), anchor).ratio()
        if ratio > best_ratio:
            best_i, best_ratio = i, ratio
    if best_ratio < 0.5:
        return "No similar text found in the file. Use grep -n to find the right location."
    return (
        f"Closest match is around line {best_i + 1}:\n"
        f"{_numbered(file_lines, best_i - 6, best_i + n + 6)}"
    )


_INLINE_CALL_START = re.compile(r'\{\s*"name"\s*:')


def _parse_inline_tool_calls(text: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Recover tool calls the model wrote into its text instead of the tool-call channel.

    Qwen often emits bare `{"name": ..., "arguments": ...}` objects (optionally inside
    <tool_call> tags or code fences). Returns (calls in textual order, text without them).
    """
    decoder = json.JSONDecoder()
    calls: list[tuple[str, dict[str, Any]]] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    while (m := _INLINE_CALL_START.search(text, pos)):
        try:
            obj, end = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            pos = m.end()
            continue
        pos = end
        name = obj.get("name") if isinstance(obj, dict) else None
        args = obj.get("arguments", obj.get("parameters", {})) if name else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if name in _TOOL_NAMES and isinstance(args, dict):
            calls.append((name, args))
            spans.append((m.start(), end))
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    text = re.sub(r"</?tool_call>|```(?:json)?\s*```", "", text)
    return calls, text.strip()


def _clip_for_context(text: str, limit: int = _MAX_OUTPUT_IN_CONTEXT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} chars omitted]...\n{text[-tail:]}"


def _is_context_overflow(exc: Exception) -> bool:
    s = str(exc)
    return "maximum context length" in s or "input_tokens" in s or "too long" in s.lower()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def build_stages(task: dict[str, Any], baseline: dict[str, str]) -> list[dict[str, Any]]:
    """Split FAIL_TO_PASS into up to _MAX_STAGES target stages, then a regression stage.

    Each target stage also re-checks the targets of earlier stages, so a later
    edit that breaks an earlier fix shows up immediately.
    """
    f2p = task["fail_to_pass"]
    p2p_ok = [t for t in task["pass_to_pass"] if baseline.get(t) in test_runner.PASSING]
    n_groups = min(len(f2p), _MAX_STAGES)
    groups = [f2p[i::n_groups] for i in range(n_groups)] if n_groups else []

    stages: list[dict[str, Any]] = []
    done: list[str] = []
    for i, group in enumerate(groups):
        stages.append({
            "kind": "target",
            "targets": group,
            "required": list(done),
            "selector": done + group,
        })
        done = done + group
    if p2p_ok or len(stages) > 1:
        stages.append({"kind": "regression", "targets": f2p, "required": p2p_ok, "selector": None})
    for i, st in enumerate(stages):
        st["index"] = i
    return stages


def _stage_title(stage: dict[str, Any], n: int) -> str:
    i = stage["index"] + 1
    if stage["kind"] == "regression":
        return (f"Stage {i}/{n} (regression): all target tests plus "
                f"{len(stage['required'])} previously-passing tests in the same test files must pass")
    names = ", ".join(stage["targets"][:5]) + (" ..." if len(stage["targets"]) > 5 else "")
    return f"Stage {i}/{n}: make these tests pass: {names}"


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

class SessionRunner:
    """Runs one agent session against a prepared DockerSandbox and logs every step."""

    def __init__(
        self,
        task: dict[str, Any],
        sandbox: Any,
        out_path: Path,
        stages: list[dict[str, Any]],
        max_steps: int = MAX_STEPS,
        use_hints: bool = False,
    ) -> None:
        self.task = task
        self.sb = sandbox
        self.out_path = out_path
        self.stages = stages
        self.stage_idx = 0
        self.max_steps = max_steps

        self.step_id = 0
        self.tokens_so_far = 0
        self.gen_tokens_so_far = 0
        self.tok_ratio = 1.15  # Qwen tokens per cl100k token; recalibrated from usage
        self.pruned_messages = 0

        self.fingerprint = sandbox.code_fingerprint()
        self.last_test_fp: str | None = None
        self.last_test_result: dict[str, Any] | None = None
        self.last_call: tuple[str, str] | None = None
        self.last_call_fp: str | None = None
        self.last_call_output: str = ""
        self.consecutive_blocked = 0
        self.tests_run = 0
        self.resolved = False
        self.stop_reason = "max_steps"

        self.head = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._initial_user_message(use_hints)},
        ]
        self.body: list[dict[str, Any]] = []
        self.tokens_so_far = sum(count_tokens(m["content"]) for m in self.head)

    # -- conversation ---------------------------------------------------

    def _initial_user_message(self, use_hints: bool) -> str:
        t = self.task
        problem = t["problem_statement"]
        if len(problem) > _MAX_PROBLEM_CHARS:
            problem = problem[:_MAX_PROBLEM_CHARS] + "\n...[issue text truncated]"
        tests = "\n".join(f"  - {x}" for x in t["fail_to_pass"][:10])
        files = "\n".join(f"  - /testbed/{f}" for f in t["test_files"])
        msg = (
            f"Fix the following GitHub issue in the {t['repo']} repository at /testbed.\n\n"
            f"<issue>\n{problem}\n</issue>\n\n"
            f"These tests reproduce the issue and currently fail:\n{tests}\n"
            f"They live in:\n{files}\n\n"
            f"The task has {len(self.stages)} stage(s). "
            f"{_stage_title(self.stages[0], len(self.stages))}.\n"
            "Start by calling run_tests."
        )
        if use_hints and t.get("hints_text"):
            msg += f"\n\nHints from the issue discussion:\n{t['hints_text'][:2000]}"
        return msg

    def _messages(self) -> list[dict[str, Any]]:
        msgs = list(self.head)
        if self.pruned_messages:
            msgs.append({
                "role": "user",
                "content": (f"[harness] {self.pruned_messages} older messages were removed to fit "
                            f"the context window. Current task: "
                            f"{_stage_title(self.stages[self.stage_idx], len(self.stages))}."),
            })
        return msgs + self.body

    def _estimate(self, msgs: list[dict[str, Any]]) -> int:
        raw = _TOOLS_OVERHEAD_TOKENS
        for m in msgs:
            raw += 4 + count_tokens(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                raw += count_tokens(tc["function"]["arguments"]) + 10
        return int(raw * self.tok_ratio)

    def _prune(self, budget: int) -> None:
        """Elide old tool outputs, then drop whole oldest exchanges, until under budget."""
        if self._estimate(self._messages()) <= budget:
            return
        groups: list[list[dict[str, Any]]] = []
        for m in self.body:
            if m["role"] == "tool" and groups:
                groups[-1].append(m)
            else:
                groups.append([m])

        for g in groups[:-3]:
            for m in g:
                c = m.get("content") or ""
                if m["role"] in ("tool", "user") and len(c) > 600:
                    m["content"] = f"{c[:200]}\n...[old output elided]...\n{c[-300:]}"
        self.body = [m for g in groups for m in g]

        while len(groups) > 2 and self._estimate(self._messages()) > budget:
            self.pruned_messages += len(groups.pop(0))
            self.body = [m for g in groups for m in g]
        print(f"[context] pruned to ~{self._estimate(self._messages())} tokens "
              f"({self.pruned_messages} messages dropped so far)")

    def _chat(self, client: Any, model: str, force_required: bool) -> Any:
        tool_choice = "required" if force_required or os.environ.get("TOOL_CHOICE") == "required" else "auto"
        extra: dict[str, Any] = {}
        if os.environ.get("TEMPERATURE"):
            extra["temperature"] = float(os.environ["TEMPERATURE"])
        last_exc: Exception | None = None
        for attempt in range(4):
            self._prune(int(_CONTEXT_BUDGET * (0.8 ** attempt)))
            msgs = self._messages()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                    max_tokens=_MAX_COMPLETION_TOKENS,
                    timeout=180,
                    **extra,
                )
                if resp.usage and resp.usage.prompt_tokens:
                    raw = self._estimate(msgs) / self.tok_ratio
                    self.tok_ratio = max(0.8, min(2.0, resp.usage.prompt_tokens / raw))
                return resp
            except Exception as exc:
                last_exc = exc
                if _is_context_overflow(exc):
                    continue
                time.sleep(5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    # -- logging --------------------------------------------------------

    def _log(self, role: str, *, content: str | None = None, tool_name: str | None = None,
             tool_args: dict[str, Any] | None = None, tool_output: str | None = None,
             **extra: Any) -> dict[str, Any]:
        self.tokens_so_far += count_tokens(content or "") + count_tokens(tool_output or "")
        if tool_args is not None and extra.get("source", "agent") == "agent":
            self.tokens_so_far += count_tokens(json.dumps(tool_args))
        self.gen_tokens_so_far += extra.pop("completion_tokens", 0) or 0
        step = _make_step(self.step_id, role, content, tool_name, tool_args, tool_output,
                          self.tokens_so_far)
        step.update({"stage": self.stage_idx, "gen_tokens_so_far": self.gen_tokens_so_far})
        step.update(extra)
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(step) + "\n")
        self.step_id += 1
        return step

    # -- tests ----------------------------------------------------------

    def run_stage(self, stage: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        """Run a stage's tests. Returns (graded result, command, raw output)."""
        selector = stage["selector"] if stage.get("selector_ok", True) else None
        cmd = test_runner.build_command(self.task, selector)
        code, out = self.sb.run_with_code(cmd, timeout=_TEST_TIMEOUT, max_chars=60000)
        timed_out = code is None and out.startswith("[TIMEOUT")
        result = test_runner.evaluate(self.task, out, stage["targets"], stage["required"], timed_out)
        if (selector is not None and not timed_out
                and len(result["missing_targets"]) == len(stage["targets"])):
            # The per-test selector didn't select anything; fall back to whole test files.
            stage["selector_ok"] = False
            cmd = test_runner.build_command(self.task, None)
            code, out = self.sb.run_with_code(cmd, timeout=_TEST_TIMEOUT, max_chars=60000)
            timed_out = code is None and out.startswith("[TIMEOUT")
            result = test_runner.evaluate(self.task, out, stage["targets"], stage["required"], timed_out)
        self.tests_run += 1
        return result, cmd, out

    @staticmethod
    def _test_fields(result: dict[str, Any], cmd: str, raw: str) -> dict[str, Any]:
        return {
            "event": f"test_{result['status']}",
            "test_status": result["status"],
            "n_passed": result["n_passed"],
            "n_failed": result["n_failed"],
            "failing_tests": result["failing_tests"],
            "error_types": result["error_types"],
            "error_category": result["error_category"],
            "test_command": cmd,
            "raw_output": raw[-12000:],
        }

    def _tool_run_tests(self) -> tuple[str, dict[str, Any]]:
        stage = self.stages[self.stage_idx]
        if self.last_test_fp == self.fingerprint and self.last_test_result is not None \
                and self.last_test_result["status"] != "pass":
            r = self.last_test_result
            return (
                "[BLOCKED] No source code has changed since your last run_tests, so the result "
                f"would be identical ({r['n_failed']} failing: {', '.join(r['failing_tests'][:3])}; "
                f"errors: {', '.join(r['error_types']) or 'none'}).\n"
                f"{self._repo_state()}\n"
                f"The failure you still need to fix:\n{_clip_for_context(r['failure_text'].strip(), 1500)}\n"
                "Re-read it, find the source code responsible, and change it with str_replace.",
                {"event": "blocked", "blocked_reason": "repeat_test_without_edit"},
            )
        result, cmd, raw = self.run_stage(stage)
        self.last_test_fp = self.fingerprint
        self.last_test_result = result
        header = f"[run_tests] {_stage_title(stage, len(self.stages))}\n$ {cmd}"
        text = test_runner.format_for_agent(result, raw, header)
        fields = self._test_fields(result, cmd, raw)
        fields["tool_args_effective"] = {"command": cmd}
        return text, fields

    # -- bash / edits ---------------------------------------------------

    def _repo_state(self) -> str:
        stat = self.sb.run("git diff HEAD --stat", timeout=30).strip()
        if not stat:
            return "Current state: you have NOT changed any source file yet."
        return f"Current state: your changes so far (git diff --stat):\n{stat}"

    def _refresh_fingerprint(self) -> tuple[bool, list[str]]:
        fp = self.sb.code_fingerprint()
        changed = fp != self.fingerprint
        self.fingerprint = fp
        return changed, (self.sb.changed_files() if changed else [])

    def _revert_test_edits(self, files: list[str]) -> list[str]:
        touched = [f for f in files if _is_test_path(f)]
        if touched:
            self.sb.run("git checkout HEAD -- " + " ".join(f"'{f}'" for f in touched))
            self.fingerprint = self.sb.code_fingerprint()
        return touched

    def _tool_bash(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        command = str(args.get("command", ""))
        reason = _blocked_reason(command)
        if reason:
            return f"[BLOCKED] {reason}", {"event": "blocked", "blocked_reason": "disallowed_command"}

        output = self.sb.run(command, timeout=_BASH_TIMEOUT, max_chars=20000)
        changed, files = self._refresh_fingerprint()
        reverted = self._revert_test_edits(files) if changed else []
        if reverted:
            output += (f"\n[harness] Reverted your changes to test files ({', '.join(reverted)}). "
                       "Fix the library code instead.")
        cls = test_runner.classify_bash_output(output)
        fields: dict[str, Any] = {"error_types": cls["error_types"],
                                  "error_category": cls["error_category"]}
        src_files = [f for f in files if f not in reverted]
        if src_files:
            fields.update(event="edit", edit_via="bash", edit_files=src_files)
        elif cls["kind"] in ("test_pass", "test_fail"):
            fields["event"] = cls["kind"]
        elif cls["kind"] == "error":
            fields["event"] = "bash_error"
        else:
            fields["event"] = "explore"
        return _clip_for_context(output), fields

    def _tool_str_replace(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        path = str(args.get("path", "")).strip()
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        if path and not path.startswith("/"):
            path = "/testbed/" + path
        if self.tests_run == 0:
            return ("[BLOCKED] Call run_tests first and read the failure before editing.",
                    {"event": "blocked", "blocked_reason": "edit_before_test"})
        if not path or not isinstance(old_str, str) or not isinstance(new_str, str):
            return "Error: str_replace needs string 'path', 'old_str' and 'new_str'.", {"event": "edit_failed"}
        if _is_test_path(path):
            return (f"[BLOCKED] {path} is a test file. Tests define the expected behaviour; "
                    "edit the library source instead.",
                    {"event": "blocked", "blocked_reason": "edit_test_file"})
        if self.sb.run(f"test -f '{path}' && echo yes").strip() != "yes":
            return f"Error: file not found: {path}", {"event": "edit_failed", "edit_error": "not_found"}
        if old_str == new_str:
            return "Error: old_str and new_str are identical.", {"event": "edit_failed", "edit_error": "noop"}

        content = self.sb.read_file(path)
        count = content.count(old_str) if old_str else 0
        if count == 0:
            return (f"Error: old_str not found in {path}.\n{_locate_near_miss(content, old_str)}",
                    {"event": "edit_failed", "edit_error": "no_match"})
        if count > 1:
            lines = [content[:m.start()].count("\n") + 1 for m in re.finditer(re.escape(old_str), content)]
            return (f"Error: old_str occurs {count} times in {path} (lines {lines[:10]}). "
                    "Include more surrounding lines so it is unique.",
                    {"event": "edit_failed", "edit_error": "ambiguous"})

        start = content.index(old_str)
        self.sb.write_file(path, content.replace(old_str, new_str, 1))
        changed, files = self._refresh_fingerprint()

        new_lines = content[:start].count("\n")
        lines = self.sb.read_file(path).splitlines()
        span = new_str.count("\n") + 1
        msg = (f"Edited {path}. Lines {new_lines + 1}-{new_lines + span} now read:\n"
               f"{_numbered(lines, new_lines - 3, new_lines + span + 3)}")
        fields: dict[str, Any] = {"event": "edit", "edit_via": "str_replace",
                                  "edit_files": files or [path.removeprefix('/testbed/')]}
        if path.endswith(".py"):
            compile_out = self.sb.run(f"python -m py_compile '{path}' 2>&1", timeout=60).strip()
            if compile_out:
                types = test_runner.extract_error_types(compile_out)
                fields.update(syntax_error=True, error_types=types,
                              error_category=test_runner.error_category(types))
                msg += f"\n\nWARNING: the file no longer compiles:\n{compile_out[-1500:]}"
        return msg, fields

    # -- main loop ------------------------------------------------------

    def _dispatch(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        key = (name, json.dumps(args, sort_keys=True))
        if name != "run_tests" and key == self.last_call and self.last_call_fp == self.fingerprint:
            return (
                "[BLOCKED] You just made this exact call and nothing has changed since. Its output was:\n"
                f"{self.last_call_output[:800]}\n{self._repo_state()}\nDo something different.",
                {"event": "blocked", "blocked_reason": "identical_repeat"},
            )
        if name == "run_tests":
            text, fields = self._tool_run_tests()
        elif name == "bash":
            text, fields = self._tool_bash(args)
        elif name == "str_replace":
            text, fields = self._tool_str_replace(args)
        else:
            text, fields = f"Error: unknown tool '{name}'.", {"event": "invalid_tool"}
        if fields.get("event") != "blocked":
            self.last_call, self.last_call_fp, self.last_call_output = key, self.fingerprint, text
        return text, fields

    def _advance_stages(self, result: dict[str, Any]) -> None:
        """Called after the current stage passed: move on, auto-running the next stage."""
        while result["status"] == "pass":
            passed = self.stages[self.stage_idx]
            if self.stage_idx + 1 >= len(self.stages):
                self.resolved = True
                self.stop_reason = "resolved"
                return
            self.stage_idx += 1
            stage = self.stages[self.stage_idx]
            result, cmd, raw = self.run_stage(stage)
            self.last_test_fp = self.fingerprint
            self.last_test_result = result
            header = f"[run_tests] {_stage_title(stage, len(self.stages))}\n$ {cmd}"
            text = (f"[harness] Stage {passed['index'] + 1}/{len(self.stages)} passed. "
                    f"Moving to stage {stage['index'] + 1}. I ran its tests for you:\n\n"
                    + test_runner.format_for_agent(result, raw, header))
            if result["status"] != "pass":
                text += "\n\nContinue: find the cause, edit the source, and call run_tests."
            self._log("tool", tool_name="run_tests", tool_args={"command": cmd},
                      tool_output=text, source="harness", **self._test_fields(result, cmd, raw))
            self.body.append({"role": "user", "content": text})

    def run(self) -> None:
        from openai import OpenAI

        client = OpenAI(base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                        api_key="EMPTY")
        model = os.environ.get("VLLM_MODEL", "/models")
        nudges = 0
        force_required = False

        for _ in range(self.max_steps):
            try:
                resp = self._chat(client, model, force_required)
            except Exception as exc:
                if _is_context_overflow(exc):
                    self.stop_reason = "context_overflow"
                    return
                raise
            force_required = False
            choice = resp.choices[0]
            msg = choice.message
            usage = resp.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else 0
            content = (msg.content or "").strip()

            calls: list[tuple[str, str, dict[str, Any]]] = []
            for tc in msg.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                calls.append((tc.id, tc.function.name, args if isinstance(args, dict) else {"raw": args}))
            if content:
                inline, content = _parse_inline_tool_calls(content)
                calls = [(f"call_{uuid.uuid4().hex[:12]}", n, a) for n, a in inline] + calls
                # Qwen sometimes keeps generating fake follow-up turns after its call.
                content = content.split("<|im_start|>")[0].strip()
            # Execute only the model's first decision; later calls in the same response are
            # usually hallucinated repeats generated without seeing any tool output.
            calls = calls[:1]

            if content:
                self._log("assistant", content=content, event="reasoning",
                          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                completion_tokens = 0

            if not calls:
                self.body.append({"role": "assistant", "content": content})
                nudges += 1
                if nudges > _MAX_NUDGES:
                    self.stop_reason = "model_stopped"
                    return
                nudge = ("[harness] The task is not complete: "
                         f"{_stage_title(self.stages[self.stage_idx], len(self.stages))}. "
                         "Continue by calling a tool (run_tests, bash, or str_replace).")
                self.body.append({"role": "user", "content": nudge})
                force_required = True
                continue
            nudges = 0

            self.body.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [{"id": cid, "type": "function",
                                "function": {"name": n, "arguments": json.dumps(a)}}
                               for cid, n, a in calls],
            })
            stage_passed: dict[str, Any] | None = None
            for cid, name, args in calls:
                if stage_passed is not None:
                    text, fields = "[skipped: the stage already passed]", {"event": "skipped"}
                else:
                    text, fields = self._dispatch(name, args)
                self._log("tool", tool_name=name, tool_args=args, tool_output=text, source="agent",
                          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, **fields)
                completion_tokens = 0
                self.body.append({"role": "tool", "tool_call_id": cid, "content": text})

                if fields.get("event") == "blocked":
                    self.consecutive_blocked += 1
                else:
                    self.consecutive_blocked = 0
                if name == "run_tests" and fields.get("test_status") == "pass":
                    stage_passed = self.last_test_result

            if stage_passed is not None:
                self._advance_stages(stage_passed)
                if self.resolved:
                    return
            if self.consecutive_blocked >= _MAX_CONSECUTIVE_BLOCKED:
                self.stop_reason = "stuck"
                return

    def final_eval(self) -> dict[str, Any]:
        """Official-style check: all FAIL_TO_PASS pass and no baseline-passing PASS_TO_PASS regress."""
        regression = self.stages[-1]
        stage = {"targets": self.task["fail_to_pass"], "required": regression["required"],
                 "selector": None}
        result, cmd, raw = self.run_stage(stage)
        self._log("tool", tool_name="run_tests", tool_args={"command": cmd},
                  tool_output=test_runner.format_for_agent(result, raw, "[final_eval]"),
                  source="harness", **{**self._test_fields(result, cmd, raw), "event": "final_eval"})
        return result


# ---------------------------------------------------------------------------
# Setup + entry points
# ---------------------------------------------------------------------------

def prepare_sandbox(task: dict[str, Any], sandbox: Any) -> dict[str, Any]:
    """Apply the test patch, commit it, and run the baseline. Returns baseline info."""
    apply_out = sandbox.apply_patch(task["test_patch"])
    sandbox.commit_all("SWE-bench test patch")
    cmd = test_runner.build_command(task, None)
    code, out = sandbox.run_with_code(cmd, timeout=_TEST_TIMEOUT, max_chars=60000)
    statuses = test_runner.parse_statuses(task["repo"], out)
    f2p = task["fail_to_pass"]
    f2p_found = [t for t in f2p if t in statuses]
    f2p_failing = [t for t in f2p_found if statuses[t] not in test_runner.PASSING]
    crashed = not f2p_found
    valid = crashed or bool(f2p_failing)
    result = test_runner.evaluate(task, out, f2p, [], code is None and out.startswith("[TIMEOUT"))
    return {
        "valid": valid,
        "crashed": crashed,
        "patch_applied": ("Applied patch" in apply_out or "patching file" in apply_out)
                         and "rejected" not in apply_out,
        "command": cmd,
        "statuses": statuses,
        "f2p_total": len(f2p),
        "f2p_failing": len(f2p_failing),
        "f2p_missing": len(f2p) - len(f2p_found),
        "p2p_passing": sum(1 for t in task["pass_to_pass"] if statuses.get(t) in test_runner.PASSING),
        "p2p_total": len(task["pass_to_pass"]),
        "error_types": result["error_types"],
        "summary_line": result["summary_line"],
        "raw_tail": out[-3000:],
    }


def run_real_session(
    task_id: str,
    output_dir: str,
    max_steps: int = MAX_STEPS,
    use_hints: bool = False,
) -> Path:
    """Run a real SWE-bench agent session. Always writes a summary line, even on crash."""
    from harness.swebench_tasks import get_task
    from harness.docker_sandbox import DockerSandbox

    task = get_task(task_id)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_id}.jsonl"
    out_path.write_text("")

    summary: dict[str, Any] = {
        "__summary__": True,
        "resolved": False,
        "harness_version": HARNESS_VERSION,
        "model": os.environ.get("VLLM_MODEL", "/models"),
        "repo": task["repo"],
        "stop_reason": None,
    }
    runner: SessionRunner | None = None
    error: str | None = None
    try:
        with DockerSandbox(task, timeout=_BASH_TIMEOUT) as sandbox:
            baseline = prepare_sandbox(task, sandbox)
            summary["baseline"] = {k: v for k, v in baseline.items() if k not in ("statuses", "raw_tail")}
            if not baseline["valid"]:
                summary["stop_reason"] = "invalid_setup"
                print(f"[setup] {task_id}: FAIL_TO_PASS tests already pass at baseline; skipping.")
            else:
                stages = build_stages(task, baseline["statuses"])
                runner = SessionRunner(task, sandbox, out_path, stages, max_steps, use_hints)
                try:
                    runner.run()
                finally:
                    final = runner.final_eval()
                    summary["resolved"] = final["status"] == "pass"
                    summary["final_eval"] = {k: final[k] for k in
                                             ("status", "n_passed", "n_failed", "failing_tests")}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        steps = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        events = [s.get("event") for s in steps]
        summary.update({
            "total_steps": len(steps),
            "total_tokens": steps[-1]["tokens_so_far"] if steps else 0,
            "gen_tokens": steps[-1].get("gen_tokens_so_far", 0) if steps else 0,
            "stages_total": len(runner.stages) if runner else 0,
            "stages_passed": (runner.stage_idx + int(runner.resolved)) if runner else 0,
            "n_test_fail": sum(e in ("test_fail", "test_error") for e in events),
            "n_test_pass": events.count("test_pass"),
            "n_edit": events.count("edit"),
            "n_edit_failed": events.count("edit_failed"),
            "n_blocked": events.count("blocked"),
        })
        if runner and not summary["stop_reason"]:
            summary["stop_reason"] = runner.stop_reason
        if error:
            summary["stop_reason"] = "error"
            summary["error"] = error
        with open(out_path, "a") as fh:
            fh.write(json.dumps(summary) + "\n")

    return out_path


def check_setup(task_id: str) -> dict[str, Any]:
    """Prepare the sandbox and run the baseline without calling the LLM."""
    from harness.swebench_tasks import get_task
    from harness.docker_sandbox import DockerSandbox

    task = get_task(task_id)
    with DockerSandbox(task, timeout=_BASH_TIMEOUT) as sandbox:
        baseline = prepare_sandbox(task, sandbox)
    stages = build_stages(task, baseline["statuses"]) if baseline["valid"] else []
    baseline["stages"] = [{"kind": s["kind"], "targets": s["targets"],
                           "n_required": len(s["required"])} for s in stages]
    return baseline


def main() -> None:
    """Entry point for the session runner CLI."""
    parser = argparse.ArgumentParser(
        description="Run an agent session and log it to JSONL."
    )
    parser.add_argument("--task-id", required=True, help="Task identifier")
    parser.add_argument(
        "--output-dir",
        default="data/sessions/",
        help="Directory to write session JSONL files",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate a synthetic session without calling a real LLM",
    )
    parser.add_argument("--check-setup", action="store_true",
                        help="Only apply the test patch and run the baseline tests (no LLM)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--hints", action="store_true",
                        help="Include SWE-bench hints_text (often contains the fix)")
    args = parser.parse_args()

    if args.check_setup:
        info = check_setup(args.task_id)
        print(json.dumps({k: v for k, v in info.items() if k != "statuses"}, indent=2))
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mock:
        steps = generate_mock_session(args.task_id)
        resolved = False
        for step in reversed(steps):
            if step.get("tool_name") == "bash" and step.get("tool_output"):
                out = step["tool_output"].lower()
                if "passed" in out or "ok" in out or "0 failed" in out:
                    resolved = True
                break

        out_path = write_session(steps, args.task_id, args.output_dir, resolved)
        print(f"Mock session written to {out_path} ({len(steps)} steps, resolved={resolved})")
    else:
        out_path = run_real_session(args.task_id, args.output_dir, args.max_steps, args.hints)
        print(f"Session written to {out_path}")


if __name__ == "__main__":
    main()
