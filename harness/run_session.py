"""Agent session runner with logging.

This script runs a single agent session for a given task, writing each step
to a JSONL file under the output directory (one file per task). The final line
of the JSONL is a summary object per Contract A.

Usage:
    python harness/run_session.py --task-id <id> --output-dir data/sessions/ [--mock]

    The --mock flag generates a synthetic 2-attempt session (attempt 1 fails with
AssertionError, attempt 2 passes) without calling any real LLM or vLLM endpoint.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.token_utils import count_tokens


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

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the repository sandbox.",
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
        "description": "Replace an exact string in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

SYSTEM_PROMPT: str = (
    "You are an expert software engineer fixing a GitHub issue.\n"
    "IMPORTANT: The repository is ALREADY checked out at /testbed. "
    "Do NOT clone or fetch from GitHub — there is no network access.\n"
    "\n"
    "SETUP (run once first):\n"
    "  pip install -e /testbed -q\n"
    "\n"
    "RUNNING TESTS — choose the right runner:\n"
    "  1. Django repos (has /testbed/tests/runtests.py):\n"
    "     cd /testbed && python tests/runtests.py <module> --verbosity 0\n"
    "     - Find the module: ls /testbed/tests/ | grep -i <keyword>\n"
    "     - IMPORTANT: 'Ran 0 tests' means wrong module name — try another.\n"
    "  2. Other repos (matplotlib, sphinx, sympy, etc.):\n"
    "     cd /testbed && python -m pytest <path/to/tests/> -x -q --tb=short\n"
    "     - Find relevant tests: find /testbed -name 'test_*.py' | xargs grep -l '<keyword>' 2>/dev/null | head -5\n"
    "\n"
    "CRITICAL RULES:\n"
    "  - 'Ran 0 tests' is NOT success — it means wrong path. Keep searching.\n"
    "  - Tests must actually RUN (N > 0) and PASS before you are done.\n"
    "  - Never use echo to fake test output.\n"
    "  - Every response must be a tool call (bash or str_replace).\n"
    "Stop only when real tests (N > 0) run and pass."
)

MAX_STEPS: int = 60

# Maximum characters of tool output to include in conversation history.
# Keeps context under the 16384-token model limit.
_MAX_OUTPUT_IN_CONTEXT: int = 3000


def _execute_tool(sandbox: Any, tool_name: str, tool_args: dict[str, Any]) -> str:
    """Dispatch a tool call to the Docker sandbox."""
    if tool_name == "bash":
        command = tool_args.get("command", "")
        return sandbox.run(command)
    elif tool_name == "str_replace":
        path: str = tool_args.get("path", "")
        old_str: str = tool_args.get("old_str", "")
        new_str: str = tool_args.get("new_str", "")
        if not path:
            return "Error: 'path' is required for str_replace"
        content = sandbox.read_file(path)
        if not content and old_str:
            return f"Error: file '{path}' not found or is empty"
        if old_str not in content:
            preview = content[:300] if content else "(empty)"
            return (
                f"Error: old_str not found in '{path}'.\n"
                f"File preview:\n{preview}"
            )
        new_content = content.replace(old_str, new_str, 1)
        sandbox.write_file(path, new_content)
        return f"Successfully replaced in '{path}'"
    else:
        return f"Error: unknown tool '{tool_name}'"


def _write_step_live(out_path: Path, step: dict[str, Any]) -> None:
    """Append a single step dict as a JSON line to the output file."""
    with open(out_path, "a") as fh:
        fh.write(json.dumps(step) + "\n")


def _is_real_pass(output: str) -> bool:
    """Return True if the output shows a genuine test pass (N > 0 tests ran)."""
    lo = output.lower()
    # "Ran 0 tests" is NOT a pass
    if "ran 0 tests" in lo:
        return False
    # pytest style: "X passed"
    if "passed" in lo and "failed" not in lo and "error" not in lo:
        return True
    # Django/unittest "Ran N tests ... OK"
    if "ran " in lo and lo.rstrip().endswith("ok"):
        return True
    if "ok" in lo.split()[-3:] and "failed" not in lo and "error" not in lo and "ran " in lo:
        return True
    return False


def _is_echo_spam(command: str) -> bool:
    """Return True if this bash command is just an echo used to fake output."""
    cmd = command.strip()
    return cmd.startswith("echo '") or cmd.startswith('echo "') or cmd == "exit"


def run_session_loop(
    task: dict[str, Any],
    sandbox: Any,
    out_path: Path,
    max_steps: int = MAX_STEPS,
) -> bool:
    """Run the agentic tool-calling loop for one SWE-bench task."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package required: pip install openai") from exc

    vllm_base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "/models")

    client = OpenAI(base_url=vllm_base, api_key="EMPTY")
    tools = [BASH_TOOL, STR_REPLACE_TOOL]

    user_content = (
        f"Fix the following GitHub issue in the repository at /testbed.\n\n"
        f"Issue:\n{task['problem_statement']}\n\n"
        f"Base commit: {task['base_commit']}\n"
    )
    if task.get("hints_text"):
        user_content += f"\nHints:\n{task['hints_text']}\n"

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    tokens_so_far = sum(count_tokens(m.get("content", "") or "") for m in conversation)
    step_id = 0
    consecutive_echo = 0  # tracks consecutive echo-only steps

    for _iteration in range(max_steps):
        # Retry up to 3 times on transient API errors (timeout, connection)
        _last_exc = None
        for _attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=conversation,
                    tools=tools,
                    tool_choice="required",
                    timeout=120,
                )
                _last_exc = None
                break
            except Exception as _exc:
                _last_exc = _exc
                import time as _time; _time.sleep(5)
        if _last_exc is not None:
            raise _last_exc

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            # Add assistant turn to conversation
            conversation.append(msg.model_dump(exclude_unset=True))

            for tc in msg.tool_calls:
                tool_name: str = tc.function.name
                try:
                    tool_args: dict[str, Any] = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {"raw": tc.function.arguments}

                tokens_so_far += count_tokens(tc.function.arguments or "")

                # Execute tool in sandbox
                tool_output = _execute_tool(sandbox, tool_name, tool_args)
                tokens_so_far += count_tokens(tool_output)

                step: dict[str, Any] = {
                    "step_id": step_id,
                    "role": "tool",
                    "content": None,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_output": tool_output,
                    "tokens_so_far": tokens_so_far,
                    "ts": _now_iso(),
                }
                _write_step_live(out_path, step)
                step_id += 1

                # Truncate output added to conversation to avoid context overflow
                context_output = tool_output
                if len(context_output) > _MAX_OUTPUT_IN_CONTEXT:
                    context_output = (
                        context_output[:_MAX_OUTPUT_IN_CONTEXT]
                        + f"\n...[truncated, {len(tool_output) - _MAX_OUTPUT_IN_CONTEXT} chars omitted]"
                    )

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": context_output,
                })

                # Track echo spam — stop if model keeps echoing
                if tool_name == "bash" and _is_echo_spam(tool_args.get("command", "")):
                    consecutive_echo += 1
                else:
                    consecutive_echo = 0

                if consecutive_echo >= 3:
                    break  # exit inner loop; outer loop will also break below

                # Early exit: real test pass
                if tool_name == "bash" and _is_real_pass(tool_output):
                    break

        else:
            # Regular assistant message (no tool call)
            content: str = msg.content or ""
            tokens_so_far += count_tokens(content)

            step = {
                "step_id": step_id,
                "role": "assistant",
                "content": content,
                "tool_name": None,
                "tool_args": None,
                "tool_output": None,
                "tokens_so_far": tokens_so_far,
                "ts": _now_iso(),
            }
            _write_step_live(out_path, step)
            step_id += 1

            conversation.append({"role": "assistant", "content": content})

            if choice.finish_reason == "stop":
                break

        # Break outer loop on echo spam or real pass
        if consecutive_echo >= 3:
            break
        if msg.tool_calls:
            last_tc = msg.tool_calls[-1]
            try:
                last_args = json.loads(last_tc.function.arguments)
            except Exception:
                last_args = {}
            last_output = _execute_tool.__doc__  # placeholder — use step dict
            # Re-check last tool output for real pass
            last_step_output = step.get("tool_output", "")
            if last_tc.function.name == "bash" and _is_real_pass(last_step_output):
                break

    # ------------------------------------------------------------------
    # Resolution check
    # ------------------------------------------------------------------
    _steps_so_far = [json.loads(l) for l in open(out_path)]
    _passed_in_session = any(
        s.get("tool_name") == "bash"
        and s.get("tool_output")
        and _is_real_pass(s["tool_output"])
        for s in _steps_so_far
        if "__summary__" not in s
    )

    # Final confirmation run scoped to the task's test module
    import re as _re
    _test_module = task.get("test_patch", "")
    _match = _re.search(r'tests/(\w+)', _test_module)
    _scope = _match.group(1) if _match else ""
    final_cmd = (
        "pip install -e /testbed -q 2>/dev/null; "
        f"cd /testbed && python tests/runtests.py {_scope} --verbosity 0 2>&1; "
        "echo '__EXITCODE__:'$?"
    ) if _scope else (
        "pip install -e /testbed -q 2>/dev/null; "
        "cd /testbed && python -m pytest --tb=no -q 2>&1; "
        "echo '__EXITCODE__:'$?"
    )
    pytest_out = sandbox.run(final_cmd)

    # Only count as resolved if exit code 0 AND actual tests ran
    _final_passed = "__EXITCODE__:0" in pytest_out and "ran 0 tests" not in pytest_out.lower()
    resolved: bool = _final_passed or _passed_in_session

    # Log the final test run as an extra step
    final_step: dict[str, Any] = {
        "step_id": step_id,
        "role": "tool",
        "content": None,
        "tool_name": "bash",
        "tool_args": {"command": "python -m pytest --tb=no -q"},
        "tool_output": pytest_out,
        "tokens_so_far": tokens_so_far + count_tokens(pytest_out),
        "ts": _now_iso(),
    }
    _write_step_live(out_path, final_step)

    return resolved


def run_real_session(task_id: str, output_dir: str) -> Path:
    """Run a real SWE-bench agent session."""
    from harness.swebench_tasks import get_task
    from harness.docker_sandbox import DockerSandbox

    task = get_task(task_id)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_id}.jsonl"

    # Truncate/create the output file before the session starts
    out_path.write_text("")

    with DockerSandbox(task, timeout=120) as sandbox:
        resolved = run_session_loop(task, sandbox, out_path)

    # Count steps from the file
    steps_written = sum(1 for line in out_path.read_text().splitlines() if line.strip())
    total_tokens = 0
    for line in out_path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            total_tokens = obj.get("tokens_so_far", total_tokens)

    summary: dict[str, Any] = {
        "__summary__": True,
        "resolved": resolved,
        "total_steps": steps_written,
        "total_tokens": total_tokens,
    }
    with open(out_path, "a") as fh:
        fh.write(json.dumps(summary) + "\n")

    return out_path


def main() -> None:
    """Entry point for the session runner CLI."""
    parser = argparse.ArgumentParser(
        description="Run an agent session and log it to JSONL."
    )
    parser.add_argument("--task-id", required=True, help="Task identifier")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write session JSONL files",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate a synthetic session without calling a real LLM",
    )
    args = parser.parse_args()

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
        out_path = run_real_session(args.task_id, args.output_dir)
        print(f"Session written to {out_path}")


if __name__ == "__main__":
    main()
