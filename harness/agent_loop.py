"""Minimal OpenAI-compatible tool-calling loop for the agent harness.

This module implements the core agentic loop that communicates with an
OpenAI-compatible endpoint (e.g. vLLM). It accepts messages and tool
definitions, calls the endpoint, and yields step dicts matching Contract A.

The endpoint base URL is configured via the VLLM_BASE_URL environment
variable (default: http://localhost:8000/v1).
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Generator

from harness.token_utils import count_tokens

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_agent_loop(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_steps: int = 50,
    model: str = DEFAULT_MODEL,
) -> Generator[dict[str, Any], None, None]:
    """Run an OpenAI-compatible tool-calling loop and yield step dicts.

    Each yielded dict conforms to Contract A step schema. The loop continues
    until the model stops calling tools, a test passes, or max_steps is reached.

    Args:
        messages: Initial conversation messages (system + user).
        tools: List of tool definitions in OpenAI function-calling format.
        max_steps: Maximum number of steps before stopping.
        model: Model name to use for the endpoint call.

    Yields:
        Step dicts conforming to Contract A schema.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package required for agent_loop") from e

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    conversation = list(messages)
    tokens_so_far = sum(count_tokens(m.get("content", "") or "") for m in messages)
    step_id = 0

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {"raw": tc.function.arguments}

                tokens_so_far += count_tokens(tc.function.arguments or "")

                step = {
                    "step_id": step_id,
                    "role": "tool",
                    "content": None,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_output": None,  # caller fills this in
                    "tokens_so_far": tokens_so_far,
                    "ts": _now_iso(),
                }
                yield step
                step_id += 1

                # Add tool call + result to conversation
                conversation.append(msg.model_dump(exclude_unset=True))
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": step.get("tool_output") or "",
                })

        else:
            # Regular assistant message
            content = msg.content or ""
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
            yield step
            step_id += 1

            conversation.append({"role": "assistant", "content": content})

            if choice.finish_reason == "stop":
                break
