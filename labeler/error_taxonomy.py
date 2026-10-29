"""Error type classification for bash tool-call output.

Four strata following the execution pipeline, ordered by how much prior-attempt
context the model needs to recover. This motivates the X* hypothesis:
X* increases from environment → syntax → runtime → logic.

  environment — file not found, module not installed, command not found
                Fully self-contained error message. (Lowest X*)

  syntax      — SyntaxError, Cython/gcc compile failure, parse error
                Code is malformed; error pinpoints the exact line. (Low X*)

  runtime     — Traceback, TypeError, RuntimeError, NameError etc.
                Code compiled but crashed; traceback is partially self-contained. (Medium X*)

  logic       — FAILED tests, AssertionError, wrong output
                Code ran cleanly but produced wrong result; model needs prior
                context to understand what was wrong. (Highest X*)
"""

# Priority order: environment > syntax > runtime > logic
_ENVIRONMENT_SIGNALS = [
    "No such file or directory",
    "No module named",
    "ModuleNotFoundError",
    "ImportError",
    "command not found",
    "can't open file",
    "cannot find",
    "[Errno 2]",
    "[TIMEOUT",
]

_SYNTAX_SIGNALS = [
    "SyntaxError",
    "SyntaxWarning",
    "IndentationError",
    "TabError",
    "error: command",       # gcc/Cython compile error
    "compilation failed",
    "error generated",
    "Build failed",
    ".pyx:",                # Cython source line error
]

_RUNTIME_SIGNALS = [
    "Traceback (most recent call last)",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "AttributeError",
    "NameError",
    "KeyError",
    "IndexError",
    "ZeroDivisionError",
    "RecursionError",
    "Exception",
    "Error:",
]

_LOGIC_SIGNALS = [
    "FAILED",
    "AssertionError",
    "assert ",
    " != ",
    " == False",
    "Expected",
    "expected",
    "but got",
    "wrong answer",
]


def classify_error(tool_output: str) -> str | None:
    """Classify the error stratum from a bash tool call's output.

    Priority order (first match wins):
      1. environment — missing file/module/command       (lowest X*)
      2. syntax      — SyntaxError / compile failure     (low X*)
      3. runtime     — Traceback / RuntimeError          (medium X*)
      4. logic       — FAILED / AssertionError           (highest X*)

    Returns None if no error signal is detected (passing step).

    Args:
        tool_output: The stdout/stderr from a bash tool call.

    Returns:
        One of "environment", "syntax", "runtime", "logic", or None.
    """
    if not tool_output:
        return None

    if any(sig in tool_output for sig in _ENVIRONMENT_SIGNALS):
        return "environment"

    if any(sig in tool_output for sig in _SYNTAX_SIGNALS):
        return "syntax"

    if any(sig in tool_output for sig in _RUNTIME_SIGNALS):
        return "runtime"

    if any(sig in tool_output for sig in _LOGIC_SIGNALS):
        return "logic"

    return None
