"""Token counting utilities using tiktoken with cl100k_base encoding.

This module provides a proxy for Qwen token counts using the cl100k_base
encoding, which is used throughout the harness for tracking token usage.
"""

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using cl100k_base encoding.

    Args:
        text: The text to tokenize.

    Returns:
        The number of tokens in the text.
    """
    if not text:
        return 0
    return len(_ENCODING.encode(text))
