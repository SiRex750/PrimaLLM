from __future__ import annotations

import tiktoken


_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def sdpt(unique_acus_preserved: float, compressed_token_count: int) -> float:
    """Return SDpT as compressed tokens per preserved ACU; lower values indicate better compression efficiency."""
    if unique_acus_preserved <= 0:
        raise ValueError("unique_acus_preserved must be positive.")
    if compressed_token_count < 0:
        raise ValueError("compressed_token_count must be non-negative.")
    return compressed_token_count / unique_acus_preserved


def calculate_sdpt(unique_acus_preserved: float, compressed_token_count: int) -> float:
    return sdpt(unique_acus_preserved, compressed_token_count)
