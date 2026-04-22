from __future__ import annotations

import tiktoken


_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def sdpt(unique_acus_preserved: float, compressed_token_count: int) -> float:
    if compressed_token_count <= 0:
        raise ValueError("compressed_token_count must be positive.")
    return unique_acus_preserved / compressed_token_count
