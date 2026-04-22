from __future__ import annotations


def count_tokens(text: str) -> int:
    return len(text.split())


def sdpt(raw_accuracy: float, compressed_accuracy: float, raw_tokens: int, compressed_tokens: int) -> float:
    if raw_tokens <= 0 or compressed_tokens <= 0:
        raise ValueError("Token counts must be positive.")
    delta_accuracy = compressed_accuracy - raw_accuracy
    delta_tokens = raw_tokens - compressed_tokens
    return delta_accuracy / delta_tokens if delta_tokens else 0.0
