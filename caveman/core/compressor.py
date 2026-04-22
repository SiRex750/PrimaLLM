from __future__ import annotations

from shared.triple import KnowledgeTriple


def compress_triples(triples: list[KnowledgeTriple], max_items: int = 5) -> str:
    selected = triples[:max_items]
    if not selected:
        return "No facts available to compress."
    return "\n".join(f"- {triple.as_text()}" for triple in selected)


def build_caveman_prompt(triples: list[KnowledgeTriple]) -> str:
    fact_block = compress_triples(triples)
    return (
        "You are Caveman, a local compression layer. "
        "Summarize the following facts into compact, factual prose without inventing new claims.\n"
        f"{fact_block}"
    )
