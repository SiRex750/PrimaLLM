from __future__ import annotations

from caveman.core.cache import L1Cache
from caveman.core.compressor import compress_triples
from caveman.core.extractor import extract_knowledge_triples
from caveman.core.graph import rank_triples_by_importance


def run_demo() -> None:
    cache = L1Cache(token_budget=128)
    print("Caveman demo. Enter text to extract local triples; type exit to quit.")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break

        triples = extract_knowledge_triples(user_input)
        if not triples:
            print("Caveman: no triples extracted.")
            continue

        for triple, score in rank_triples_by_importance(triples):
            cache.add_fact(triple, pagerank_score=score)

        print("Caveman compressed context:")
        print(compress_triples(list(entry.triple for entry in cache.active_facts.values())))


if __name__ == "__main__":
    run_demo()
