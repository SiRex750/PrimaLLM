from __future__ import annotations

import json
from typing import Any

import ollama

from caveman.benchmark.metrics import calculate_sdpt, count_tokens
from caveman.core.cache import L1Cache
from caveman.core.compressor import generate_caveman_prose
from caveman.core.graph import rank_triples_by_importance
from sentinel.core.source_graph import build_source_graph
from sentinel.core.verifier import verify_claim
from sentinel.core.wiki_storage import load_wiki, save_verified_fact
from shared.extractor import extract_claim_triples, extract_source_triples
from shared.triple import KnowledgeTriple


DATASET: list[dict[str, str]] = [
    {
        "text": (
            "The mitochondria is the powerhouse of the cell. "
            "It generates ATP through oxidative phosphorylation. "
            "Ribosomes synthesize proteins."
        ),
        "question": "How does the cell generate ATP?",
        "expected": "oxidative phosphorylation",
    },
    {
        "text": (
            "Julius Caesar crossed the Rubicon river in 49 BC, igniting a civil war. "
            "Pompey fled to Greece. "
            "Caesar later became dictator for life."
        ),
        "question": "What river did Caesar cross?",
        "expected": "Rubicon",
    },
    {
        "text": (
            "NVIDIA reported record revenue of $26 billion for the first quarter. "
            "The Data Center segment generated $22 billion in sales. "
            "Hopper GPUs drove the massive increase in AI infrastructure demand."
        ),
        "question": "How much revenue did the Data Center segment generate?",
        "expected": "22 billion",
    },
    {
        "text": (
            "CRISPR-Cas9 is a revolutionary gene-editing technology. "
            "It uses a guide RNA to target specific DNA sequences in the genome. "
            "The Cas9 enzyme acts as molecular scissors to cut the DNA strand."
        ),
        "question": "What acts as molecular scissors to cut the DNA?",
        "expected": "Cas9 enzyme",
    },
    {
        "text": (
            "The Apollo 11 mission launched on a Saturn V rocket. "
            "Neil Armstrong and Buzz Aldrin descended to the lunar surface in the Eagle module. "
            "Michael Collins remained in lunar orbit aboard the Command Module Columbia."
        ),
        "question": "Who remained in lunar orbit?",
        "expected": "Michael Collins",
    },
    {
        "text": (
            "The central bank raised interest rates by 50 basis points to combat rising inflation. "
            "The stock market reacted negatively, with the S&P 500 dropping 2 percent. "
            "Bond yields surged to their highest levels in a decade."
        ),
        "question": "Why did the central bank raise interest rates?",
        "expected": "combat rising inflation",
    },
    {
        "text": (
            "Transformers process sequential data using a mechanism called self-attention. "
            "Unlike recurrent neural networks, they do not require data to be processed in order. "
            "This allows for massive parallelization during training."
        ),
        "question": "What mechanism do Transformers use to process data?",
        "expected": "self-attention",
    },
    {
        "text": (
            "Photosynthesis occurs in the chloroplasts of plant cells. "
            "Chlorophyll pigments absorb sunlight to convert carbon dioxide and water into glucose. "
            "Oxygen is released as a byproduct of this chemical reaction."
        ),
        "question": "What is released as a byproduct of photosynthesis?",
        "expected": "Oxygen",
    }
]


from app import get_embedder
from sklearn.metrics.pairwise import cosine_similarity

def query_l2_memory(keyword: str, source_graph) -> str:
    if source_graph is None or not keyword:
        return ""

    embedder = get_embedder()
    keyword_vector = embedder.encode(keyword)
    graph = source_graph.graph

    best_node = None
    best_score = -1.0

    for node, data in graph.nodes(data=True):
        if "vector" in data and data["vector"] is not None:
            node_vector = data["vector"]
            sim = cosine_similarity([keyword_vector], [node_vector])[0][0]
            if sim > best_score:
                best_score = sim
                best_node = node

    if best_score < 0.60 or best_node is None:
        return ""

    edge_lines: list[str] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for subject, obj, data in graph.out_edges(best_node, data=True):
        verb = str((data or {}).get("verb", "")).strip()
        edge_key = (str(subject), verb, str(obj))
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            edge_lines.append(f"{subject} {verb} {obj}".strip())

    for subject, obj, data in graph.in_edges(best_node, data=True):
        verb = str((data or {}).get("verb", "")).strip()
        edge_key = (str(subject), verb, str(obj))
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            edge_lines.append(f"{subject} {verb} {obj}".strip())

    return " | ".join(edge_lines)


def query_l3_wiki(keyword: str) -> str:
    tokens = {token.lower() for token in keyword.split() if token.strip()}
    if not tokens:
        return ""

    matched_facts: list[str] = []
    for fact in load_wiki():
        fact_text = " ".join(
            [
                str(fact.get("subject", "")),
                str(fact.get("verb", "")),
                str(fact.get("object", "")),
            ]
        ).strip()
        if fact_text and any(token in fact_text.lower() for token in tokens):
            matched_facts.append(fact_text)

    return " ".join(matched_facts)


def _extract_keyword(tool_call) -> str:
    raw_arguments = getattr(getattr(tool_call, "function", None), "arguments", "")
    if not raw_arguments:
        return ""

    try:
        payload = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return ""

    keyword = payload.get("keyword") if isinstance(payload, dict) else ""
    return str(keyword or "").strip()


def ask_judge(caveman_context: str, question: str, source_graph) -> str:
    print("\n" + "=" * 100)
    print("L1 CONTEXT GENERATED")
    print("=" * 100)
    print(caveman_context)

    SYSTEM_INSTRUCTION = """You are the NMMU (Neural Memory Management Unit), a strict hardware instruction decoder. You do not converse. You do not explain your thoughts. 

You have two operating modes. You must output ONLY ONE of the following:

MODE 1: CACHE HIT (Answer Synthesis)
If the L1 Cache (Context) contains the answer to the user's query, output the final answer directly.

MODE 2: CACHE MISS (Memory Fault)
If the answer is NOT in the L1 Cache, you MUST trigger an L2 Page Fault by outputting STRICTLY a JSON object matching this schema. Do not output any text before or after the JSON:
{
    "tool": "search_memory",
    "keyword": "exact_semantic_keyword_to_search"
}"""

    conversation: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": f"Context: {caveman_context}\nQuestion: {question}",
        },
    ]

    response = ollama.chat(model='qwen2.5:1.5b', messages=conversation)
    content = response.get('message', {}).get('content', '').strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("tool") == "search_memory":
            keyword = str(parsed.get("keyword", "")).strip()
            print("\n" + "=" * 100)
            print("TOOL CALL TRIGGERED")
            print("=" * 100)
            print(f"Tool: search_memory | Keyword: {keyword or '<empty>'}")

            l2_result = query_l2_memory(keyword, source_graph)
            if l2_result:
                memory_result = f"L2 HIT: {l2_result}"
                print("\n" + "=" * 100)
                print("MEMORY HIT (L2)")
                print("=" * 100)
                print(memory_result)
            else:
                l3_result = query_l3_wiki(keyword)
                if l3_result:
                    memory_result = f"L3 HIT: {l3_result}"
                    print("\n" + "=" * 100)
                    print("MEMORY HIT (L3)")
                    print("=" * 100)
                    print(memory_result)
                else:
                    memory_result = "No matching memory found in L2 or L3."
                    print("\n" + "=" * 100)
                    print("MEMORY HIT (L2/L3)")
                    print("=" * 100)
                    print(memory_result)

            conversation.append({"role": "assistant", "content": content})
            conversation.append({"role": "user", "content": f"TOOL RESULT: {memory_result}"})

            final_response = ollama.chat(model='qwen2.5:1.5b', messages=conversation)
            final_answer = final_response.get('message', {}).get('content', '').strip()
        else:
            final_answer = content
    except json.JSONDecodeError:
        final_answer = content

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(final_answer)

    dirty_triples = extract_claim_triples(final_answer)
    print("\n" + "=" * 100)
    print("SENTINEL VERIFICATION STATUS")
    print("=" * 100)
    if not dirty_triples:
        print("No triples extracted from final answer; nothing to verify.")
        return final_answer

    for triple in dirty_triples:
        result = verify_claim(triple, source_graph)
        if result.is_verified:
            print(f"✅ CLEAN: [{triple.as_text()}]")
            save_verified_fact(triple)
        else:
            print(f"❌ DIRTY (Hallucination): [{triple.as_text()}]")

    return final_answer


def evaluate_accuracy(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def main() -> int:
    rows: list[dict[str, Any]] = []

    for item in DATASET:
        text = item["text"]
        question = item["question"]
        expected = item["expected"]

        raw_tokens = count_tokens(text)

        triples = extract_source_triples(text)
        ranked_triples = rank_triples_by_importance(triples)
        source_graph = build_source_graph(triples, embedder=get_embedder())

        cache = L1Cache(budgets={"facts": 30})
        for triple, score in ranked_triples:
            cache.add_fact(triple, pagerank_score=score)

        cached_triples = [entry.triple for entry in cache.active_facts.values()]
        caveman_text = generate_caveman_prose(cached_triples)

        caveman_tokens = count_tokens(caveman_text)
        reduction = ((raw_tokens - caveman_tokens) / raw_tokens * 100.0) if raw_tokens else 0.0
        sdpt_value = calculate_sdpt(len(cached_triples), caveman_tokens) if caveman_tokens > 0 else 0.0

        answer = ask_judge(caveman_text, question, source_graph)
        is_correct = evaluate_accuracy(answer, expected)

        rows.append(
            {
                "question": question,
                "expected": expected,
                "answer": answer,
                "raw_tokens": raw_tokens,
                "caveman_tokens": caveman_tokens,
                "reduction": reduction,
                "sdpt": sdpt_value,
                "accuracy": is_correct,
            }
        )

    print("=" * 106)
    print("CAVEMAN BENCHMARK REPORT")
    print("=" * 106)
    print(
        f"{'Case':<4} {'Raw Tokens':>11} {'Caveman Tokens':>15} {'Reduction %':>12} {'SDpT (Lower Better)':>20} {'Accuracy':>10}"
    )
    print("-" * 106)
    for index, row in enumerate(rows, start=1):
        accuracy_label = "PASS" if row["accuracy"] else "FAIL"
        print(
            f"{index:<4} {row['raw_tokens']:>11} {row['caveman_tokens']:>15} "
            f"{row['reduction']:>11.2f}% {row['sdpt']:>20.4f} {accuracy_label:>10}"
        )
    print("-" * 106)
    for index, row in enumerate(rows, start=1):
        print(f"Case {index}: {row['question']}")
        print(f"  Expected: {row['expected']}")
        print(f"  Answer:   {row['answer']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
