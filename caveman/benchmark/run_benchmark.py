from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from caveman.benchmark.metrics import calculate_sdpt, count_tokens
from caveman.core.cache import L1Cache
from caveman.core.compressor import generate_caveman_prose
from caveman.core.graph import rank_triples_by_importance
from sentinel.core.source_graph import build_source_graph
from sentinel.core.verifier import verify_claim
from sentinel.core.wiki_storage import load_wiki, save_verified_fact
from shared.extractor import extract_knowledge_triples
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


def query_l2_memory(keyword: str, caveman_context: str) -> str:
    tokens = {token.lower() for token in keyword.split() if token.strip()}
    if not tokens:
        return ""

    matched_lines = [
        line.strip()
        for line in caveman_context.splitlines()
        if line.strip() and any(token in line.lower() for token in tokens)
    ]
    return " ".join(matched_lines)


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
    client = OpenAI()
    print("\n" + "=" * 100)
    print("L1 CONTEXT GENERATED")
    print("=" * 100)
    print(caveman_context)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Searches memory stores when context is insufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Keyword to find in L2 and L3 memory.",
                        }
                    },
                    "required": ["keyword"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    conversation: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "Answer using the Context. If the answer is NOT there, DO NOT GUESS. Use the search_memory tool.",
        },
        {
            "role": "user",
            "content": f"Context: {caveman_context}\nQuestion: {question}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        tools=tools,
    )

    message = response.choices[0].message
    conversation.append({"role": "assistant", "content": message.content or ""})

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            keyword = _extract_keyword(tool_call)
            print("\n" + "=" * 100)
            print("TOOL CALL TRIGGERED")
            print("=" * 100)
            print(f"Tool: search_memory | Keyword: {keyword or '<empty>'}")

            l2_result = query_l2_memory(keyword, caveman_context)
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

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "search_memory",
                    "content": memory_result,
                }
            )

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            tools=tools,
        )
        final_answer = (final_response.choices[0].message.content or "").strip()
    else:
        final_answer = (message.content or "").strip()

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(final_answer)

    dirty_triples = extract_knowledge_triples(final_answer)
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

        triples = extract_knowledge_triples(text)
        ranked_triples = rank_triples_by_importance(triples)
        source_graph = build_source_graph(triples)

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

    print("=" * 90)
    print("CAVEMAN BENCHMARK REPORT")
    print("=" * 90)
    print(
        f"{'Case':<4} {'Raw Tokens':>11} {'Caveman Tokens':>15} {'Reduction %':>12} {'SDpT':>10} {'Accuracy':>10}"
    )
    print("-" * 90)
    for index, row in enumerate(rows, start=1):
        accuracy_label = "PASS" if row["accuracy"] else "FAIL"
        print(
            f"{index:<4} {row['raw_tokens']:>11} {row['caveman_tokens']:>15} "
            f"{row['reduction']:>11.2f}% {row['sdpt']:>10.4f} {accuracy_label:>10}"
        )
    print("-" * 90)
    for index, row in enumerate(rows, start=1):
        print(f"Case {index}: {row['question']}")
        print(f"  Expected: {row['expected']}")
        print(f"  Answer:   {row['answer']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
