from __future__ import annotations

import json
import re
import sys

from openai import OpenAI

from caveman.core import L1Cache, generate_caveman_prose, rank_triples_by_importance
from sentinel.core import build_source_graph, verify_claim
from sentinel.core.wiki_storage import load_wiki, save_verified_fact
from shared.extractor import extract_knowledge_triples


source_text = (
    "The Apollo 11 mission launched on a Saturn V rocket. "
    "Neil Armstrong and Buzz Aldrin descended to the lunar surface in the Eagle module. "
    "Michael Collins remained in lunar orbit aboard the Command Module Columbia."
)
question = "Who descended to the lunar surface in the Eagle module?"


def _configure_console_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _banner(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def query_l2_memory(keyword: str, source_graph) -> str:
    if source_graph is None:
        return ""

    keywords = [word for word in re.findall(r"[A-Za-z0-9]+", keyword.lower()) if len(word) > 3]
    if not keywords:
        return ""

    graph = source_graph.graph
    matching_nodes = [
        node
        for node in graph.nodes
        if any(term in str(node).lower() for term in keywords)
    ]
    if not matching_nodes:
        return ""

    edge_lines: list[str] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for node in matching_nodes:
        for subject, obj, data in graph.out_edges(node, data=True):
            verb = str((data or {}).get("verb", "")).strip()
            edge_key = (str(subject), verb, str(obj))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_lines.append(f"{subject} {verb} {obj}".strip())

        for subject, obj, data in graph.in_edges(node, data=True):
            verb = str((data or {}).get("verb", "")).strip()
            edge_key = (str(subject), verb, str(obj))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_lines.append(f"{subject} {verb} {obj}".strip())

    return " | ".join(edge_lines)


def query_l3_wiki(keyword: str) -> str:
    keywords = [word for word in keyword.lower().split() if len(word) > 3]
    if not keywords:
        return ""

    for fact in load_wiki():
        triple_text = (
            f"{fact.get('subject', '')} {fact.get('verb', '')} {fact.get('object', '')}".strip()
        )
        lowered_triple_text = triple_text.lower()
        if any(term in lowered_triple_text for term in keywords):
            return triple_text
    return ""


def main() -> None:
    _configure_console_encoding()

    _banner("STEP 1 - INGESTION (L2 RAM)")
    source_triples = extract_knowledge_triples(source_text)
    source_graph = build_source_graph(source_triples)
    print(f"Source triples extracted: {len(source_triples)}")
    for triple in source_triples:
        print(f"  - {triple.as_text()}")

    _banner("STEP 2 - L1 CAVEMAN")
    l1_cache = L1Cache(budgets={"facts": 15})
    ranked = rank_triples_by_importance(source_triples)
    for triple, score in ranked:
        l1_cache.add_fact(triple, pagerank_score=score)

    active_triples = [entry.triple for entry in l1_cache.active_facts.values()]
    caveman_context = generate_caveman_prose(active_triples)
    print("L1 Context generated:")
    for line in l1_cache.as_context_lines():
        print(f"  - {line}")
    print(f"Compressed Caveman context: {caveman_context}")

    _banner("STEP 3 - TOOL SCHEMA")
    search_memory = {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search tiered memory using a keyword and return relevant memory text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search in memory tiers.",
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    }
    print(json.dumps(search_memory, indent=2))

    _banner("STEP 4 - CALL 1")
    client = OpenAI()
    conversation = [
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
        tools=[search_memory],
    )

    message = response.choices[0].message
    final_answer = (message.content or "").strip()

    _banner("STEP 5 - TOOL EXECUTION")
    if message.tool_calls:
        print("Tool Call triggered.")

        conversation.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            raw_args = tool_call.function.arguments or "{}"
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"keyword": raw_args}

            keyword = str(parsed_args.get("keyword", "")).strip()
            print(f"Tool request keyword: {keyword}")

            l2_result = query_l2_memory(keyword, source_graph)
            if l2_result:
                tool_result = l2_result
                print(f"Memory Hit (L2): {l2_result}")
            else:
                l3_result = query_l3_wiki(keyword)
                if l3_result:
                    tool_result = l3_result
                    print(f"Memory Hit (L3): {l3_result}")
                else:
                    tool_result = f"No memory hit for keyword: {keyword}"
                    print(f"Memory Hit (L2/L3): {tool_result}")

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "search_memory",
                    "content": tool_result,
                }
            )

        follow_up = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
        )
        final_answer = (follow_up.choices[0].message.content or "").strip()
    else:
        print("Tool Call not triggered. Answered from context directly.")

    _banner("FINAL ANSWER")
    print(final_answer or "<empty>")

    _banner("STEP 6 - SENTINEL WRITE-BACK")
    dirty_triples = extract_knowledge_triples(final_answer)
    if not dirty_triples:
        print("No triples extracted from final answer. Nothing to verify.")

    for triple in dirty_triples:
        verdict = verify_claim(triple, source_graph)
        if verdict.is_verified:
            print(f"✅ CLEAN: [{triple.as_text()}]")
            print(f"   Sentinel reason: {verdict.reason}")
            save_verified_fact(triple)
        else:
            print(f"❌ DIRTY (Hallucination): [{triple.as_text()}]")
            print(f"   Sentinel reason: {verdict.reason}")


if __name__ == "__main__":
    main()