from __future__ import annotations

import json
import os
import re
import sys

import ollama

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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5")


def _configure_console_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _banner(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


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
    from shared.l3_memory import fetch_clean_facts_by_similarity
    embedder = get_embedder()

    results = fetch_clean_facts_by_similarity(
        keyword=keyword,
        embedder=embedder,
        threshold=0.55,
        limit=3
    )

    if not results:
        return ""

    lines = []
    for fact in results:
        triple_text = (
            f"{fact['subject']} {fact['verb']} {fact['object']}".strip()
        )
        source_page = int(fact.get("source_page") or 0)
        citation = f" (source_page={source_page})" if source_page > 0 else ""
        lines.append(f"{triple_text}{citation}")

    return " | ".join(lines)


def main() -> None:
    _configure_console_encoding()

    _banner("STEP 1 - INGESTION (L2 RAM)")
    source_triples = extract_knowledge_triples(source_text)
    source_graph = build_source_graph(source_triples, embedder=get_embedder())
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

    _banner("STEP 3 - INFERENCE CALL 1")
    
    SYSTEM_INSTRUCTION = """You are the NMMU (Neural Memory Management Unit), a strict hardware instruction decoder. 
You do not converse. You do not explain your thoughts. 

You have two operating modes. You MUST output ONLY the mode's payload, NEVER the mode name itself.

1. CACHE HIT (Answer Synthesis):
   If the L1 Cache (Context) contains the answer to the user's query, output the final answer directly.

2. CACHE MISS (Memory Fault):
   If the answer is NOT in the L1 Cache, you MUST trigger an L2 Page Fault by outputting STRICTLY a JSON object matching this schema. Do not output any text before or after the JSON:
{
    "tool": "search_memory",
    "keyword": "exact_semantic_keyword_to_search"
}

Examples:
Input: "What is an apple?"
Assistant: "An apple is a round, edible fruit."

Input: "Who is King Rerir?"
Assistant: {"tool": "search_memory", "keyword": "King Rerir"}

Input: "What is the formula for photosynthesis?" (If not in L1 Cache)
Assistant: {"tool": "search_memory", "keyword": "photosynthesis formula"}

[SYSTEM: TOOL RESULT: No memory hit for keyword]
Assistant: INSUFFICIENT DATA. The source document does not contain this information."""
    
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": f"Context: {caveman_context}\nQuestion: {question}",
        },
        {
            "role": "assistant",
            "content": "",
        }
    ]
    
    response = ollama.chat(model=OLLAMA_MODEL, messages=conversation, options={
        "temperature": 0.0,
        "num_predict": 512,
        "repeat_penalty": 1.05,
    })
    content = response.get('message', {}).get('content', '').strip()
    
    # Robust sanitation: remove any mode-locking prefixes
    content = re.sub(r"^(MODE \d:|CACHE (HIT|MISS)( \(Memory Fault\))?:)", "", content, flags=re.IGNORECASE).strip()

    _banner("STEP 4 - TOOL EXECUTION")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("tool") == "search_memory":
            keyword = str(parsed.get("keyword", "")).strip()
            print(f"Memory Fault (JSON tool call) triggered for keyword: {keyword}")

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

            conversation.append({"role": "assistant", "content": content})
            conversation.append({
                "role": "user",
                "content": (
                    f"TOOL RESULT: {tool_result}\n\n"
                    "COMMAND: Search complete. You MUST synthesize the final answer now "
                    "using ONLY the tool result above. DO NOT output JSON. DO NOT call "
                    "search_memory again. If the result is empty, reply with "
                    "'INSUFFICIENT DATA'."
                ),
            })

            follow_up = ollama.chat(model=OLLAMA_MODEL, messages=conversation)
            final_answer = follow_up.get('message', {}).get('content', '').strip()
        else:
            print("CACHE HIT. No tool call triggered.")
            final_answer = content
    except json.JSONDecodeError:
        print("CACHE HIT. No valid JSON tool call triggered.")
        final_answer = content

    _banner("FINAL ANSWER")
    print(final_answer or "<empty>")

    _banner("STEP 6 - SENTINEL WRITE-BACK")
    dirty_triples = extract_knowledge_triples(final_answer)
    has_contradiction = False
    if not dirty_triples:
        print("No triples extracted from final answer. Nothing to verify.")

    for triple in dirty_triples:
        verdict = verify_claim(triple, source_graph)
        if verdict.is_verified:
            print(f"\u2705 CLEAN: [{triple.as_text()}]")
            print(f"   Sentinel reason: {verdict.reason}")
            save_verified_fact(triple)
        elif verdict.label == "contradiction":
            has_contradiction = True
            print(f"\u274c CONTRADICTION: [{triple.as_text()}]")
            print(f"   Sentinel reason: {verdict.reason}")
        else:
            print(f"\u26a0\ufe0f NEUTRAL: [{triple.as_text()}]")
            print(f"   Sentinel reason: {verdict.reason}")

    if has_contradiction:
        final_answer = (
            "\ud83d\udea8 NLI SENTINEL BLOCK: My policy engine attempted to answer this, "
            "but the local DeBERTa-v3 verification failed. The source document "
            "does not support this claim."
        )
        _banner("SENTINEL OVERRIDE")
        print(final_answer)


if __name__ == "__main__":
    main()