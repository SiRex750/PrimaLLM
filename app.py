from __future__ import annotations

import json
import re

import openai
import PyPDF2
import streamlit as st

from caveman.core import L1Cache, rank_triples_by_importance
from sentinel.core import build_source_graph, verify_claim
from sentinel.core.wiki_storage import load_wiki, save_verified_fact
from shared.extractor import extract_knowledge_triples


st.set_page_config(layout="wide", page_title="PrimaLLM: Context OS")


SYSTEM_INSTRUCTION = """You are the policy engine for a Set-Associative Tiered Memory System. 
1. Answer the user's question using ONLY the provided Context.
2. If the answer is missing from the Context, you MUST execute the 'search_memory' tool.
3. CIRCUIT BREAKER: If you receive a tool response that says "No memory hit", DO NOT attempt to use the tool again. Simply state: "I could not find the exact answer in the source document."
4. CRITICAL: Do NOT generate any conversational filler. If you need to use the tool, output ONLY the tool call and absolutely no other text."""


def _required_system_budget() -> int:
    # Keep pinned system memory large enough for the full instruction with headroom.
    return max(128, len(SYSTEM_INSTRUCTION.split()) * 3)


def _init_session_state() -> None:
    desired_system_budget = _required_system_budget()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "source_graph" not in st.session_state:
        st.session_state.source_graph = None

    if "l1_cache" not in st.session_state:
        st.session_state.l1_cache = L1Cache(
            budgets={
                "system": desired_system_budget,
                "facts": 100,
                "history": 200,
                "tools": 150,
            }
        )

    if "telemetry" not in st.session_state:
        st.session_state.telemetry = {
            "l1_status": "initialized",
            "tool_calls": 0,
            "memory_faults": [],
            "sentinel_log": [],
        }

    if "loaded_pdf_name" not in st.session_state:
        st.session_state.loaded_pdf_name = None

    cache: L1Cache = st.session_state.l1_cache
    if cache.budgets.get("system", 0) < desired_system_budget:
        upgraded_cache = L1Cache(
            budgets={
                "system": desired_system_budget,
                "facts": cache.budgets.get("facts", 100),
                "history": cache.budgets.get("history", 200),
                "tools": cache.budgets.get("tools", 150),
            }
        )

        for entry in cache.set_facts.values():
            upgraded_cache.add_fact(entry.triple, pagerank_score=entry.pagerank_score)
        for turn in cache.set_history:
            upgraded_cache.add_history_turn(turn.role, turn.text)
        for tool_result in cache.set_tools:
            upgraded_cache.add_tool_result(tool_result.tool_name, tool_result.text)

        st.session_state.l1_cache = upgraded_cache
        cache = upgraded_cache

    if not cache.set_system:
        try:
            cache.add_system_instruction(SYSTEM_INSTRUCTION)
        except ValueError:
            fallback_cache = L1Cache(
                budgets={
                    "system": desired_system_budget * 2,
                    "facts": cache.budgets.get("facts", 100),
                    "history": cache.budgets.get("history", 200),
                    "tools": cache.budgets.get("tools", 150),
                }
            )

            for entry in cache.set_facts.values():
                fallback_cache.add_fact(entry.triple, pagerank_score=entry.pagerank_score)
            for turn in cache.set_history:
                fallback_cache.add_history_turn(turn.role, turn.text)
            for tool_result in cache.set_tools:
                fallback_cache.add_tool_result(tool_result.tool_name, tool_result.text)

            fallback_cache.add_system_instruction(SYSTEM_INSTRUCTION)
            st.session_state.l1_cache = fallback_cache


def _push_telemetry_item(key: str, value: str, max_items: int = 12) -> None:
    items = st.session_state.telemetry.setdefault(key, [])
    items.append(value)
    if len(items) > max_items:
        del items[:-max_items]


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


def process_pdf(file) -> tuple[int, int]:
    reader = PyPDF2.PdfReader(file)
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        st.session_state.source_graph = None
        return 0, 0

    triples = extract_knowledge_triples(full_text)
    source_graph = build_source_graph(triples)
    st.session_state.source_graph = source_graph

    cache: L1Cache = st.session_state.l1_cache
    cache.set_facts.clear()
    cache.set_tools.clear()

    ranked = rank_triples_by_importance(triples)
    for triple, score in ranked:
        cache.add_fact(triple, pagerank_score=score)

    st.session_state.telemetry["l1_status"] = "pdf_loaded"
    _push_telemetry_item(
        "memory_faults",
        f"PDF ingested: {len(triples)} triples -> {len(cache.set_facts)} active fact entries",
    )
    return len(triples), source_graph.graph.number_of_nodes()


def _render_sidebar() -> None:
    st.sidebar.title("NMMU Telemetry")

    source_graph = st.session_state.source_graph
    active_nodes = source_graph.graph.number_of_nodes() if source_graph is not None else 0
    st.sidebar.metric("Active L2 Nodes", active_nodes)
    st.sidebar.caption(f"L1 Status: {st.session_state.telemetry.get('l1_status', 'unknown')}")
    st.sidebar.caption(f"Tool Calls: {st.session_state.telemetry.get('tool_calls', 0)}")

    cache: L1Cache = st.session_state.l1_cache
    with st.sidebar.expander("L1 Cache Partitions", expanded=True):
        st.markdown("**System**")
        st.write(cache.set_system or ["<empty>"])

        st.markdown("**Facts**")
        st.write([entry.text for entry in cache.set_facts.values()] or ["<empty>"])

        st.markdown("**History**")
        st.write([f"{turn.role}: {turn.text}" for turn in cache.set_history] or ["<empty>"])

        st.markdown("**Tools**")
        st.write([f"{item.tool_name}: {item.text}" for item in cache.set_tools] or ["<empty>"])

    st.sidebar.subheader("Latest Memory Faults")
    memory_faults = st.session_state.telemetry.get("memory_faults", [])
    if memory_faults:
        for item in reversed(memory_faults[-8:]):
            st.sidebar.write(f"- {item}")
    else:
        st.sidebar.write("No memory faults recorded yet.")

    st.sidebar.subheader("Sentinel Log")
    sentinel_log = st.session_state.telemetry.get("sentinel_log", [])
    if sentinel_log:
        for item in reversed(sentinel_log[-8:]):
            st.sidebar.write(f"- {item}")
    else:
        st.sidebar.write("No sentinel events recorded yet.")


def _search_memory_tool_schema() -> dict:
    return {
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


def _extract_keyword(tool_call) -> str:
    raw_args = tool_call.function.arguments or "{}"
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        parsed = {"keyword": raw_args}
    return str(parsed.get("keyword", "")).strip()


def _run_sentinel_writeback(final_answer: str) -> None:
    source_graph = st.session_state.source_graph
    if source_graph is None:
        _push_telemetry_item("sentinel_log", "No source graph loaded; Sentinel verification skipped.")
        return

    answer_triples = extract_knowledge_triples(final_answer)
    if not answer_triples:
        _push_telemetry_item("sentinel_log", "No triples extracted from assistant answer.")
        return

    for triple in answer_triples:
        verdict = verify_claim(triple, source_graph)
        if verdict.is_verified:
            save_verified_fact(triple)
            _push_telemetry_item(
                "sentinel_log",
                f"✅ CLEAN | {triple.as_text()} | {verdict.reason}",
            )
        else:
            _push_telemetry_item(
                "sentinel_log",
                f"❌ DIRTY | {triple.as_text()} | {verdict.reason}",
            )


def _chat_loop(prompt: str) -> str:
    cache: L1Cache = st.session_state.l1_cache
    source_graph = st.session_state.source_graph

    cache.add_history_turn("user", prompt)
    context_text = "\n".join(cache.as_context_lines())
    tools = [_search_memory_tool_schema()]

    conversation = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"},
    ]

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        tools=tools,
    )

    message = response.choices[0].message
    final_answer = (message.content or "").strip()

    if message.tool_calls:
        st.session_state.telemetry["tool_calls"] = st.session_state.telemetry.get("tool_calls", 0) + len(message.tool_calls)

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
            keyword = _extract_keyword(tool_call)
            l2_result = query_l2_memory(keyword, source_graph)

            if l2_result:
                tool_output = l2_result
                fault_line = f"L2 HIT | keyword='{keyword}' | {l2_result}"
            else:
                l3_result = query_l3_wiki(keyword)
                if l3_result:
                    tool_output = l3_result
                    fault_line = f"L3 HIT | keyword='{keyword}' | {l3_result}"
                else:
                    tool_output = f"No memory hit for keyword: {keyword}"
                    fault_line = f"MISS | keyword='{keyword}'"

            _push_telemetry_item("memory_faults", fault_line)
            cache.add_tool_result("search_memory", tool_output)

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "search_memory",
                    "content": tool_output,
                }
            )

        follow_up = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
        )
        final_answer = (follow_up.choices[0].message.content or "").strip()
        if not final_answer:
            final_answer = "Data retrieved from L2 Memory, but generation failed."

    cache.add_history_turn("assistant", final_answer)
    return final_answer


def main() -> None:
    _init_session_state()
    _render_sidebar()

    st.title("PrimaLLM: Context OS")
    st.caption("Tiered Memory Runtime: L1 Set Cache, L2 Source Graph, L3 Verified Wiki")

    uploaded_pdf = st.file_uploader("Upload source PDF", type=["pdf"])
    if uploaded_pdf is not None and st.session_state.loaded_pdf_name != uploaded_pdf.name:
        with st.spinner("Ingesting PDF into L2 and populating L1 facts..."):
            triple_count, node_count = process_pdf(uploaded_pdf)
            st.session_state.loaded_pdf_name = uploaded_pdf.name
        st.success(f"Loaded {uploaded_pdf.name}: {triple_count} triples, {node_count} L2 graph nodes")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about the loaded document")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        try:
            with st.status("NMMU thinking...", expanded=True) as status:
                status.write("Updating conversation history in L1 cache")
                status.write("Generating context and calling policy model")
                final_answer = _chat_loop(prompt)
                status.write("Running Sentinel verification and L3 write-back")
                _run_sentinel_writeback(final_answer)
                status.update(label="NMMU complete", state="complete")
        except Exception as exc:
            final_answer = f"Error: {exc}"
            st.error(final_answer)
            st.session_state.telemetry["l1_status"] = "error"

        answer_placeholder.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})


if __name__ == "__main__":
    main()