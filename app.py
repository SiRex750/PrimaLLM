from __future__ import annotations

import json
import os
import re
import tempfile

import networkx as nx
import ollama
import PyPDF2
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from caveman.core import L1Cache, rank_triples_by_importance
from sentinel.core import build_source_graph, verify_claim
from shared.extractor import extract_claim_triples, extract_source_triples
from shared.l3_memory import fetch_clean_facts, save_fact
from shared.triple import KnowledgeTriple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


st.set_page_config(
    layout="wide",
    page_title="PrimaLLM",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)


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
Input: "What year did the French Revolution begin?"
Assistant: "The French Revolution began in 1789."

Input: "Who founded Microsoft?"
Assistant: {"tool": "search_memory", "keyword": "Microsoft founder"}

Input: "What is the boiling point of nitrogen?" (If not in L1 Cache)
Assistant: {"tool": "search_memory", "keyword": "nitrogen boiling point"}

[SYSTEM: TOOL RESULT: No memory hit for keyword]
Assistant: INSUFFICIENT DATA. The source document does not contain this information."""

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "num_predict": 512,  # Increased to prevent truncating complex tool payloads
    "repeat_penalty": 1.05,
}
FORCE_SEARCH_PROMPT = "CACHE MISS. You must use the search_memory tool now."


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
                "facts": 400,     # Expanded for dense L2/L3 context (128k window)
                "history": 400,   # Expanded for longer technical conversations
                "tools": 300,
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

    if "graph_html" not in st.session_state:
        st.session_state.graph_html = None

    if "graph_rendered_for" not in st.session_state:
        st.session_state.graph_rendered_for = None

    if "triple_source_pages" not in st.session_state:
        st.session_state.triple_source_pages = {}

    cache: L1Cache = st.session_state.l1_cache
    if cache.budgets.get("system", 0) < desired_system_budget:
        upgraded_cache = L1Cache(
            budgets={
                "system": desired_system_budget,
                "facts": cache.budgets.get("facts", 400),
                "history": cache.budgets.get("history", 400),
                "tools": cache.budgets.get("tools", 300),
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
                    "facts": cache.budgets.get("facts", 400),
                    "history": cache.budgets.get("history", 400),
                    "tools": cache.budgets.get("tools", 300),
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


def _triple_key(triple: KnowledgeTriple) -> tuple[str, str, str]:
    return (
        str(triple.subject).strip().lower(),
        str(triple.verb).strip().lower(),
        str(triple.object).strip().lower(),
    )


def _resolve_source_page(
    triple: KnowledgeTriple,
    source_page_lookup: dict[tuple[str, str, str], int],
) -> int:
    key = _triple_key(triple)
    if key in source_page_lookup:
        return source_page_lookup[key]

    subj, verb, obj = key
    for (src_subj, src_verb, src_obj), page in source_page_lookup.items():
        if src_subj == subj and src_obj == obj:
            return page
        if src_subj == subj and src_verb == verb:
            return page

    return 0


def _inject_clean_facts_into_l1(cache: L1Cache, limit: int = 64) -> int:
    injected = 0
    for fact in fetch_clean_facts()[-limit:]:
        subject = str(fact.get("subject", "")).strip()
        verb = str(fact.get("verb", "")).strip()
        object_text = str(fact.get("object", "")).strip()
        if not (subject and verb and object_text):
            continue

        triple = KnowledgeTriple(subject, verb, object_text)
        if triple.as_text() in cache.set_facts:
            continue

        # L3 facts are valuable context, but should be lower-priority than active PDF facts.
        cache.add_fact(triple, pagerank_score=-1.0)
        injected += 1

    return injected


def _is_single_word_reply(text: str) -> bool:
    cleaned = text.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9]+[.!?]?", cleaned))


def _extract_search_keyword(content: str) -> str | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or parsed.get("tool") != "search_memory":
        return None

    keyword = str(parsed.get("keyword", "")).strip()
    return keyword or None


def _build_partitioned_messages(cache: L1Cache, prompt: str) -> list[dict[str, str]]:
    facts = [entry.text for entry in cache.set_facts.values()]
    facts_block = "\n".join(f"- {fact}" for fact in facts) if facts else "<empty>"

    user_turns = [turn.text for turn in cache.set_history if turn.role == "user"]
    last_two_user_turns = user_turns[-2:]
    user_turns_block = (
        "\n".join(f"- {turn}" for turn in last_two_user_turns)
        if last_two_user_turns
        else "<empty>"
    )

    user_content = (
        "L1 Context (Facts):\n"
        f"{facts_block}\n\n"
        "Last 2 User Messages:\n"
        f"{user_turns_block}\n\n"
        f"Current Question: {prompt}"
    )

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def _call_policy_model(messages: list[dict[str, str]]) -> str:
    request_messages = [dict(message) for message in messages]
    request_messages.append({"role": "assistant", "content": ""})

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=request_messages,
        options=OLLAMA_OPTIONS,
    )
    content = response.get("message", {}).get("content", "").strip()
    
    # Robust sanitation: remove any mode-locking prefixes
    content = re.sub(r"^(MODE \d:|CACHE (HIT|MISS)( \(Memory Fault\))?:)", "", content, flags=re.IGNORECASE).strip()
    return content


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
            # cosine_similarity expects 2D arrays
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


def process_pdf(file) -> tuple[int, int]:
    reader = PyPDF2.PdfReader(file)
    triples: list[KnowledgeTriple] = []
    source_page_lookup: dict[tuple[str, str, str], int] = {}
    all_sentences: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        page_triples = extract_source_triples(page_text)
        triples.extend(page_triples)

        import re
        page_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', page_text) if len(s.strip()) > 20]
        all_sentences.extend(page_sentences)

        for triple in page_triples:
            source_page_lookup.setdefault(_triple_key(triple), page_number)

    if not triples:
        st.session_state.source_graph = None
        st.session_state.triple_source_pages = {}
        return 0, 0
    
    embedder = get_embedder()
    source_graph = build_source_graph(triples, embedder=embedder, source_sentences=all_sentences)
    st.session_state.source_graph = source_graph
    st.session_state.triple_source_pages = source_page_lookup

    cache: L1Cache = st.session_state.l1_cache
    cache.set_facts.clear()
    cache.set_tools.clear()

    ranked = rank_triples_by_importance(triples)
    for triple, score in ranked:
        cache.add_fact(triple, pagerank_score=score)

    injected_from_l3 = _inject_clean_facts_into_l1(cache)

    st.session_state.telemetry["l1_status"] = "pdf_loaded"
    _push_telemetry_item(
        "memory_faults",
        (
            f"PDF ingested: {len(triples)} triples -> {len(cache.set_facts)} active fact entries "
            f"(L3 injected: {injected_from_l3})"
        ),
    )
    return len(triples), source_graph.graph.number_of_nodes()


def _render_sidebar() -> None:
    with st.sidebar:
        st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; 
     font-size:0.65rem; text-transform:uppercase; 
     letter-spacing:0.12em; color:#4d9fff; 
     padding:0.5rem 0; border-bottom:1px solid #2a2a3a;
     margin-bottom:0.5rem;">
  ▸ NMMU Telemetry
</div>
""", unsafe_allow_html=True)
        st.divider()

        # Node Density Card
        source_graph = st.session_state.source_graph
        active_nodes = source_graph.graph.number_of_nodes() if source_graph is not None else 0
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">Active L2 Node Density</div>
                <div class="telemetry-value">{active_nodes}</div>
            </div>
            """, unsafe_allow_html=True)

        # L1 Status Card
        l1_status = st.session_state.telemetry.get("l1_status", "idle").upper()
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">L1 Context Status</div>
                <div class="telemetry-value">{l1_status} <span class="status-tag">Live</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Tool Call Counters
        tool_calls = st.session_state.telemetry.get("tool_calls", 0)
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">Total Tool Faults (L2/L3)</div>
                <div class="telemetry-value">{tool_calls}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("<div class='telemetry-label'>Memory Fault Logs</div>", unsafe_allow_html=True)
        for fault in st.session_state.telemetry.get("memory_faults", [])[-5:]:
            st.caption(f"> {fault}")

        st.divider()
        if st.button("Flush L1 Cache"):
            st.session_state.l1_cache.set_facts.clear()
            st.session_state.l1_cache.set_history.clear()
            st.session_state.l1_cache.set_tools.clear()
            st.session_state.telemetry["l1_status"] = "flushed"
            st.rerun()

        st.divider()
        st.markdown(
            "<div class='telemetry-label' style='color:#b388ff'>Sentinel Gate Log</div>",
            unsafe_allow_html=True
        )
        for entry in st.session_state.telemetry.get("sentinel_log", [])[-5:]:
            if "CLEAN" in entry:
                colour = "#00e676"
            elif "CONTRADICTION" in entry:
                colour = "#ff4444"
            else:
                colour = "#ffab40"
            short = entry[:60] + "…" if len(entry) > 60 else entry
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;"
                f"font-size:0.62rem;color:{colour};padding:2px 0;"
                f"border-left:2px solid {colour};padding-left:6px;"
                f"margin:2px 0'>{short}</div>",
                unsafe_allow_html=True
            )

    cache: L1Cache = st.session_state.l1_cache
    with st.sidebar.expander("L1 CACHE PARTITIONS", expanded=False):
        st.markdown("**System**")
        st.write(cache.set_system or ["<empty>"])

        st.markdown("**Facts**")
        st.write([entry.text for entry in cache.set_facts.values()] or ["<empty>"])

        st.markdown("**History**")
        st.write([f"{turn.role}: {turn.text}" for turn in cache.set_history] or ["<empty>"])

        st.markdown("**Tools**")
        st.write([f"{item.tool_name}: {item.text}" for item in cache.set_tools] or ["<empty>"])


def _run_sentinel_writeback(final_answer: str) -> bool:
    """Verify answer triples against the source graph.

    Returns ``True`` if no triples actively **contradict** the source.
    Returns ``False`` if any triple is labelled *contradiction* by DeBERTa,
    indicating an active hallucination.
    """
    source_graph = st.session_state.source_graph
    if source_graph is None:
        _push_telemetry_item("sentinel_log", "No source graph loaded; Sentinel verification skipped.")
        return True

    answer_triples = extract_claim_triples(final_answer)
    if not answer_triples:
        _push_telemetry_item("sentinel_log", "No triples extracted from assistant answer.")
        return True

    source_page_lookup: dict[tuple[str, str, str], int] = st.session_state.get("triple_source_pages", {})
    has_contradiction = False

    for triple in answer_triples:
        verdict = verify_claim(triple, source_graph, source_sentences=source_graph.source_sentences)
        if verdict.is_verified:
            source_page = _resolve_source_page(triple, source_page_lookup)
            inserted = save_fact(
                triple,
                source_page=source_page,
                sentinel_status="CLEAN",
            )
            citation = f"p.{source_page}" if source_page > 0 else "p.?"
            write_result = "stored" if inserted else "duplicate_ignored"
            _push_telemetry_item(
                "sentinel_log",
                f"✅ CLEAN | {triple.as_text()} | {verdict.reason} | {citation} | {write_result}",
            )
        elif verdict.label == "contradiction":
            has_contradiction = True
            _push_telemetry_item(
                "sentinel_log",
                f"❌ CONTRADICTION | {triple.as_text()} | {verdict.reason}",
            )
        else:
            # Neutral — not entailed but not contradicted either (paraphrase)
            _push_telemetry_item(
                "sentinel_log",
                f"⚠️ NEUTRAL | {triple.as_text()} | {verdict.reason}",
            )

    _inject_clean_facts_into_l1(st.session_state.l1_cache)
    return not has_contradiction


def _chat_loop(prompt: str) -> str:
    cache: L1Cache = st.session_state.l1_cache
    source_graph = st.session_state.source_graph

    _inject_clean_facts_into_l1(cache)

    cache.add_history_turn("user", prompt)
    conversation = _build_partitioned_messages(cache, prompt)
    content = _call_policy_model(conversation)
    keyword = _extract_search_keyword(content)

    if keyword is None:
        tool_calls_so_far = int(st.session_state.telemetry.get("tool_calls", 0))
        if tool_calls_so_far == 0 and _is_single_word_reply(content):
            _push_telemetry_item(
                "memory_faults",
                "FORCE SEARCH | single-word non-JSON output while tool_calls=0",
            )
            conversation.append({"role": "assistant", "content": content})
            conversation.append({"role": "user", "content": FORCE_SEARCH_PROMPT})
            content = _call_policy_model(conversation)
            keyword = _extract_search_keyword(content)

    if keyword:
        st.session_state.telemetry["tool_calls"] = st.session_state.telemetry.get("tool_calls", 0) + 1

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

        conversation.append({"role": "assistant", "content": content})
        conversation.append({
            "role": "user",
            "content": (
                f"TOOL RESULT: {tool_output}\n\n"
                "COMMAND: Search complete. You MUST synthesize the final answer now "
                "using ONLY the tool result above. DO NOT output JSON. DO NOT call "
                "search_memory again. If the result is empty, reply with "
                "'INSUFFICIENT DATA'."
            ),
        })

        final_answer = _call_policy_model(conversation)
        if not final_answer:
            final_answer = "Data retrieved from L2 Memory, but generation failed."
    else:
        final_answer = content

    cache.add_history_turn("assistant", final_answer)
    return final_answer


def render_graph_visual(source_graph) -> str:
    """Build a PyVis interactive graph from the L2 SourceGraph.

    Nodes are sized by PageRank and colored by Sentinel verification
    status: green = Clean, red = Dirty, blue = unverified.
    Returns the generated HTML as a string.
    """
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="#fafafa",
    )
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.06,
          "damping": 0.5
        },
        "stabilization": {
          "enabled": true,
          "iterations": 200,
          "updateInterval": 25
        },
        "minVelocity": 0.75
      },
      "edges": {
        "color": { "color": "#3d3d5c", "highlight": "#4d9fff" },
        "font": { "size": 9, "color": "#555570", "face": "IBM Plex Mono" },
        "smooth": { "type": "continuous" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
        "width": 1
      },
      "nodes": {
        "shape": "dot",
        "font": { "size": 11, "face": "IBM Plex Mono" }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true,
        "navigationButtons": false
      }
    }
    """)

    graph: nx.DiGraph = source_graph.graph

    # ── PageRank scores for node sizing ──
    try:
        pr_scores = nx.pagerank(graph)
    except Exception:
        pr_scores = {}

    max_pr = max(pr_scores.values()) if pr_scores else 1.0

    # ── Sentinel verification status lookup ──
    sentinel_log = st.session_state.telemetry.get("sentinel_log", [])
    node_status: dict[str, str] = {}  # node label -> "clean" | "dirty"
    for entry in sentinel_log:
        entry_lower = entry.lower()
        if "clean" in entry_lower:
            status = "clean"
        elif "dirty" in entry_lower:
            status = "dirty"
        else:
            continue
        # Extract the triple text between the first and second "|"
        parts = entry.split("|")
        if len(parts) >= 2:
            triple_text = parts[1].strip()
            for word in triple_text.split():
                cleaned = word.strip()
                if cleaned:
                    node_status.setdefault(cleaned, status)

    # ── Color palette ──
    COLOR_CLEAN = "#00c853"   # green
    COLOR_DIRTY = "#ff1744"   # red
    COLOR_DEFAULT = "#448aff" # high-contrast blue

    # ── Add nodes ──
    for node in graph.nodes:
        label = str(node)
        pr = pr_scores.get(node, 0.0)
        size = 10 + 40 * (pr / max_pr) if max_pr else 15

        # Determine color from Sentinel status
        status = node_status.get(label)
        if status == "clean":
            color = COLOR_CLEAN
        elif status == "dirty":
            color = COLOR_DIRTY
        else:
            color = COLOR_DEFAULT

        # Truncate long labels for clean display
        display_label = label if len(label) <= 20 else label[:18] + "…"
        
        net.add_node(
            label,
            label=display_label,
            size=size,
            color=color,
            title=f"{label}\nPageRank: {pr:.4f}",  # full text on hover
            font={"size": 11, "color": "#e8e8f0", "face": "IBM Plex Mono"},
            borderWidth=1,
            borderWidthSelected=2,
        )

    # ── Add edges ──
    for src, dst, data in graph.edges(data=True):
        verb = str((data or {}).get("verb", "")).strip()
        net.add_edge(str(src), str(dst), label=verb, title=verb)

    # ── Write to temporary HTML file and return contents ──
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as fh:
        return fh.read()


def main() -> None:
    st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root theme ── */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-hover: #1e1e2a;
    --border: #2a2a3a;
    --border-accent: #3d3d5c;
    --text-primary: #e8e8f0;
    --text-secondary: #8888aa;
    --text-muted: #555570;
    --accent-blue: #4d9fff;
    --accent-green: #00e676;
    --accent-red: #ff4444;
    --accent-amber: #ffab40;
    --accent-purple: #b388ff;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-sans: 'IBM Plex Sans', sans-serif;
}

/* ── Global app background ── */
.stApp {
    background-color: var(--bg-primary);
    font-family: var(--font-sans);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] .stMetric label {
    color: var(--text-muted) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-size: 1.4rem !important;
    font-weight: 600;
}

/* ── Main content area ── */
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}

/* ── Title ── */
h1 {
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 0.25rem !important;
}

/* ── Caption / subtitle ── */
[data-testid="stCaptionContainer"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-accent) !important;
    border-radius: 6px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] * {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    gap: 0;
    padding: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    margin-bottom: 0.5rem !important;
    padding: 0.75rem 1rem !important;
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid var(--accent-blue) !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    border-left: 3px solid var(--accent-green) !important;
}
[data-testid="stChatMessage"] p {
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 6px !important;
}
[data-testid="stChatInput"] textarea {
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Status boxes (thinking indicator) ── */
[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
}

/* ── Success / error / info alerts ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    padding: 0.5rem 0.75rem !important;
}

/* ── Expander (L1 cache view in sidebar) ── */
[data-testid="stExpander"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border-accent);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--accent-blue) !important;
}

/* ── Sidebar section headers ── */
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent-purple) !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
    margin-top: 1rem !important;
}

/* ── Telemetry cards ── */
.telemetry-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.4rem;
}
.telemetry-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}
.telemetry-value {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent-blue);
    margin-top: 0.1rem;
}
.status-tag {
    font-size: 0.55rem;
    background: var(--accent-green);
    color: #000;
    padding: 1px 5px;
    border-radius: 3px;
    vertical-align: middle;
    margin-left: 4px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

    _init_session_state()
    _render_sidebar()

    st.markdown("""
<div style="display:flex; align-items:baseline; gap:1rem; 
     border-bottom:1px solid #2a2a3a; padding-bottom:0.75rem; 
     margin-bottom:1rem;">
  <span style="font-family:'IBM Plex Mono',monospace; 
               font-size:1.4rem; font-weight:600; 
               color:#e8e8f0; letter-spacing:-0.02em;">
    PrimaLLM
  </span>
  <span style="font-family:'IBM Plex Mono',monospace; 
               font-size:0.65rem; color:#555570; 
               text-transform:uppercase; letter-spacing:0.1em;">
    Neural Memory Management Unit &nbsp;·&nbsp; 
    L1 Set Cache &nbsp;·&nbsp; L2 Source Graph &nbsp;·&nbsp; 
    L3 Verified Wiki
  </span>
</div>
""", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("Upload source PDF", type=["pdf"])
    if uploaded_pdf is not None and st.session_state.loaded_pdf_name != uploaded_pdf.name:
        with st.spinner("Ingesting PDF into L2 and populating L1 facts..."):
            triple_count, node_count = process_pdf(uploaded_pdf)
            st.session_state.loaded_pdf_name = uploaded_pdf.name
            # Invalidate cached graph rendering so the map tab re-renders
            st.session_state.graph_rendered_for = None
        st.success(f"Loaded {uploaded_pdf.name}: {triple_count} triples, {node_count} L2 graph nodes")

    # ── Tabbed layout: Chat + Knowledge Map ──
    tab_chat, tab_map = st.tabs(["Chat", "Knowledge Map"])

    # ── Chat tab ──
    with tab_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question about the loaded document")
        if prompt:
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
                        is_clean = _run_sentinel_writeback(final_answer)
                        if not is_clean:
                            final_answer = (
                                "🚨 NLI SENTINEL BLOCK: My policy engine attempted to answer this, "
                                "but the local DeBERTa-v3 verification failed. The source document "
                                "does not support this claim."
                            )
                        status.update(label="NMMU complete", state="complete")
                except Exception as exc:
                    final_answer = f"Error: {exc}"
                    st.error(final_answer)
                    st.session_state.telemetry["l1_status"] = "error"

                answer_placeholder.markdown(final_answer)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # ── Knowledge Map tab ──
    with tab_map:
        source_graph = st.session_state.source_graph
        if source_graph is None:
            st.info("Upload a PDF to visualize the L2 Knowledge Map.")
        else:
            # Only re-render if the source graph has changed
            current_pdf = st.session_state.loaded_pdf_name
            if st.session_state.graph_rendered_for != current_pdf:
                st.session_state.graph_html = render_graph_visual(source_graph)
                st.session_state.graph_rendered_for = current_pdf

            components.html(st.session_state.graph_html, height=800, scrolling=True)


if __name__ == "__main__":
    main()