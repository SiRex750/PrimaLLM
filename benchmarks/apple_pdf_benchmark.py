import pymupdf4llm, re, json, os, datetime
from shared.extractor import extract_source_triples
from caveman.core.graph import build_graph, merge_similar_nodes
from caveman.core import rank_triples_by_importance, L1Cache, generate_caveman_prose
from sentinel.core import build_source_graph, verify_claim
from shared.triple import KnowledgeTriple
from sentence_transformers import SentenceTransformer
from caveman.benchmark.metrics import count_tokens, sdpt as calculate_sdpt
from caveman.benchmark.run_benchmark import _check_accuracy, ask_judge
from app import _build_partitioned_messages

STOP_MARKERS = (
    '## references', '## further reading', '## see also',
    '## external links', '## bibliography', '# references',
)

def ingest_pdf(pdf_path: str):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    triples = []
    all_sentences = []
    
    for page_chunk in md_text:
        page_text = page_chunk.get('text', '')
        page_lower = page_text.lower()
        for marker in STOP_MARKERS:
            idx = page_lower.find(marker)
            if idx != -1:
                page_text = page_text[:idx]
                break
        page_text = re.sub(
            r'==> picture \[.*?\] intentionally omitted <==', 
            '', page_text
        )
        page_text = re.sub(r'^#{1,6}\s+.*$', '', page_text, 
                           flags=re.MULTILINE)
        page_text = re.sub(r'\*\*|__|\*|_', '', page_text)
        page_text = re.sub(r'\[\s*\d+\s*\]', '', page_text)
        page_text = re.sub(r'\s+', ' ', page_text).strip()
        if page_text and len(page_text) > 30:
            page_triples = extract_source_triples(page_text)
            triples.extend(page_triples)
            sentences = [
                s.strip() for s in 
                re.split(r'(?<=[.!?])\s+', page_text)
                if len(s.strip()) > 20
            ]
            all_sentences.extend(sentences)
    
    graph = build_graph(triples)
    # Get raw node count before merge
    raw_node_count = graph.number_of_nodes()
    
    merged_graph, merged_count = merge_similar_nodes(
        graph, embedder, threshold=0.82
    )
    final_node_count = merged_graph.number_of_nodes()

    source_graph = build_source_graph(
        triples, embedder=embedder, source_sentences=all_sentences
    )
    
    ranked = rank_triples_by_importance(triples)
    
    # Scale L1 budget proportionally to document size
    # Minimum 150 tokens, maximum 600 tokens
    # Target: retain roughly 12% of document tokens in L1
    raw_text = ' '.join(all_sentences)
    raw_tokens = count_tokens(raw_text)
    
    dynamic_facts_budget = max(150, min(600, raw_tokens // 8))
    cache = L1Cache(budgets={
        "facts": dynamic_facts_budget, 
        "scratch": 100
    })
    print(f"Dynamic L1 budget: {dynamic_facts_budget} tokens "
          f"(document: {raw_tokens} tokens)")

    for triple, score in ranked:
        cache.route_triple(triple, pagerank_score=score)
    
    return cache, source_graph, raw_tokens, len(triples), raw_node_count, final_node_count, dynamic_facts_budget

APPLE_QA_CASES = [
    {
        "question": "Where does the apple tree originally come from?",
        "expected": "Kazakhstan",
        "domain": "botany"
    },
    {
        "question": "What is the scientific name of the wild ancestor of apple trees?",
        "expected": "Malus sieversii",
        "domain": "botany"
    },
    {
        "question": "What percentage of global apple production does China account for in 2013?",
        "expected": "49%",
        "domain": "production"
    },
    {
        "question": "How many known variants of apples are there?",
        "expected": "10000",
        "domain": "botany"
    },
    {
        "question": "What chemical in apple seeds can release cyanide?",
        "expected": "amygdalin",
        "domain": "botany"
    },
    {
        "question": "When was the first apple orchard in North America established?",
        "expected": "1625",
        "domain": "history"
    },
    {
        "question": "Who wrote the Prose Edda that mentions the goddess Idunn?",
        "expected": "Snorri Sturluson",
        "domain": "culture"
    },
    {
        "question": "What is the scientific name of the cultivated apple species?",
        "expected": "Malus domestica",
        "domain": "botany"
    },
    {
        "question": "What was the total worldwide apple production in 2013?",
        "expected": "90.8 million tonnes",
        "domain": "production"
    },
    {
        "question": "In which plant family are apples classified?",
        "expected": "Rosaceae",
        "domain": "botany"
    },
]

def ask_with_l2_fallback(question, cache, source_graph, embedder, raw_tokens):
    from caveman.benchmark.run_benchmark import count_tokens
    import ollama, json, re
    
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    
    # --- ARCHITECTURAL BYPASS & DYNAMIC CONTEXT SLICER ---
    active_facts = [entry.text for entry in cache.set_facts.values()]
    forced_facts = []

    if not active_facts:
        from app import query_l2_memory
        l2_result = query_l2_memory(question, question, source_graph)
        if l2_result:
            forced_facts = [l2_result]
    else:
        from app import get_cross_encoder
        ce = get_cross_encoder()
        pairs = [[question, f] for f in active_facts]
        scores = ce.predict(pairs)
        scored_facts = sorted(zip(active_facts, scores), key=lambda x: x[1], reverse=True)

        if scored_facts[0][1] < 0.0:
            from app import query_l2_memory
            l2_result = query_l2_memory(question, question, source_graph)
            if l2_result:
                forced_facts = [l2_result]
        else:
            forced_facts = [f for f, s in scored_facts if s > 0.0][:3]

    # Build context from L1 with filtered/forced facts
    messages = _build_partitioned_messages(cache, question, forced_facts=forced_facts)
    
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.0, "num_predict": 200}
    )
    content = response['message']['content'].strip()
    
    # Check for tool call (fallback if bypass wasn't used)
    match = re.search(r'"keyword":\s*"([^"]+)"', content)
    if match and not forced_facts:
        keyword = match.group(1)
        from app import query_l2_memory
        l2_result = query_l2_memory(question, keyword, source_graph)
        
        if l2_result:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": 
                f"Additional context from memory: {l2_result}\n"
                f"Now answer in plain text: {question}"})
            response2 = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"temperature": 0.0, "num_predict": 200}
            )
            answer = response2['message']['content'].strip()
        else:
            answer = "INSUFFICIENT DATA"
    else:
        answer = content

    caveman_tokens = count_tokens(cache.as_context_text())
    return answer, caveman_tokens

def main():
    pdf_path = "Apple-1.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print(f"Ingesting {pdf_path}...")
    cache, source_graph, raw_tokens, total_triples, raw_nodes, final_nodes, dynamic_budget = ingest_pdf(pdf_path)
    
    results = []
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    for case in APPLE_QA_CASES:
        question = case["question"]
        expected = case["expected"]
        domain = case["domain"]
        
        print(f"\nProcessing [{domain}] question: {question}")
        
        # Query-aware re-ranking
        cache.rerank_facts_for_query(question, embedder)
        
        # Get answer with L2 fallback
        answer, caveman_tokens = ask_with_l2_fallback(
            question, cache, source_graph, embedder, raw_tokens
        )
        is_correct = _check_accuracy(answer, expected)
        
        results.append({
            "question": question,
            "expected": expected,
            "answer": answer,
            "is_correct": is_correct,
            "domain": domain,
            "raw_tokens": raw_tokens,
            "caveman_tokens": caveman_tokens,
            "reduction": (raw_tokens - caveman_tokens) / raw_tokens * 100 if raw_tokens > 0 else 0,
            "baseline_sdpt": raw_tokens / total_triples if total_triples > 0 else 0,
            "caveman_sdpt": calculate_sdpt(len(cache.active_facts), caveman_tokens) if len(cache.active_facts) > 0 else 0
        })
    
    # Statistics
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    avg_reduction = sum(r["reduction"] for r in results) / len(results)
    avg_sdpt_imp = sum(r["baseline_sdpt"] - r["caveman_sdpt"] for r in results) / len(results)
    
    domain_stats = {}
    for r in results:
        d = r["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"correct": 0, "total": 0}
        domain_stats[d]["total"] += 1
        if r["is_correct"]:
            domain_stats[d]["correct"] += 1
    
    print("\n" + "="*100)
    print("APPLE PDF END-TO-END BENCHMARK REPORT (with L2 Fallback)")
    print("="*100)
    print(f"Overall Accuracy: {accuracy*100:.1f}%")
    print(f"Avg Compression: {avg_reduction:.1f}%")
    print(f"Avg SDpT Improvement: {avg_sdpt_imp:.2f} tokens/ACU")
    print(f"L1 Budget Used: {dynamic_budget} tokens")
    print(f"Graph Nodes: {raw_nodes} (raw) -> {final_nodes} (merged)")
    print("-" * 100)
    
    for d, stats in domain_stats.items():
        print(f"Domain [{d:<10}]: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.1f}%)")
    
    print("-" * 100)
    for i, r in enumerate(results, 1):
        status = "PASS" if r["is_correct"] else "FAIL"
        print(f"Case {i:<2}: {status} | {r['question']}")
        print(f"   Expected: {r['expected']}")
        print(f"   Got:      {r['answer'][:100].strip()}...")
        print()
        
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pdf": pdf_path,
        "accuracy": accuracy,
        "avg_reduction": avg_reduction,
        "avg_sdpt_improvement": avg_sdpt_imp,
        "raw_nodes": raw_nodes,
        "final_nodes": final_nodes,
        "l1_budget_used": dynamic_budget,
        "document_raw_tokens": raw_tokens,
        "compression_target": f"{100 * dynamic_budget / raw_tokens:.1f}%",
        "domain_stats": domain_stats,
        "cases": results
    }
    
    output_path = "benchmarks/apple_pdf_benchmark_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"[SUCCESS] Detailed results saved to {output_path}")

if __name__ == "__main__":
    main()
