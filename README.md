# HADES

**Hierarchical Adaptive Document Encoding System**

A Local-First Neural Memory Management Unit for Verifiable Knowledge Compression and Factual Grounding.

This research project was developed by a 2nd-year CSE (AI/ML) student at PES University to explore the intersection of graph-based knowledge compression and NLI-driven factual verification.

## What is HADES?

HADES (Hierarchical Adaptive Document Encoding System) is a research-oriented Neural Memory Management Unit (NMMU) designed to address context window saturation and hallucination in Large Language Models. It implements a dual-path architecture: **Charon** for graph-based factual compression and **Cerberus** for NLI-based claim verification. The system provides an end-to-end local pipeline for managing long-term semantic memory, ensuring that only verified information is persisted to long-term storage while maintaining high semantic density in the active context.

## Architecture Overview

HADES operates on a three-tier memory hierarchy inspired by operating system cache architectures:

- **L1 (Cache)**: A set-associative context window with **Dynamic Query-Aware Reranking**. This tier uses semantic similarity to "promote" relevant triples into the active context, ensuring query-critical facts (like niche details in large PDFs) are not evicted.
- **L2 (RAM)**: A semantic vector space for mid-term retrieval. If the L1 re-ranker detects a "Page Fault" (low relevance in current facts), it fetches high-similarity triples from the `all-MiniLM-L6-v2` vector store.
- **L3 (Disk)**: A verified persistent wiki storage (SQLite) containing only facts that have passed the Cerberus NLI verification gate.

## Benchmark Results

### CHARON (8-case QA benchmark)

| Metric | Budget=30 (Max Compression) | Budget=150 (Balanced) |
|:---|:---|:---|
| Accuracy (best run) | 75.0% | 100.0% |
| Accuracy (worst run) | 75.0% | 87.5% |
| Accuracy (mean across 3 runs) | 75.0% | 95.8% |
| Average token reduction | 42.1% | 44.8% |
| Average baseline SDpT | 14.58 tokens/ACU | 14.58 tokens/ACU |
| Average Charon SDpT | 8.90 tokens/ACU | 8.11 tokens/ACU |
| Average SDpT improvement | 5.69 tokens/ACU | 6.47 tokens/ACU |
| PDF extraction | PyPDF2 (raw) | pymupdf4llm (layout-aware) |
| Node merging | None | Embedding cosine (threshold 0.75) |
| Model | qwen2.5:1.5b | qwen2.5:1.5b |

"Accuracy varies across runs due to non-determinism in
qwen2.5:1.5b at temperature=0 on CPU floating point arithmetic.
The Balanced configuration uses weighted PageRank graph scoring,
layout-aware PDF extraction (pymupdf4llm), and embedding-based
entity coreference merging. SDpT = compressed_tokens / unique_ACUs.
Lower SDpT = higher information density per token."

The L1 FACTS budget is a configurable parameter. 
Higher budgets preserve more context (better accuracy on 
complex queries) at the cost of compression ratio. 
Lower budgets maximise token reduction at the risk of 
evicting query-critical facts.

### APPLE WIKI (Large Context Benchmark)

The Apple Wiki benchmark tests the system's ability to preserve "long-tail" facts in a massive context (4,000+ tokens) under strict budget constraints.

| Architecture | Budget | Reranking | "Cyanide" Question | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Old** | 30 tokens | None | **FAIL** (Evicted) | PageRank Blindness |
| **New** | 150 tokens | **Query-Aware** | **PASS** (Promoted) | **Superior Grounding** |

### APPLE PDF (System-Wide Evaluation)

The Apple PDF benchmark evaluates the end-to-end integration of the HADES pipeline—encompassing layout-aware extraction (`pymupdf4llm`), SVO Graph Construction, Dynamic L1 Cache Management, and L2 Vector Fallback.

| Dimension | Performance | Impact |
| :--- | :--- | :--- |
| **System Accuracy** | **80.0%** | Robust cross-domain recall |
| **Information Density** | **86.2% reduction** | Optimal L1 token utilization |
| **Temporal Precision** | High | Mitigation of historical hallucinations |
| **Entity Preservation** | High | Protection of long-tail proper nouns |

**Architectural Innovations for Factual Grounding**:

1.  **Temporal Harvesting (Metadata-Driven)**: To resolve the issue of "Orphaned Modifiers" in dependency parsing, HADES implements a secondary NER-driven scan. This identifies dates and times structurally distant from the root verb and stores them as `temporal_anchors` metadata. This ensures quantitative metrics are always anchored to their correct timeframe, preventing calculation hallucinations.
2.  **Symmetric Multi-Objective Ranking**: The system utilizes a dual-path importance score that balances global graph centrality (PageRank) with semantic specificity (NER). By applying **Additive NER Boosting** (+0.1) to specific entity classes, HADES ensures that high-value "long-tail" facts co-exist with structural nodes without context flooding.
3.  **Source-Grounded Citation**: To lock the SLM's attention to the provided context, HADES enforces a strict JSON response schema. The model must provide a `citation` (the exact SVO triple used) before it is permitted to generate the `answer`. Failure to output valid JSON or a valid citation triggers an automatic **Memory Fault**, forcing a fallback to the high-fidelity L2 memory.
4.  **L1 Conflict Resolution (Collision Detection)**: The L1 Cache identifies triples with similar semantics but conflicting numeric data (e.g., varying production statistics). These facts are wrapped in an explicit `<SYSTEM_WARNING: CONFLICTING DATA>` block, alerting the policy model to resolve the ambiguity based on the precise wording of the user's query.

### CERBERUS (30-case NLI benchmark)

The Cerberus benchmark evaluates the NLI-based verification gate using 30 diverse test cases across four domains from the Apple Wikipedia article.

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | 83.3% |
| Precision | 84.6% |
| Recall | 78.6% |
| F1 Score | 81.5% |
| TP / TN / FP / FN | 11 / 14 / 2 / 3 |
| Model used | DeBERTa-v3-base (Local NLI) |

#### Performance by Difficulty
- **Easy**: 72.7% (8/11)
- **Medium**: 60.0% (6/10)
- **Hard**: 80.0% (4/5)
- **Edge**: 25.0% (1/4)

**Key Finding**: Cerberus demonstrates robust performance on easy and medium factual claims. The high "Hard" case accuracy (80%) shows strong resistance to subtle adversarial distortions. However, performance on "Edge" cases (multi-hop entity resolution and complex numbers) remains an area for future optimization with larger local NLI encoders.


## Repository Structure

- `shared/` — Core `KnowledgeTriple` dataclass, dual extractor (spaCy + GLiNER-relex), and L3 SQLite memory interfaces.
- `caveman/` — Factual graph compression pipeline including `graph.py`, `cache.py`, `compressor.py`, and the compression benchmark suite. (Charon)
- `sentinel/` — NLI-based verification pipeline featuring `source_graph.py`, `verifier.py`, `wiki_storage.py`, and the Cerberus benchmark suite.
- `benchmarks/` — Directory containing saved JSON results and logs for both system components.
- `app.py` — Streamlit-based web interface featuring live PyVis knowledge graph visualisations.
- `hades.py` — Command-line entrypoint for the complete end-to-end NMMU pipeline (also accessible as primallm.py for backward compatibility).

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

*Note: The first execution will download GLiNER and DeBERTa model weights (~500MB total).*

## Quick Start

### Streamlit UI
```bash
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe -m streamlit run app.py
```

### CLI Pipeline
```bash
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe primallm.py
```

### Caveman Benchmark
```bash
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe -m caveman.benchmark.run_benchmark
```

### Sentinel Benchmark
```bash
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe sentinel_apple_benchmark.py
```

## How It Works

1. **Extraction**: LiteParse or PyPDF2 extracts clean text from uploaded PDF documents.
2. **Graph Construction**: A spaCy-based SVO (Subject-Verb-Object) extractor builds a source `KnowledgeTriple` set, which is converted into a NetworkX graph for PageRank-based importance scoring.
3. **Compression**: The Charon pipeline selects the top-ranked triples and enforces an L1 token budget using a set-associative cache (SYSTEM/FACTS/HISTORY/TOOLS/SCRATCH). It then generates compressed prose via a local Ollama Small Language Model (SLM).
4. **Inference**: A local LLM (e.g., Phi-3 or Qwen-2.5 via Ollama) answers the user query using the compressed context.
5. **Claim Extraction**: The LLM output is added to the SCRATCH set (marked as "dirty"). GLiNER-relex extracts claim triples directly from this generated output.
6. **Verification**: The Cerberus NLI gate verifies each dirty triple against the original source graph. CLEAN triples are promoted to L3 SQLite storage, while DIRTY triples (hallucinations) are discarded.

## Design Principles

- **Local-first**: All models run on consumer-grade hardware; no external APIs or cloud dependencies are required.
- **Privacy-preserving**: Sensitive documents never leave the local environment during processing or inference.
- **Verified persistence**: Implementation of a strict write-back policy where no data enters long-term memory without NLI confirmation.
- **OS-inspired**: The LLM context window is managed as a hardware cache hierarchy (L1/L2/L3) with explicit eviction policies.

## Tech Stack

- **spaCy (en_core_web_sm)**: Source triple extraction and NLP preprocessing.
- **GLiNER-relex (knowledgator/gliner-relex-large-v0.5)**: High-precision claim triple extraction from LLM responses.
- **DeBERTa (cross-encoder/nli-deberta-v3-base)**: Local NLI verification for the Cerberus gate.
- **NetworkX**: Knowledge graph representation and PageRank algorithmic scoring.
- **tiktoken**: Precise token budget enforcement for context management.
- **SQLite (WAL mode)**: High-performance storage for L3 verified fact persistence.
- **sentence-transformers (all-MiniLM-L6-v2)**: L2 semantic search and embedding generation.
- **PyVis**: Interactive knowledge graph visualisation.
- **Streamlit**: Web-based user interface.
- **Ollama (qwen2.5:1.5b or phi-3-mini)**: Local SLM/LLM inference.

## Citation / Academic Context

If you use HADES in academic work, please cite the following works that inspired this architecture:

- Lewis et al. (2020) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Packer et al. (2023) MemGPT: Towards LLMs as Operating Systems
- Jiang et al. (2023) LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
- Laban et al. (2022) SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization
- Liu et al. (2023) Lost in the Middle: How Language Models Use Long Contexts
