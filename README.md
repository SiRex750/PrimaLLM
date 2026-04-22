# PrimaLLM

PrimaLLM is a local-first research codebase for factual compression and factual verification.

It is intentionally split into two paths:

- Path A (Caveman): compress source knowledge into a compact context layer.
- Path B (Sentinel): verify claims against a source-of-truth graph before persistence.

## Design Goals

- Keep extraction local and deterministic with spaCy (`en_core_web_sm`).
- Keep verification local and deterministic with DeBERTa NLI (`cross-encoder/nli-deberta-v3-small`).
- Share core data structures and extraction logic through `shared/`.
- Avoid verification against prior AI output; verify against ingested source facts.

## Repository Structure

- `shared/`
- `shared/triple.py`: shared `KnowledgeTriple` dataclass.
- `shared/extractor.py`: single extraction implementation for both paths.
- `caveman/`
- `caveman/main.py`: end-to-end compression demo entrypoint.
- `caveman/core/graph.py`: directed graph + PageRank ranking.
- `caveman/core/cache.py`: token-budget L1 cache with eviction.
- `caveman/core/compressor.py`: compressed prose generation (OpenAI or local Qwen).
- `caveman/benchmark/`: benchmark scaffold for compression experiments.
- `sentinel/`
- `sentinel/main.py`: end-to-end verification demo entrypoint.
- `sentinel/core/source_graph.py`: source graph + per-triple and master checksums.
- `sentinel/core/verifier.py`: local NLI verification logic.
- `sentinel/core/wiki_storage.py`: JSON persistence for verified facts.
- `sentinel/benchmark/`: benchmark scaffold for verification experiments.

## Prerequisites

- Python 3.10+ recommended.
- A virtual environment.
- For Caveman OpenAI mode only: `OPENAI_API_KEY` in `.env`.

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r caveman\requirements.txt
```

Install spaCy English model:

```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Optional environment file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Quick Start

Run Caveman (Path A):

```powershell
.\.venv\Scripts\python.exe -m caveman.main
```

Run Caveman with local SLM generation:

```powershell
$env:USE_LOCAL_SLM="true"; .\.venv\Scripts\python.exe -m caveman.main
```

Run Sentinel (Path B):

```powershell
.\.venv\Scripts\python.exe -m sentinel.main
```

Run benchmark scaffolds:

```powershell
.\.venv\Scripts\python.exe -m caveman.benchmark.run_benchmark
.\.venv\Scripts\python.exe -m sentinel.benchmark.run_verification_benchmark
```

## How It Works

1. `shared/extractor.py` converts raw text into `(subject, verb, object)` triples.
2. Caveman ranks triples with PageRank and trims them to a token budget.
3. Caveman generates compact prose from retained triples.
4. Sentinel builds a source graph and computes:
- per-triple SHA-256 checksums
- a master SHA-256 checksum from sorted triple text
5. Sentinel runs local NLI to accept or reject claims.
6. Verified claims are persisted through `sentinel/core/wiki_storage.py`.

## Example Flow

Sentinel demo output includes lines like:

```text
--- SENTINEL VERIFICATION ORACLE ---
1. Extracting Source Graph (Semantic Checksum)...
	Nodes: 6 | Edges: 3
	Master Checksum (SHA-256): 08f8c85e6d9e5063...
...
	Result: ✅ VALID (...)
...
	Result: ❌ INVALID (...)
```

This demonstrates the intended guardrail behavior: accepted claims are persisted, rejected claims are blocked.

## Notes

- `sentinel/core/wiki.json` is runtime persistence and can be recreated automatically.
- First run may download local model artifacts (spaCy, transformers models).
- If Hugging Face warns about unauthenticated requests, setting `HF_TOKEN` is optional but can improve download limits.
