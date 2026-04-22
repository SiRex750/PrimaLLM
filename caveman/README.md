# Caveman

Caveman is Path A of PrimaLLM: a local-first compression layer that keeps high-value facts under a strict context budget.

## What Caveman Does

- Extracts knowledge triples from raw text using the shared spaCy extractor.
- Builds a directed graph over entities and relations.
- Uses PageRank to estimate triple importance.
- Maintains an L1 cache that evicts low-scoring facts when token budget is exceeded.
- Produces compact prose from the surviving facts.

## Pipeline

1. Read input text (`source_material.txt` by default or custom file path).
2. Extract triples with `shared/extractor.py`.
3. Rank triples with `caveman/core/graph.py`.
4. Insert ranked facts into `L1Cache` (`caveman/core/cache.py`) with token-budget trimming.
5. Generate final compressed prose with `caveman/core/compressor.py`.

## Dependencies

Install required packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r caveman\requirements.txt
```

Install the required spaCy model:

```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Optional for OpenAI mode:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Running Caveman

Default run (uses root `source_material.txt` if present):

```powershell
.\.venv\Scripts\python.exe -m caveman.main
```

Use a custom input file:

```powershell
.\.venv\Scripts\python.exe -m caveman.main my_input.txt
```

Force local SLM generation (Qwen):

```powershell
$env:USE_LOCAL_SLM="true"; .\.venv\Scripts\python.exe -m caveman.main
```

Force OpenAI generation:

```powershell
$env:USE_LOCAL_SLM="false"; .\.venv\Scripts\python.exe -m caveman.main
```

## Example Output Structure

`caveman.main` prints three sections:

```text
(1) Total Triples Extracted
(2) Triples Kept After Cache Eviction
(3) Final Compressed Caveman Text
```

This gives a direct view of compression effectiveness and what facts survived eviction.

## Key Modules

- `caveman/core/graph.py`
- builds relation graph
- computes PageRank
- ranks triples by node centrality sum

- `caveman/core/cache.py`
- stores `KnowledgeTriple` entries
- estimates token usage with `tiktoken`
- evicts lowest PageRank entries to honor `token_budget`

- `caveman/core/compressor.py`
- constructs a constrained compression prompt
- generates prose via local Qwen or OpenAI
- applies post-processing to keep concise Caveman-style output

## Benchmark Scaffold

Run the scaffold:

```powershell
.\.venv\Scripts\python.exe -m caveman.benchmark.run_benchmark
```

Current scaffold reports dataset count plus example SDpT and token metrics, and is intended to be extended with task-specific evaluation logic.
