# Caveman

Path A for local-first compression.

## Scope

- spaCy-based SVO extraction lives in `shared/extractor.py`
- graph construction and PageRank live in `caveman/core/graph.py`
- token-budgeted L1 caching lives in `caveman/core/cache.py`
- demo entry point: `python -m caveman.main [optional_file_path]`
- `caveman/core/compressor.py` supports OpenAI or local Qwen generation via `USE_LOCAL_SLM`

## Run

Default run using `source_material.txt`:

```powershell
.\.venv\Scripts\python.exe -m caveman.main
```

Custom input file:

```powershell
.\.venv\Scripts\python.exe -m caveman.main my_input.txt
```

Local Qwen mode:

```powershell
$env:USE_LOCAL_SLM="true"; .\.venv\Scripts\python.exe -m caveman.main
```

OpenAI mode:

```powershell
$env:USE_LOCAL_SLM="false"; .\.venv\Scripts\python.exe -m caveman.main
```

## Research Question

Can local semantic compression preserve enough signal for downstream Q&A while reducing context size?
