# PrimaLLM

PrimaLLM is now organized as a two-path research repo:

- `shared/` contains the local spaCy triple extractor and `KnowledgeTriple`
- `caveman/` contains Path A compression, PageRank scoring, and L1 caching
- `sentinel/` contains Path B verification, source graphs, and L3 persistence

## Current Layout

- `shared/extractor.py` - single source of truth for spaCy SVO extraction
- `shared/triple.py` - shared `KnowledgeTriple` dataclass
- `caveman/main.py` - local compression demo entry point
- `caveman/core/graph.py` - NetworkX graph and PageRank scoring
- `caveman/core/cache.py` - token-budgeted L1 cache
- `sentinel/core/source_graph.py` - source-of-truth graph construction
- `sentinel/core/verifier.py` - deterministic NLI verification
- `sentinel/core/wiki_storage.py` - L3 JSON persistence
- `sentinel/core/wiki.json` - local verified fact store

## Setup

1. Create a local `.env` file in the project root if you need model downloads or external services.
2. Install the package dependencies required by the path you are working on.
3. For extraction, make sure `en_core_web_sm` is installed in your spaCy environment.

## Run

Path A demo:

```powershell
.\.venv\Scripts\python.exe -m caveman.main
```

Path B benchmark scaffold:

```powershell
.\.venv\Scripts\python.exe -m sentinel.benchmark.run_verification_benchmark
```

## Notes

- The legacy root-level prototype files have been retired in favor of the package layout.
- Sentinel owns the JSON store now; the default file lives at `sentinel/core/wiki.json`.
