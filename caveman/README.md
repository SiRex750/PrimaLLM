# Caveman

Path A for local-first compression.

## Scope

- spaCy-based SVO extraction lives in `shared/extractor.py`
- graph construction and PageRank live in `caveman/core/graph.py`
- token-budgeted L1 caching lives in `caveman/core/cache.py`
- demo entry point: `python -m caveman.main`

## Research Question

Can local semantic compression preserve enough signal for downstream Q&A while reducing context size?
