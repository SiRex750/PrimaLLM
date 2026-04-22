# Sentinel

Path B for deterministic verification and L3 persistence.

## Scope

- `sentinel/core/source_graph.py` builds the source-of-truth graph from ingested documents
- `sentinel/core/verifier.py` runs local NLI verification with `cross-encoder/nli-deberta-v3-small`
- `sentinel/core/wiki_storage.py` owns JSON persistence for verified facts

## Entry Point

- `python -m sentinel.benchmark.run_verification_benchmark`
