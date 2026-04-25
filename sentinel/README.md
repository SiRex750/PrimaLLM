# Sentinel — NLI Write-Back Verification Gate

Prevents hallucinated facts from entering persistent memory using local Natural Language Inference verification.

## What Sentinel Does

- Extracts claim triples from LLM-generated text using GLiNER-relex.
- Builds a source-of-truth SourceGraph from ingested document triples (spaCy).
- Computes SHA-256 per-triple checksums and a master document checksum.
- Verifies each claim against the source graph using DeBERTa NLI (cross-encoder/nli-deberta-v3-small).
- Writes only ENTAILMENT-verified facts to L3 SQLite storage; discards CONTRADICTION and NEUTRAL claims.

## The Verification Pipeline

Step 1 — Claim extraction: GLiNER-relex extracts (subject, verb, object) triples from LLM output.

Step 2 — Premise construction: `_build_localized_premise()` finds the highest-overlap source sentence (by keyword matching using spaCy POS tags) and up to 2 supporting source triples.

Step 3 — NLI scoring: DeBERTa cross-encoder scores (premise, hypothesis) pair.

Step 4 — Label resolution: entailment → CLEAN (write to L3), contradiction/neutral → DIRTY (discard).

Step 5 — GLiNER-extracted triples require 0.85+ entailment confidence (higher threshold for non-deterministic extraction).

## The Semantic CRC

- Source graph computed before LLM inference (pre-computation).
- SHA-256 checksum over sorted triple text = deterministic document fingerprint.
- Any LLM output that contradicts the checksum-verified source is flagged.
- Analogous to CRC error detection: inject known redundancy (source graph), verify at output (NLI gate).
- This is the "dirty bit" concept from computer architecture applied to LLM memory.

## Benchmark Results

Benchmark: Apple Wikipedia article (Simple English), 30 cases
5 difficulty tiers: easy, medium, hard, adversarial, edge
4 domains: botany, history, production, culture

### Overall Metrics

| Metric | Value |
| :--- | :--- |
| Accuracy | 63.3% (19/30) |
| Precision | 63.6% |
| Recall | 50.0% |
| F1 Score | 0.56 |
| TP / TN / FP / FN | 7 / 12 / 4 / 7 |

### By Domain Breakdown

| Domain | Pass Rate | Percentage |
| :--- | :--- | :--- |
| History | 4/4 | 100% |
| Botany | 11/17 | 65% |
| Production | 3/6 | 50% |
| Culture | 1/3 | 33% |

### By Difficulty Breakdown

| Difficulty | Pass Rate | Percentage |
| :--- | :--- | :--- |
| Easy | 8/11 | 73% |
| Medium | 6/10 | 60% |
| Hard | 4/5 | 80% |
| Edge | 1/4 | 25% |

## Key Engineering Findings

- **Finding 1**: DeBERTa achieves 98-99% entailment confidence when given clean, single-sentence premises. Premise noise (multiple competing facts) drops confidence to near-zero.
- **Finding 2**: Premise construction uses ranked sentence retrieval — the single highest-overlap source sentence is selected, not top-N. This "surgical context" approach outperformed broader retrieval.
- **Finding 3**: spaCy verb lemmatisation ("come" vs "comes") caused keyword overlap failures. Fixed by storing original verb form in KnowledgeTriple and using lemma-aware matching only in the retrieval layer.
- **Finding 4**: NLI models correctly return NEUTRAL for claims that add information beyond the source (e.g. "China grew 49% of apple production" when source only says "China grew 49% of the total"). This is correct behaviour — the gate should not confirm over-specified claims.

## Module Reference

- **sentinel/core/source_graph.py**: Implements `SourceGraph` dataclass, `build_source_graph()`, SHA-256 checksums, and `source_sentences` storage.
- **sentinel/core/verifier.py**: Contains `verify_claim()`, `_build_localized_premise()`, `_get_claim_keywords()`, and `_resolve_label()`.
- **sentinel/core/wiki_storage.py**: Manages SQLite WAL-mode storage, `save_verified_fact()`, `fetch_clean_facts()`, and `fetch_clean_facts_by_similarity()`.
- **sentinel/benchmark/run_verification_benchmark.py**: Original 6-case proof-of-concept benchmark.
- **benchmarks/sentinel_apple_benchmark.py**: 30-case Apple Wikipedia benchmark with full metrics.

## Running the benchmarks

**6-case benchmark**:
```bash
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe -m sentinel.benchmark.run_verification_benchmark
```

**30-case Apple benchmark**:
```bash
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe sentinel_apple_benchmark.py
```

Results saved to: `benchmarks/sentinel_benchmark_results.json`

## Known Limitations

- **Edge cases (1/4 passing)**: multi-sentence anaphora resolution (pronoun "they" requiring cross-sentence reference) is beyond single-sentence NLI capability.
- **Numeric claims requiring multi-sentence context aggregation**: Connecting a year in sentence 1 to a statistic in sentence 2 scores as neutral.
- **Future work**: Integration of a larger NLI model (DeBERTa-large) or domain-specific fine-tuning to improve recall without sacrificing precision.

## Using Sentinel as a library

```python
from shared.extractor import extract_claim_triples, extract_source_triples
from sentinel.core.source_graph import build_source_graph
from sentinel.core.verifier import verify_claim
from sentinel.core.wiki_storage import save_verified_fact
import re

source_text = "Neil Armstrong piloted the Lunar Module Eagle."
sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', source_text) if len(s.strip()) > 20]
source_graph = build_source_graph(extract_source_triples(source_text), source_sentences=sentences)

for triple in extract_claim_triples("Armstrong piloted Eagle module"):
    result = verify_claim(triple, source_graph, source_sentences=sentences)
    if result.is_verified:
        save_verified_fact(triple)
```
