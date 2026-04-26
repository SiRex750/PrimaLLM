# HADES: L1/L2 Memory Management Analysis & Fix Plan

## 1. Problem Statement: The "Context Saturation" Trade-off
The HADES system uses a tiered memory architecture (L1 Context -> L2 Vector RAM -> L3 SQLite Disk). Current benchmarking on the `Apple-1.pdf` reveals a critical failure mode: **High-Value Data Displacement.**

### Symptoms:
- **Accuracy Loss**: Accuracy dropped from 60% to 50% when we introduced a hard boost for Proper Nouns.
- **Entity Crowding**: Specific multi-word names (e.g., "National Fruit Collection") pushed out critical categorical facts (e.g., "Family: Rosaceae") and numeric stats.
- **Hallucinated Calculation**: The SLM tried to calculate percentages (e.g., global production) from raw numbers in the L1 instead of admitting a Cache Miss and fetching the correct percentage from L2.

## 2. Current (Problematic) Code
```python
# caveman/core/graph.py
def rank_triples_by_importance(triples):
    # ... PageRank logic ...
    for triple, score in ranked:
        # BRITTLE: Hardcoded to multi-word proper nouns only
        tokens = [t for t in doc if t.pos_ == 'PROPN']
        if len(tokens) >= 2:
            score = max(score, 0.15) # TOO HIGH: Floods the cache
```

## 3. Proposed "General Case" Fix
The fix moves away from heuristics (word counts) and toward **Semantic Class Preservation**.

### A. Balanced Importance Scoring
We will implement a multi-dimensional boost using spaCy's `ENT_TYPE` (Named Entity Recognition). This covers Names, Places, Dates, and Percentages in one generalized sweep.
- **Categories to Floor (0.10)**: `PERSON`, `GPE`, `ORG`, `DATE`, `PERCENT`, `QUANTITY`.
- **Reasoning**: This ensures the L1 "Summary" always contains the **Who, Where, When, and How Much** of the document, regardless of the domain. It prevents Names from evicting Stats.

### B. Anti-Hallucination Guardrail (L2 Fallback)
To stop the SLM from guessing or doing math, we will tighten the **Inference-Fallback Coordination**.
- **The Constraint**: Update the system instruction in `ask_with_l2_fallback` to explicitly forbid calculation.
- **The Logic**: If `Answer_Verbatim_In_L1` is False -> Output `CACHE_MISS` -> Trigger `L2_QUERY`.

## 4. Verification for Gemini 3.1 Pro
Please evaluate the following logic:
"If we use NER-type floors (0.10) combined with PageRank connectivity, does this effectively solve the problem of 'Generic Fact Eviction' while still protecting 'Long-tail Specifics'? Is the risk of SLM over-calculation better solved by prompt engineering or by more aggressive L1 filtering?"
