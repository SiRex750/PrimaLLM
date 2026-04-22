# Sentinel

Sentinel is Path B of PrimaLLM: a deterministic verification layer that validates claims against source facts before persistence.

## What Sentinel Does

- Builds a source-of-truth graph from extracted triples.
- Computes per-triple and master semantic checksums.
- Verifies claims using a local NLI model (`cross-encoder/nli-deberta-v3-small`).
- Persists only verified facts to local JSON storage.

## Verification Pipeline

1. Extract triples from source text using `shared/extractor.py`.
2. Build `SourceGraph` in `sentinel/core/source_graph.py`:
- graph nodes and edges from triples
- SHA-256 checksum per triple
- `master_checksum` from sorted triple text
3. Convert candidate claim(s) into text hypotheses.
4. Run NLI in `sentinel/core/verifier.py`:
- `entailment` -> accept
- `contradiction` or `neutral` -> reject
5. Save verified facts through `sentinel/core/wiki_storage.py`.

## Run Sentinel Demo

Install dependencies (shared with this repo's current requirements set):

```powershell
.\.venv\Scripts\python.exe -m pip install -r caveman\requirements.txt
```

Install spaCy model:

```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Run the end-to-end demo:

```powershell
.\.venv\Scripts\python.exe -m sentinel.main
```

Run benchmark scaffold:

```powershell
.\.venv\Scripts\python.exe -m sentinel.benchmark.run_verification_benchmark
```

## Example Demo Behavior

Typical output shape:

```text
--- SENTINEL VERIFICATION ORACLE ---
1. Extracting Source Graph (Semantic Checksum)...
	Nodes: 6 | Edges: 3
	Master Checksum (SHA-256): <prefix>...

2. Testing Valid AI Response:
	Result: ✅ VALID (...)
	-> Fact persisted to wiki.json

3. Testing Hallucinated AI Response:
	Result: ❌ INVALID (...)
	-> Blocked. Fact NOT saved to wiki.
```

## Using Sentinel as a Library

```python
from shared.extractor import extract_knowledge_triples
from sentinel.core.source_graph import build_source_graph
from sentinel.core.verifier import verify_claim
from sentinel.core.wiki_storage import save_verified_fact

source_text = "Neil Armstrong piloted the Lunar Module Eagle."
source_graph = build_source_graph(extract_knowledge_triples(source_text))

claim = "Neil Armstrong piloted the Lunar Module Eagle"
result = verify_claim(claim, source_graph)

if result.is_verified:
	 for triple in extract_knowledge_triples(claim):
		  save_verified_fact(triple)
```

## Persistence Details

- Default store path: `sentinel/core/wiki.json`.
- The file is created automatically when the first verified fact is saved.
- Duplicate `(subject, verb, object)` records are skipped.

## Operational Notes

- First run may download model weights for transformers and spaCy artifacts.
- You may see Hugging Face unauthenticated warnings; runs still work, but `HF_TOKEN` can improve limits.
- Verification is performed against source triples, not against prior AI responses.
