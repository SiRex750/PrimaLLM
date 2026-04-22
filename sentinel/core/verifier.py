from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

from sentinel.core.source_graph import SourceGraph


@dataclass(slots=True)
class VerificationResult:
    is_verified: bool
    reason: str
    label: str = ""


def verify_claim(claim: str, source_graph: SourceGraph, model_name: str = "cross-encoder/nli-deberta-v3-small") -> VerificationResult:
    import torch

    premise = " ".join(triple.as_text() for triple in source_graph.triples).strip()
    if not premise:
        return VerificationResult(is_verified=False, reason="No source facts available in the source graph.")

    tokenizer, model = _load_nli_model(model_name)
    inputs = tokenizer(
        premise,
        claim,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = int(torch.argmax(outputs.logits, dim=-1).item())

    label = _resolve_label(model, prediction)
    if label == "entailment":
        return VerificationResult(
            is_verified=True,
            reason="Verified by local DeBERTa-v3 NLI model against the source graph.",
            label=label,
        )

    if label == "contradiction":
        reason = "Rejected by local DeBERTa-v3 NLI model: the claim contradicts the source graph."
    else:
        reason = "Rejected by local DeBERTa-v3 NLI model: the claim is not entailed by the source graph."

    return VerificationResult(is_verified=False, reason=reason, label=label)


@lru_cache(maxsize=1)
def _load_nli_model(model_name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _resolve_label(model, prediction: int) -> str:
    id2label = getattr(model.config, "id2label", {}) or {}
    label = str(id2label.get(prediction, "")).lower()
    if "entail" in label:
        return "entailment"
    if "contrad" in label:
        return "contradiction"
    if "neutral" in label:
        return "neutral"
    if prediction == 2:
        return "entailment"
    if prediction == 0:
        return "contradiction"
    return "neutral"
