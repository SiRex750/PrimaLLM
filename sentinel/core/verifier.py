from __future__ import annotations

import re
from functools import lru_cache
from dataclasses import dataclass

from sentinel.core.source_graph import SourceGraph
from shared.triple import KnowledgeTriple


@dataclass(slots=True)
class VerificationResult:
    is_verified: bool
    reason: str
    label: str = ""


_STOPWORDS = {"the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "for", "by", "with"}


def verify_claim(claim: KnowledgeTriple, source_graph: SourceGraph, model_name: str = "cross-encoder/nli-deberta-v3-small") -> VerificationResult:
    import torch
    import torch.nn.functional as F

    premise = _build_localized_premise(claim, source_graph)
    if not premise:
        return VerificationResult(is_verified=False, reason="No relevant facts found in the source graph context")

    tokenizer, model = _load_nli_model(model_name)
    inputs = tokenizer(
        premise,
        claim.as_text(),
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = int(torch.argmax(outputs.logits, dim=-1).item())

    label = _resolve_label(model, prediction)

    if not claim.is_deterministic and label == "entailment":
        probs = F.softmax(outputs.logits, dim=-1)
        entailment_idx = [i for i, l in model.config.id2label.items() if "entail" in l.lower()][0]
        entailment_score = probs[0][entailment_idx].item()

        if entailment_score <= 0.85:
            return VerificationResult(
                is_verified=False,
                reason="GLiNER-extracted triple requires higher confidence threshold",
                label="neutral"
            )

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


def _build_localized_premise(claim: KnowledgeTriple, source_graph: SourceGraph) -> str:
    claim_subject_words = _filtered_words(claim.subject)
    claim_object_words = _filtered_words(claim.object)
    claim_text_words = _filtered_words(claim.as_text())

    relevant_triples: list[str] = []

    for triple in source_graph.triples:
        subject_words = _filtered_words(triple.subject)
        object_words = _filtered_words(triple.object)

        if (
            claim_subject_words.intersection(subject_words)
            or claim_subject_words.intersection(object_words)
            or claim_object_words.intersection(subject_words)
            or claim_object_words.intersection(object_words)
        ):
            relevant_triples.append(triple.as_text())
            continue

        source_text_words = _filtered_words(triple.as_text())
        if claim_text_words.intersection(source_text_words):
            relevant_triples.append(triple.as_text())

    return " ".join(relevant_triples).strip()


def _filtered_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[A-Za-z0-9]+", text.lower()) if word not in _STOPWORDS}


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
