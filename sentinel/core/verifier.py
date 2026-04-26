from __future__ import annotations

import re
import spacy
from functools import lru_cache
from dataclasses import dataclass

from sentinel.core.source_graph import SourceGraph
from shared.triple import KnowledgeTriple


@dataclass(slots=True)
class VerificationResult:
    is_verified: bool
    reason: str
    label: str = ""




def verify_claim(claim: KnowledgeTriple, source_graph: SourceGraph, model_name: str = "cross-encoder/nli-deberta-v3-base", source_sentences: list[str] = None) -> VerificationResult:
    import torch
    import torch.nn.functional as F

    premise = _build_localized_premise(claim, source_graph, source_sentences)
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
    
    # print(f"CLAIM: {claim.as_text()}")
    # print(f"PREMISE: {premise}")
    # print(f"PREDICTION: {label}")

    if not claim.is_deterministic and label == "entailment":
        probs = F.softmax(outputs.logits, dim=-1)
        entailment_idx = [i for i, l in model.config.id2label.items() if "entail" in l.lower()][0]
        entailment_score = probs[0][entailment_idx].item()

        if entailment_score <= 0.85:
            return VerificationResult(
                is_verified=False,
                reason=f"GLiNER-extracted triple requires higher confidence threshold (got {entailment_score:.2f})",
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
        reason = f"Rejected by local DeBERTa-v3 NLI model: the claim is not entailed by the source graph (label: {label})."

    return VerificationResult(is_verified=False, reason=reason, label=label)



@lru_cache(maxsize=1)
def _load_spacy():
    return spacy.load("en_core_web_sm")


def _get_claim_keywords(text: str) -> set[str]:
    """
    Extract keywords from a claim for premise retrieval.
    
    Unlike aggressive lemmatisation, this preserves:
    - All nouns and proper nouns (original form)
    - All verbs (lemmatised for matching flexibility)  
    - All numbers and percentages
    - Words longer than 3 characters that aren't pure stop words
    
    Does NOT filter out: numbers, short verbs like "has"/"grew",
    determiners that carry meaning, or domain terms.
    """
    nlp = _load_spacy()
    
    doc = nlp(text.lower())
    keywords = set()
    
    for token in doc:
        # Always keep: nouns, proper nouns, verbs, numbers
        if token.pos_ in {"NOUN", "PROPN", "NUM"}:
            keywords.add(token.lemma_)
            keywords.add(token.text)  # add both forms
        elif token.pos_ == "VERB" and len(token.text) > 1:
            keywords.add(token.lemma_)
        # Keep numbers and percentages regardless of POS
        elif any(c.isdigit() for c in token.text):
            keywords.add(token.text)
        # Keep words > 3 chars that aren't pure punctuation
        elif len(token.text) > 3 and not token.is_punct:
            keywords.add(token.lemma_)
    
    return keywords


def _build_localized_premise(claim: KnowledgeTriple, source_graph: SourceGraph, source_sentences: list[str] = None) -> str:
    claim_keywords = _get_claim_keywords(claim.as_text())
    
    # Match triples
    matching_triples: list[KnowledgeTriple] = []
    for triple in source_graph.triples:
        triple_keywords = _get_claim_keywords(triple.as_text())
        overlap = len(claim_keywords.intersection(triple_keywords))
        if overlap >= 2:
            matching_triples.append(triple)
    
    # Match sentences  
    scored_sentences: list[tuple[int, str]] = []
    if source_sentences:
        for sentence in source_sentences:
            sent_keywords = _get_claim_keywords(sentence)
            overlap = len(claim_keywords.intersection(sent_keywords))
            if overlap >= 2:  # sentences need stronger match
                clean_sent = " ".join(sentence.split())
                scored_sentences.append((overlap, clean_sent))
    
    # Sort by overlap score descending, take only the TOP 1
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    matching_sentences = [s for _, s in scored_sentences[:1]]
    
    # Build premise: prose first, then triples
    # Limit to 1 sentences and 2 triples to avoid noise
    premise_parts = []
    for sent in matching_sentences:
        premise_parts.append(sent)
    for triple in matching_triples[:2]:
        clean_triple = " ".join(triple.as_text().split())
        premise_parts.append(clean_triple)
    
    premise = " ".join(premise_parts)
    return " ".join(premise.split())  # final whitespace normalisation




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
