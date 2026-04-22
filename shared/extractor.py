from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from .triple import KnowledgeTriple


@lru_cache(maxsize=1)
def _load_spacy_model():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model en_core_web_sm is required for local-first extraction. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc


def extract_knowledge_triples(text: str) -> list[KnowledgeTriple]:
    nlp = _load_spacy_model()
    doc = nlp(text)
    triples: list[KnowledgeTriple] = []

    for sent in doc.sents:
        subject = _find_subject(sent)
        verb = _find_root_verb(sent)
        obj = _find_object(sent)
        if subject and verb and obj:
            triples.append(KnowledgeTriple(subject=subject, verb=verb, object=obj))

    return _deduplicate(triples)


def _find_subject(sent) -> str:
    for token in sent:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            return _span_text(token)
    return ""


def _find_root_verb(sent) -> str:
    for token in sent:
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            return token.lemma_.strip() or token.text.strip()
    return ""


def _find_object(sent) -> str:
    for token in sent:
        if token.dep_ in {"dobj", "attr", "oprd", "pobj"}:
            return _span_text(token)
    return ""


def _span_text(token) -> str:
    subtree = getattr(token, "subtree", None)
    if subtree is not None:
        try:
            text = " ".join(node.text for node in subtree).strip()
            if text:
                return text
        except TypeError:
            text = getattr(subtree, "text", "").strip()
            if text:
                return text

    return getattr(token, "text", "").strip()


def _deduplicate(triples: Iterable[KnowledgeTriple]) -> list[KnowledgeTriple]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[KnowledgeTriple] = []
    for triple in triples:
        key = (triple.subject, triple.verb, triple.object)
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    return unique
