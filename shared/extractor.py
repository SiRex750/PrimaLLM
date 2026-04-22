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
        verb_token = _find_root_verb_token(sent)
        verb = _verb_text(verb_token)
        obj = _find_object(sent, verb_token)
        if subject and verb and obj:
            triples.append(KnowledgeTriple(subject=subject, verb=verb, object=obj))

    return _deduplicate(triples)


def _find_subject(sent) -> str:
    for token in sent:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            return _span_text(token)
    return ""


def _find_root_verb_token(sent):
    for token in sent:
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            return token
    return None


def _verb_text(verb_token) -> str:
    if verb_token is None:
        return ""
    return verb_token.lemma_.strip() or verb_token.text.strip()


def _find_object(sent, verb_token) -> str:
    object_token = _select_object_token(sent, verb_token)
    collected_tokens = set()

    if object_token is not None:
        collected_tokens.update(getattr(object_token, "subtree", []))

    if verb_token is not None:
        for child in getattr(verb_token, "children", []):
            if child.dep_ == "prep" and any(grandchild.dep_ == "pobj" for grandchild in getattr(child, "children", [])):
                collected_tokens.update(getattr(child, "subtree", []))

            if child.dep_ in {"advcl", "xcomp"}:
                collected_tokens.update(getattr(child, "subtree", []))

    return _join_tokens_by_index(collected_tokens)


def _select_object_token(sent, verb_token):
    object_deps = {"dobj", "obj", "attr", "oprd", "pobj"}

    if verb_token is not None:
        for token in sent:
            if token.dep_ in {"dobj", "obj", "attr", "oprd"} and getattr(token, "head", None) is verb_token:
                return token

        for token in sent:
            head = getattr(token, "head", None)
            prep_head = getattr(head, "head", None)
            if token.dep_ == "pobj" and getattr(head, "dep_", "") == "prep" and prep_head is verb_token:
                return token

    for token in sent:
        if token.dep_ in object_deps:
            return token

    return None


def _span_text(token) -> str:
    subtree_tokens = set(getattr(token, "subtree", []))
    if not subtree_tokens:
        text = getattr(token, "text", "").strip()
        return text
    return _join_tokens_by_index(subtree_tokens)


def _join_tokens_by_index(tokens) -> str:
    ordered = sorted(tokens, key=lambda node: node.i)
    return " ".join(node.text for node in ordered).strip()


def _deduplicate(triples: Iterable[KnowledgeTriple]) -> list[KnowledgeTriple]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[KnowledgeTriple] = []
    for triple in triples:
        key = (triple.subject, triple.verb, triple.object)
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    return unique
