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
    """Extract knowledge triples from *text*.

    Handles:
    - Standard SVO triples (including copular *attr* for forms of "to be")
    - Appositive triples  (``appos`` → ``subject is appositive``)
    - Adnominal-clause triples (``acl`` → ``noun  acl-verb  acl-object``)
    """
    nlp = _load_spacy_model()
    doc = nlp(text)
    triples: list[KnowledgeTriple] = []

    for sent in doc.sents:
        triples.extend(_extract_svo_triples(sent))
        triples.extend(_extract_appos_triples(sent))
        triples.extend(_extract_acl_triples(sent))

    return _deduplicate(triples)


# ── SVO (+ copular attr) ─────────────────────────────────────────────

def _extract_svo_triples(sent) -> Iterable[KnowledgeTriple]:
    """Yield the primary subject-verb-object triple for *sent*.

    When the ROOT verb is a copula (a form of *to be*), spaCy labels
    the complement as ``attr`` rather than ``dobj``.  ``_select_object_token``
    already accepts ``attr``, so this works transparently.
    """
    subject = _find_subject(sent)
    verb_token = _find_root_verb_token(sent)
    verb = _verb_text(verb_token)
    obj = _find_object(sent, verb_token)
    if subject and verb and obj:
        yield KnowledgeTriple(subject=subject, verb=verb, object=obj)


# ── Appositive triples ───────────────────────────────────────────────

def _extract_appos_triples(sent) -> Iterable[KnowledgeTriple]:
    """Yield ``(Noun, "is", Appositive)`` for every appositive in *sent*.

    An appositive (``dep_ == "appos"``) renames or further identifies its
    head noun, so the relationship is expressed as *"Noun is Appositive"*.
    """
    for token in sent:
        if token.dep_ == "appos":
            head_noun = token.head
            subject = _span_text_without_appos(head_noun)
            appositive = _span_text(token)
            if subject and appositive:
                yield KnowledgeTriple(subject=subject, verb="is", object=appositive)


# ── Adnominal-clause (acl) triples ───────────────────────────────────

def _extract_acl_triples(sent) -> Iterable[KnowledgeTriple]:
    """Yield triples for adnominal clause modifiers (``acl``).

    When a noun is modified by an ``acl``, the clause verb introduces a
    new predicate whose subject is the modified noun.  We look for the
    direct object (or ``attr`` / ``oprd``) of that clause verb and, if
    present, yield a triple.
    """
    for token in sent:
        if token.dep_ == "acl" and token.pos_ in {"VERB", "AUX"}:
            modified_noun = token.head
            subject = _span_text_without_acl(modified_noun)
            verb = _verb_text(token)
            obj = _find_acl_object(token)
            if subject and verb and obj:
                yield KnowledgeTriple(subject=subject, verb=verb, object=obj)


def _find_acl_object(acl_verb) -> str:
    """Find the object governed by an ``acl`` verb token."""
    # Direct object / attribute / adjectival complement / object predicate
    for child in acl_verb.children:
        if child.dep_ in {"dobj", "obj", "attr", "acomp", "oprd"}:
            return _span_text(child)

    # Prepositional / agent object (prep/agent → pobj)
    for child in acl_verb.children:
        if child.dep_ in {"prep", "agent"}:
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return _span_text(grandchild)

    return ""


# ── Shared helpers ────────────────────────────────────────────────────

def _find_subject(sent) -> str:
    for token in sent:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            return _span_text_clean(token)
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
    collected_tokens: set = set()

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
    object_deps = {"dobj", "obj", "attr", "acomp", "oprd", "pobj"}

    if verb_token is not None:
        for token in sent:
            if token.dep_ in {"dobj", "obj", "attr", "acomp", "oprd"} and _same_token(token.head, verb_token):
                return token

        for token in sent:
            head = getattr(token, "head", None)
            prep_head = getattr(head, "head", None) if head is not None else None
            if token.dep_ == "pobj" and getattr(head, "dep_", "") == "prep" and _same_token(prep_head, verb_token):
                return token

    for token in sent:
        if token.dep_ in object_deps:
            return token

    return None


def _same_token(a, b) -> bool:
    """Compare two spaCy tokens by document index.

    spaCy ``Token`` objects do not guarantee ``is``-identity across
    different access paths, so we compare by ``.i`` instead.
    """
    if a is None or b is None:
        return False
    return a.i == b.i


def _span_text(token) -> str:
    """Return the full subtree text for *token*, merged by token index."""
    subtree_tokens = set(getattr(token, "subtree", []))
    if not subtree_tokens:
        text = getattr(token, "text", "").strip()
        return text
    return _join_tokens_by_index(subtree_tokens)


def _span_text_clean(token) -> str:
    """Return subtree text excluding ``appos``, ``acl``, and stray punct.

    Used for the SVO subject span so that appositives and clause modifiers
    don't leak into the subject string.
    """
    excluded_deps = {"appos", "acl"}
    subtree_tokens = {
        t for t in getattr(token, "subtree", [])
        if t.dep_ not in excluded_deps
        and not _has_ancestor_with_dep_set(t, token, excluded_deps)
        and t.dep_ != "punct"
    }
    if not subtree_tokens:
        return getattr(token, "text", "").strip()
    return _join_tokens_by_index(subtree_tokens)


def _span_text_without_appos(token) -> str:
    """Return the subtree text for *token* excluding any appositive branches.

    This prevents the appositive itself from being duplicated inside the
    subject span when we later emit the ``(Noun, "is", Appositive)`` triple.
    """
    subtree_tokens = {
        t for t in getattr(token, "subtree", [])
        if t.dep_ != "appos"
        and not _has_ancestor_with_dep(t, token, "appos")
        and t.dep_ != "punct"
    }
    if not subtree_tokens:
        return getattr(token, "text", "").strip()
    return _join_tokens_by_index(subtree_tokens)


def _span_text_without_acl(token) -> str:
    """Return the subtree text for *token* excluding any ``acl`` branches.

    This prevents the clause modifier from leaking into the subject span.
    """
    subtree_tokens = {
        t for t in getattr(token, "subtree", [])
        if t.dep_ != "acl"
        and not _has_ancestor_with_dep(t, token, "acl")
        and t.dep_ != "punct"
    }
    if not subtree_tokens:
        return getattr(token, "text", "").strip()
    return _join_tokens_by_index(subtree_tokens)


def _has_ancestor_with_dep(token, root, dep: str) -> bool:
    """Return ``True`` if *token* has an ancestor between itself and *root*
    whose ``dep_`` equals *dep*.
    """
    current = token.head
    while current != root and current != current.head:
        if current.dep_ == dep:
            return True
        current = current.head
    return False


def _has_ancestor_with_dep_set(token, root, deps: set[str]) -> bool:
    """Like :func:`_has_ancestor_with_dep` but accepts a *set* of deps."""
    current = token.head
    while current != root and current != current.head:
        if current.dep_ in deps:
            return True
        current = current.head
    return False


def _join_tokens_by_index(tokens) -> str:
    """Sort *tokens* by their document index and join with spaces."""
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
