from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import spacy
from gliner import GLiNER

from .triple import KnowledgeTriple

ENTITY_LABELS = [
    "person", "organization", "location", "event",
    "concept", "object", "date", "quantity"
]

RELATION_LABELS = [
    "is", "has", "belongs to", "located in", "founded by",
    "works at", "created", "caused", "launched on",
    "remained in", "part of", "used for", "descended to",
    "orbited", "reported", "developed", "founded",
    "comes from", "originated in", "was founded in", "was born in",
    "produced", "manufactures", "grows in", "matures in",
    "was sequenced in", "was cultivated by", "was discovered by",
    "contains", "releases", "causes", "requires", "consists of",
    "is classified as", "is native to", "was written by",
    "was built in", "was established in", "produces",
    "grows up to", "is measured at"
]


@lru_cache(maxsize=1)
def _load_gliner_model():
    return GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def extract_source_triples(text: str) -> list[KnowledgeTriple]:
    """
    Open-world SVO extraction using spaCy.
    Prioritises recall for building the SourceGraph.
    """
    nlp = _load_spacy_model()
    doc = nlp(text)
    triples: list[KnowledgeTriple] = []

    for sent in doc.sents:
        subject = _find_subject(sent)
        verb_token = _find_root_verb_token(sent)
        verb = _verb_text(verb_token)
        obj = _find_object(sent, verb_token)

        if subject and verb and obj:
            # --- TEMPORAL HARVESTING ---
            # Grab any dates/times in the sentence that the dependency parser missed
            # Store them as metadata rather than mutating the graph node strings.
            combined_triple_text = f"{subject} {verb} {obj}".lower()
            orphaned_dates = tuple(
                ent.text.strip() for ent in sent.ents
                if ent.label_ in {"DATE", "TIME"} and ent.text.lower() not in combined_triple_text
            )
            # ---------------------------

            # Move leading preposition from object to verb for cleaner nodes
            # e.g., (apple, born, in Kazakhstan) -> (apple, born in, Kazakhstan)
            PREPOSITIONS = {
                "in", "on", "at", "from", "to", "by", "with", "for", "into", "of"
            }
            obj_parts = obj.split()
            if len(obj_parts) > 1 and obj_parts[0].lower() in PREPOSITIONS:
                prep = obj_parts[0]
                verb = f"{verb} {prep}"
                obj = " ".join(obj_parts[1:])

            triples.append(KnowledgeTriple(
                subject=subject,
                verb=verb,
                object=obj,
                temporal_anchors=orphaned_dates,
                extraction_method="spacy",
                is_deterministic=True
            ))

    return _deduplicate(triples)


def extract_claim_triples(text: str) -> list[KnowledgeTriple]:
    """
    Closed-world extraction using GLiNER-relex.
    Prioritises precision for verifying LLM claims.
    """
    model = _load_gliner_model()

    # Split text into manageable chunks (sentences) using a regex
    chunks = re.split(r'(?<=[.!?])\s+', text)
    chunks = [c.strip() for c in chunks if c.strip()]

    if not chunks:
        return []

    # GLiNER can process multiple texts in a single batch
    entities_list, relations_list = model.inference(
        texts=chunks,
        labels=ENTITY_LABELS,
        relations=RELATION_LABELS,
        threshold=0.3,
        adjacency_threshold=0.5,
        relation_threshold=0.75,
        return_relations=True,
        flat_ner=False
    )

    triples: list[KnowledgeTriple] = []

    for relations in relations_list:
        for relation in relations:
            subject = relation["head"]["text"].strip()
            predicate = relation["relation"].strip()
            obj = relation["tail"]["text"].strip()
            if subject and predicate and obj:
                triples.append(KnowledgeTriple(
                    subject=subject,
                    verb=predicate,
                    object=obj,
                    extraction_method="gliner_relex",
                    is_deterministic=True
                ))

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
    return verb_token.text.strip()


def _find_object(sent, verb_token) -> str:
    if verb_token is None:
        return ""

    seen_ids = set()
    object_tokens = []

    # 1. Direct objects and attributes
    for token in sent:
        if token.dep_ in {"dobj", "obj", "attr", "oprd"} and token.head == verb_token:
            for t in token.subtree:
                if t.i not in seen_ids:
                    seen_ids.add(t.i)
                    object_tokens.append(t)

    # 2. Prepositional phrases attached to the verb
    for child in verb_token.children:
        if child.dep_ == "prep":
            for t in child.subtree:
                if t.i not in seen_ids:
                    seen_ids.add(t.i)
                    object_tokens.append(t)

    if not object_tokens:
        return ""

    # Sort by original token index to maintain sentence order
    object_tokens.sort(key=lambda x: x.i)
    return " ".join(t.text for t in object_tokens).strip()






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


def _deduplicate(triples: list[KnowledgeTriple]) -> list[KnowledgeTriple]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[KnowledgeTriple] = []
    for t in triples:
        key = (t.subject.strip().lower(),
               t.verb.strip().lower(),
               t.object.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique
