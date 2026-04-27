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


def _extract_svo_triples(text: str) -> list[KnowledgeTriple]:
    """
    Open-world SVO extraction using spaCy with N-ary Property Edge model.
    """
    nlp = _load_spacy_model()
    doc = nlp(text)
    triples: list[KnowledgeTriple] = []

    for sent in doc.sents:
        # Find all verb tokens in the sentence to handle multiple clauses
        verb_tokens = [t for t in sent if t.pos_ in {"VERB", "AUX"}]
        
        # Capture all date/time entities in the sentence for "orphaned dates" logic
        sent_dates = [ent for ent in sent.ents if ent.label_ in {"DATE", "TIME"}]

        for verb_token in verb_tokens:
            subject = _find_subject(verb_token)
            verb = verb_token.text.strip()
            obj = _find_object(verb_token)

            if subject and verb and obj:
                # N-ary Property Extraction
                is_negated = False
                modality_parts = []
                condition = ""
                
                for child in verb_token.children:
                    if child.dep_ == "neg":
                        is_negated = True
                    elif child.dep_ == "advmod":
                        modality_parts.append(child.text)
                    elif child.dep_ == "advcl":
                        condition = " ".join(t.text for t in child.subtree).strip()
                
                modality = " ".join(modality_parts).strip()

                # Preposition migration: if object starts with a preposition, attach it to the verb
                PREPOSITIONS = {"in", "on", "at", "from", "to", "by", "with", "for", "into", "of"}
                obj_parts = obj.split()
                if len(obj_parts) > 1 and obj_parts[0].lower() in PREPOSITIONS:
                    prep = obj_parts[0]
                    verb = f"{verb} {prep}"
                    obj = " ".join(obj_parts[1:])

                # Orphaned dates logic: dates in sentence not appearing in sub/verb/obj
                triple_text_blobs = {subject.lower(), verb.lower(), obj.lower(), condition.lower(), modality.lower()}
                orphaned_dates = tuple(
                    ent.text.strip() for ent in sent_dates
                    if not any(ent.text.lower() in blob for blob in triple_text_blobs)
                )

                triples.append(KnowledgeTriple(
                    subject=subject,
                    verb=verb,
                    object=obj,
                    temporal_anchors=orphaned_dates,
                    modality=modality,
                    is_negated=is_negated,
                    condition=condition,
                    extraction_method="spacy",
                    is_deterministic=True
                ))
                
    return triples


def extract_markdown_triples(markdown_text: str) -> list[KnowledgeTriple]:
    """
    Parses Markdown text and extracts SVO triples while preserving hierarchical context
    via structural triples linking headers to subjects.
    """
    lines = markdown_text.splitlines()
    active_header = "Document Root"
    grouped_text: list[str] = []
    all_triples: list[KnowledgeTriple] = []

    def process_chunk(header: str, text_lines: list[str]):
        if not text_lines:
            return
        
        chunk_text = "\n".join(text_lines).strip()
        if not chunk_text:
            return
            
        svo_triples = _extract_svo_triples(chunk_text)
        
        for svo in svo_triples:
            all_triples.append(svo)
            # Create structural linkage
            structural_triple = KnowledgeTriple(
                subject=header,
                verb="contains concept",
                object=svo.subject,
                extraction_method="structural",
                is_deterministic=True
            )
            all_triples.append(structural_triple)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        if stripped.startswith("#"):
            # Process previous chunk
            process_chunk(active_header, grouped_text)
            grouped_text = []
            
            # Update active header
            active_header = stripped.lstrip("#").strip()
        else:
            grouped_text.append(line)

    # Process final chunk
    process_chunk(active_header, grouped_text)
    
    return all_triples


def extract_source_triples(text: str) -> list[KnowledgeTriple]:
    """
    Open-world extraction using spaCy, preserving Markdown hierarchy.
    """
    return extract_markdown_triples(text)


def extract_numeric_triples(text: str) -> list[KnowledgeTriple]:
    """
    Dedicated pass to harvest percentages, dates, and quantities
    that the SVO parser might miss.
    """
    nlp = _load_spacy_model()
    doc = nlp(text)
    triples = []
    
    for sent in doc.sents:
        subject = _find_sent_root_subject(sent) or "Document"
        numeric_ents = [ent for ent in sent.ents if ent.label_ in {"DATE", "PERCENT", "QUANTITY", "CARDINAL", "MONEY"}]
        
        for num_ent in numeric_ents:
            triples.append(KnowledgeTriple(
                subject=subject,
                verb="has value",
                object=num_ent.text.strip(),
                extraction_method="spacy_numeric",
                is_deterministic=True
            ))
    return triples


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


def _find_subject(verb_token) -> str:
    """Finds the subject relative to a specific verb token."""
    for child in verb_token.children:
        if child.dep_ in {"nsubj", "nsubjpass"}:
            return _span_text(child)
    
    # Fallback: if this is an auxiliary verb, check its head (the main verb)
    if verb_token.pos_ == "AUX" and verb_token.head.pos_ == "VERB":
        for child in verb_token.head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                return _span_text(child)
                
    return ""


def _find_sent_root_subject(sent) -> str:
    """Finds the subject of the root verb in a sentence (for numeric pass fallback)."""
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root:
        return _find_subject(root)
    return ""


def _find_object(verb_token) -> str:
    """Finds the object relative to a specific verb token."""
    seen_ids = set()
    object_tokens = []

    # 1. Direct objects and attributes
    for child in verb_token.children:
        if child.dep_ in {"dobj", "obj", "attr", "oprd"}:
            for t in child.subtree:
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
    """Helper to get full subtree text for a token."""
    return " ".join(t.text for t in token.subtree).strip()


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
