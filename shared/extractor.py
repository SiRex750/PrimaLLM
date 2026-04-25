from __future__ import annotations

from functools import lru_cache
from typing import Iterable

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
    "orbited", "reported", "developed", "founded"
]


@lru_cache(maxsize=1)
def _load_model():
    return GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")


def extract_knowledge_triples(text: str) -> list[KnowledgeTriple]:
    model = _load_model()

    entities, relations = model.inference(
        texts=[text],
        labels=ENTITY_LABELS,
        relations=RELATION_LABELS,
        threshold=0.3,
        adjacency_threshold=0.5,
        relation_threshold=0.75,
        return_relations=True,
        flat_ner=False
    )

    triples: list[KnowledgeTriple] = []

    for relation in relations[0]:
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
