from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from shared.triple import KnowledgeTriple


@dataclass(slots=True)
class CacheEntry:
    triple: KnowledgeTriple
    pagerank_score: float = 0.0

    @property
    def text(self) -> str:
        return self.triple.as_text()


class L1Cache:
    def __init__(self, token_budget: int = 512) -> None:
        self.token_budget = token_budget
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    def add_fact(
        self,
        subject: str | KnowledgeTriple,
        verb: str | None = None,
        object: str | None = None,
        pagerank_score: float = 0.0,
    ) -> None:
        triple = subject if isinstance(subject, KnowledgeTriple) else KnowledgeTriple(subject, verb or "", object or "")
        entry = CacheEntry(triple=triple, pagerank_score=pagerank_score)
        self._entries[entry.text] = entry
        self._trim_to_budget()

    def extend(self, triples: Iterable[KnowledgeTriple]) -> None:
        for triple in triples:
            self.add_fact(triple)

    @property
    def active_facts(self) -> OrderedDict[str, CacheEntry]:
        return self._entries

    def as_context_lines(self) -> list[str]:
        return list(self._entries.keys())

    def _trim_to_budget(self) -> None:
        while self._estimate_tokens() > self.token_budget and self._entries:
            self._entries.popitem(last=False)

    def _estimate_tokens(self) -> int:
        return sum(len(text.split()) for text in self._entries)
