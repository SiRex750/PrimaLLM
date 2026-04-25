from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeTriple:
    subject: str
    verb: str
    object: str
    extraction_method: str = "spacy"
    is_deterministic: bool = True

    def as_text(self) -> str:
        return f"{self.subject} {self.verb} {self.object}"
