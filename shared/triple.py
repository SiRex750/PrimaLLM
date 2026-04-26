from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeTriple:
    subject: str
    verb: str
    object: str
    temporal_anchors: tuple[str, ...] = tuple()
    extraction_method: str = "spacy"
    is_deterministic: bool = True

    def as_text(self) -> str:
        text = f"{self.subject} {self.verb} {self.object}"
        if self.temporal_anchors:
            anchors_str = ", ".join(self.temporal_anchors)
            text = f"{text} (Context: {anchors_str})"
        return text
