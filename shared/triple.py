from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeTriple:
    subject: str
    verb: str
    object: str
    temporal_anchors: tuple[str, ...] = tuple()
    modality: str = ""
    is_negated: bool = False
    condition: str = ""
    extraction_method: str = "spacy"
    is_deterministic: bool = True

    def as_text(self) -> str:
        text = f"{self.subject} {self.verb} {self.object}"
        if self.is_negated:
            text = f"{self.subject} not {self.verb} {self.object}"
        
        details = []
        if self.modality:
            details.append(f"Modality: {self.modality}")
        if self.condition:
            details.append(f"Condition: {self.condition}")
        if self.temporal_anchors:
            details.append(f"Context: {', '.join(self.temporal_anchors)}")
            
        if details:
            text = f"{text} ({'; '.join(details)})"
            
        return text
