from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.triple import KnowledgeTriple

DEFAULT_STORAGE_PATH = Path(__file__).with_name("wiki.json")


def load_wiki(storage_path: str | Path | None = None) -> list[dict[str, str]]:
    path = Path(storage_path) if storage_path is not None else DEFAULT_STORAGE_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_verified_fact(
    subject: str | KnowledgeTriple,
    verb: str | None = None,
    object: str | None = None,
    storage_path: str | Path | None = None,
) -> None:
    path = Path(storage_path) if storage_path is not None else DEFAULT_STORAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    triple = subject if isinstance(subject, KnowledgeTriple) else KnowledgeTriple(subject, verb or "", object or "")
    existing = load_wiki(path)
    if any(
        fact.get("subject") == triple.subject
        and fact.get("verb") == triple.verb
        and fact.get("object") == triple.object
        for fact in existing
    ):
        return
    existing.append({"subject": triple.subject, "verb": triple.verb, "object": triple.object})
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=True), encoding="utf-8")
