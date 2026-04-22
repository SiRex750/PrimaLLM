from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Iterable
import tiktoken

from shared.triple import KnowledgeTriple


_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True)
class CacheEntry:
    triple: KnowledgeTriple
    pagerank_score: float = 0.0

    @property
    def text(self) -> str:
        return self.triple.as_text()


@dataclass(slots=True)
class HistoryTurn:
    role: str
    text: str


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    text: str


DEFAULT_BUDGETS: dict[str, int] = {
    "system": 50,
    "facts": 100,
    "history": 150,
    "tools": 100,
}


class L1Cache:
    def __init__(self, budgets: dict[str, int] | None = None) -> None:
        self.budgets = self._resolved_budgets(budgets)

        self.set_system: list[str] = []
        self.set_facts: OrderedDict[str, CacheEntry] = OrderedDict()
        self.set_history: deque[HistoryTurn] = deque()
        self.set_tools: deque[ToolResult] = deque()

    def add_system_instruction(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned or cleaned in self.set_system:
            return

        projected = self._estimate_system_tokens() + _count_tokens(cleaned)
        if projected > self.budgets["system"]:
            raise ValueError("System set budget exceeded; system instructions are pinned and cannot be evicted.")

        self.set_system.append(cleaned)

    def add_fact(
        self,
        triple: KnowledgeTriple,
        pagerank_score: float = 0.0,
    ) -> None:
        if not isinstance(triple, KnowledgeTriple):
            raise TypeError("add_fact expects a KnowledgeTriple instance.")

        entry = CacheEntry(triple=triple, pagerank_score=pagerank_score)
        self.set_facts[entry.text] = entry
        self._trim_facts_to_budget()

    def add_history_turn(self, role: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        self.set_history.append(HistoryTurn(role=role.strip() or "user", text=cleaned))
        self._trim_history_to_budget()

    def add_tool_result(self, tool_name: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        self.set_tools.append(ToolResult(tool_name=tool_name.strip() or "tool", text=cleaned))
        self._trim_tools_to_budget()

    def extend(self, triples: Iterable[KnowledgeTriple]) -> None:
        for triple in triples:
            self.add_fact(triple)

    @property
    def active_facts(self) -> OrderedDict[str, CacheEntry]:
        return self.set_facts

    def as_context_lines(self) -> list[str]:
        lines: list[str] = []

        if self.set_system:
            lines.append("System Instructions:")
            lines.extend(self.set_system)

        if self.set_tools:
            lines.append("Active Tool Results:")
            lines.extend(f"{item.tool_name}: {item.text}" for item in self.set_tools)

        if self.set_facts:
            lines.append("Fact Cache:")
            lines.extend(entry.text for entry in self.set_facts.values())

        if self.set_history:
            lines.append("Conversation History:")
            lines.extend(f"{turn.role}: {turn.text}" for turn in self.set_history)

        return lines

    def as_context_text(self) -> str:
        return "\n".join(self.as_context_lines())

    def _trim_facts_to_budget(self) -> None:
        while self._estimate_facts_tokens() > self.budgets["facts"] and self.set_facts:
            lowest_key = min(self.set_facts, key=lambda key: self.set_facts[key].pagerank_score)
            del self.set_facts[lowest_key]

    def _trim_history_to_budget(self) -> None:
        while self._estimate_history_tokens() > self.budgets["history"] and self.set_history:
            self.set_history.popleft()

    def _trim_tools_to_budget(self) -> None:
        while self._estimate_tools_tokens() > self.budgets["tools"] and self.set_tools:
            self.set_tools.popleft()

    def _estimate_system_tokens(self) -> int:
        return sum(_count_tokens(text) for text in self.set_system)

    def _estimate_facts_tokens(self) -> int:
        return sum(_count_tokens(entry.text) for entry in self.set_facts.values())

    def _estimate_history_tokens(self) -> int:
        return sum(_count_tokens(f"{turn.role}: {turn.text}") for turn in self.set_history)

    def _estimate_tools_tokens(self) -> int:
        return sum(_count_tokens(f"{item.tool_name}: {item.text}") for item in self.set_tools)

    def _resolved_budgets(self, budgets: dict[str, int] | None) -> dict[str, int]:
        resolved = dict(DEFAULT_BUDGETS)
        if budgets is not None:
            unknown = set(budgets).difference(DEFAULT_BUDGETS)
            if unknown:
                unknown_names = ", ".join(sorted(unknown))
                raise ValueError(f"Unknown budget set(s): {unknown_names}")
            resolved.update(budgets)

        for set_name, value in resolved.items():
            if value < 0:
                raise ValueError(f"Budget for '{set_name}' must be non-negative.")

        return resolved


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))
