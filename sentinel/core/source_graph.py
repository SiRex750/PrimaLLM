from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import networkx as nx

from shared.triple import KnowledgeTriple


@dataclass(slots=True)
class SourceGraph:
    triples: list[KnowledgeTriple]
    graph: nx.DiGraph
    checksums: dict[str, str]
    master_checksum: str


def triple_checksum(triple: KnowledgeTriple) -> str:
    return sha256(triple.as_text().encode("utf-8")).hexdigest()


def build_source_graph(triples: list[KnowledgeTriple]) -> SourceGraph:
    graph = nx.DiGraph()
    checksums: dict[str, str] = {}

    for triple in triples:
        checksum = triple_checksum(triple)
        graph.add_node(triple.subject, checksum=checksum)
        graph.add_node(triple.object, checksum=checksum)
        graph.add_edge(triple.subject, triple.object, verb=triple.verb, checksum=checksum)
        checksums[triple.as_text()] = checksum

    master_text = "".join(sorted(triple.as_text() for triple in triples))
    master_checksum = sha256(master_text.encode("utf-8")).hexdigest()

    return SourceGraph(
        triples=triples,
        graph=graph,
        checksums=checksums,
        master_checksum=master_checksum,
    )
