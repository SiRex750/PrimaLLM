from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from shared.triple import KnowledgeTriple


@dataclass(slots=True)
class SourceGraph:
    triples: list[KnowledgeTriple]
    graph: nx.DiGraph
    checksums: dict[str, str]


def triple_checksum(triple: KnowledgeTriple) -> str:
    return sha256(triple.as_text().encode("utf-8")).hexdigest()


def build_source_graph(triples: list[KnowledgeTriple]) -> SourceGraph:
    import networkx as nx

    graph = nx.DiGraph()
    checksums: dict[str, str] = {}

    for triple in triples:
        checksum = triple_checksum(triple)
        graph.add_node(triple.subject, checksum=checksum)
        graph.add_node(triple.object, checksum=checksum)
        graph.add_edge(triple.subject, triple.object, verb=triple.verb, checksum=checksum)
        checksums[triple.as_text()] = checksum

    return SourceGraph(triples=triples, graph=graph, checksums=checksums)
