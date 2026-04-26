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
    source_sentences: list[str] = None


def triple_checksum(triple: KnowledgeTriple) -> str:
    return sha256(triple.as_text().encode("utf-8")).hexdigest()


def build_source_graph(triples: list[KnowledgeTriple], embedder=None, source_sentences: list[str] = None) -> SourceGraph:
    from caveman.core import build_graph
    graph = build_graph(triples)
    checksums: dict[str, str] = {}

    for triple in triples:
        checksums[triple.as_text()] = triple_checksum(triple)

    # ── Entity coreference merging ──────────────────────────────
    # Collapse alias nodes into canonical entities before PageRank.
    # "Malus sieversii" and "wild ancestor" accumulate shared 
    # PageRank rather than splitting it across two nodes.
    # Uses batched embedding — ~80ms overhead for 100 nodes.
    if embedder is not None:
        from caveman.core.graph import merge_similar_nodes
        graph, merged_count = merge_similar_nodes(
            graph, embedder, threshold=0.82
        )
        if merged_count > 0:
            import logging
            logging.info(
                f"[HADES] Entity merging: resolved {merged_count} "
                f"alias nodes → {graph.number_of_nodes()} canonical nodes"
            )

    # Embed nodes (required for L2 memory queries)
    for node in graph.nodes():
        if embedder:
            graph.nodes[node]["vector"] = embedder.encode(node)

    master_text = "".join(sorted(triple.as_text() for triple in triples))
    master_checksum = sha256(master_text.encode("utf-8")).hexdigest()

    return SourceGraph(
        triples=triples,
        graph=graph,
        checksums=checksums,
        master_checksum=master_checksum,
        source_sentences=source_sentences
    )
