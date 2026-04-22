from __future__ import annotations

from typing import Iterable

from shared.triple import KnowledgeTriple


def build_graph(triples: Iterable[KnowledgeTriple]) -> nx.DiGraph:
    import networkx as nx

    graph = nx.DiGraph()
    for triple in triples:
        graph.add_node(triple.subject)
        graph.add_node(triple.object)
        graph.add_edge(triple.subject, triple.object, verb=triple.verb)
    return graph


def pagerank_scores(graph: nx.DiGraph) -> dict[str, float]:
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph)


def rank_triples_by_importance(triples: Iterable[KnowledgeTriple]) -> list[tuple[KnowledgeTriple, float]]:
    triple_list = list(triples)
    graph = build_graph(triple_list)
    scores = pagerank_scores(graph)
    return sorted(
        ((triple, scores.get(triple.subject, 0.0) + scores.get(triple.object, 0.0)) for triple in triple_list),
        key=lambda item: item[1],
        reverse=True,
    )
