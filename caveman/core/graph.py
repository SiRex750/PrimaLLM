from __future__ import annotations

from typing import Iterable
import networkx as nx

from shared.triple import KnowledgeTriple


def build_graph(triples: Iterable[KnowledgeTriple]) -> nx.DiGraph:
    from collections import Counter
    graph = nx.DiGraph()
    triple_list = list(triples)
    freq: Counter = Counter(t.as_text() for t in triple_list)
    for triple in triple_list:
        if not graph.has_node(triple.subject):
            graph.add_node(triple.subject)
        if not graph.has_node(triple.object):
            graph.add_node(triple.object)
        weight = freq[triple.as_text()]
        if graph.has_edge(triple.subject, triple.object):
            graph[triple.subject][triple.object]["weight"] += weight
        else:
            graph.add_edge(triple.subject, triple.object,
                           verb=triple.verb, weight=weight)
    return graph


def pagerank_scores(graph: nx.DiGraph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, weight="weight")


def rank_triples_by_importance(triples: Iterable[KnowledgeTriple]) -> list[tuple[KnowledgeTriple, float]]:
    triple_list = list(triples)
    graph = build_graph(triple_list)
    scores = pagerank_scores(graph)
    return sorted(
        ((triple, scores.get(triple.subject, 0.0) + scores.get(triple.object, 0.0)) for triple in triple_list),
        key=lambda item: item[1],
        reverse=True,
    )
