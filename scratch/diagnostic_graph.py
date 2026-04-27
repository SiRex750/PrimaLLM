import sys
import os
import networkx as nx

# Add the project root to sys.path to allow importing from shared/caveman
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.triple import KnowledgeTriple
from caveman.core.graph import build_graph, merge_similar_nodes

def test_graph_upgrade():
    triples = [
        KnowledgeTriple(subject="Apple", verb="released", object="iPhone", is_negated=False, modality="successfully"),
        KnowledgeTriple(subject="Apple", verb="released", object="iPhone", is_negated=True, modality="failed to"),
        KnowledgeTriple(subject="Apple Inc.", verb="is", object="technology company")
    ]
    
    print("Building MultiDiGraph...")
    graph = build_graph(triples)
    
    print(f"Nodes: {graph.nodes()}")
    print(f"Edges: {len(graph.edges())} (Expected 3)")
    
    for u, v, k, d in graph.edges(keys=True, data=True):
        print(f"  Edge {u} -> {v}: {d}")

    print("\nTesting Merge Similar Nodes...")
    # Mocking embedder - we just want to see if attributes survive
    class MockEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np
            # Return identical vectors for "Apple" and "Apple Inc." if they appear
            return np.array([[1, 0] if "Apple" in t else [0, 1] for t in texts])

    merged_graph, count = merge_similar_nodes(graph, MockEmbedder(), threshold=0.9)
    print(f"Nodes after merge: {merged_graph.nodes()}")
    print(f"Edges after merge: {len(merged_graph.edges())} (Expected 3)")
    
    for u, v, k, d in merged_graph.edges(keys=True, data=True):
        print(f"  Edge {u} -> {v}: {d}")

if __name__ == "__main__":
    test_graph_upgrade()
