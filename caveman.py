from typing import List
import os

import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class KnowledgeTriple(BaseModel):
    subject: str
    verb: str
    object: str


class GraphData(BaseModel):
    triples: List[KnowledgeTriple]


def extract_knowledge_graph(text: str, api_key: str) -> GraphData:
    client = OpenAI(api_key=api_key)

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict information extraction engine. Extract knowledge only as "
                    "Subject-Verb-Object triples from the user text. Each subject, verb, and object "
                    "must be short and concise, using 1 to 3 words only. Preserve factual meaning, "
                    "avoid speculation, and do not include any text outside the required structured "
                    "schema."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format=GraphData,
    )

    return completion.choices[0].message.parsed


def build_and_draw_graph(graph_data: GraphData):
    G = nx.DiGraph()

    for triple in graph_data.triples:
        G.add_node(triple.subject)
        G.add_node(triple.object)
        G.add_edge(triple.subject, triple.object, label=triple.verb)

    # --- NEW PAGERANK LOGIC ---
    print("\n--- PageRank Scores (L2 Importance) ---")
    # Calculate the scores
    pagerank_scores = nx.pagerank(G)
    
    # Sort them from highest to lowest
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Print the results
    for node, score in sorted_nodes:
        print(f"{node}: {score:.4f}")
    # ---------------------------

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', arrows=True)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Caveman L2 Knowledge Graph")
    plt.tight_layout()
    output_path = "knowledge_graph.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved graph image to {output_path}")
    plt.show()


if __name__ == "__main__":
    test_text = """
Julius Caesar conquered Gaul. Julius Caesar defied the Roman Senate.
The Roman Senate feared Julius Caesar. Julius Caesar initiated a civil war.
The civil war destroyed the Roman Republic. Augustus replaced the Roman Republic with the Roman Empire.
"""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to a local .env file or environment variables.")

    print("Starting knowledge graph extraction...")
    graph_data = extract_knowledge_graph(test_text, openai_api_key)
    print(f"Extraction complete. Found {len(graph_data.triples)} triples.")

    print("Building and drawing graph...")
    build_and_draw_graph(graph_data)
    print("Done.")
