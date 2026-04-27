import sys
import os
import sqlite3
import json

# Add the project root to sys.path to allow importing from shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.triple import KnowledgeTriple
from shared.l3_memory import save_fact, fetch_clean_facts, initialize_l3_memory

def test_persistence():
    db_path = "test_wiki.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    initialize_l3_memory(db_path=db_path)
    
    triple = KnowledgeTriple(
        subject="Apple",
        verb="released",
        object="iPhone",
        modality="successfully",
        is_negated=False,
        condition="if marketing worked",
        temporal_anchors=("2007", "June")
    )
    
    print("Saving N-ary triple...")
    success = save_fact(triple, db_path=db_path)
    print(f"Save success: {success}")
    
    print("\nFetching facts...")
    facts = fetch_clean_facts(db_path=db_path)
    if facts:
        f = facts[0]
        print(f"Fetched fact: {f}")
        print(f"Modality: {f['modality']}")
        print(f"Is Negated: {f['is_negated']} (Type: {type(f['is_negated'])})")
        print(f"Condition: {f['condition']}")
        print(f"Anchors: {f['temporal_anchors']} (Type: {type(f['temporal_anchors'])})")
    else:
        print("No facts found!")

if __name__ == "__main__":
    test_persistence()
