import sys
import os

# Add the project root to sys.path to allow importing from shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.extractor import _extract_svo_triples

def test_extraction():
    test_sentences = [
        "Apple did not release the car.",
        "Apple successfully dodged bankruptcy if the market holds.",
        "The car was released by Apple in 2021.",
        "SpaceX successfully launched the Starship rocket on Tuesday."
    ]
    
    for sentence in test_sentences:
        print(f"\nProcessing: {sentence}")
        triples = _extract_svo_triples(sentence)
        for t in triples:
            print(f"  Triple: {t.as_text()}")
            print(f"    - Subject: {t.subject}")
            print(f"    - Verb: {t.verb}")
            print(f"    - Object: {t.object}")
            print(f"    - Negated: {t.is_negated}")
            print(f"    - Modality: {t.modality}")
            print(f"    - Condition: {t.condition}")
            print(f"    - Anchors: {t.temporal_anchors}")

if __name__ == "__main__":
    test_extraction()
