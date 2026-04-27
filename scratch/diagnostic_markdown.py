import sys
import os

# Add the project root to sys.path to allow importing from shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.extractor import extract_markdown_triples

def test_markdown_extraction():
    markdown_text = """
# Apple Inc.
Apple released the iPhone in 2007.

## Bankruptcy
Apple successfully dodged bankruptcy if the market holds.
Apple did not fail.
"""
    
    print("Testing Markdown Extraction...")
    triples = extract_markdown_triples(markdown_text)
    
    for t in triples:
        print(f"\n  Triple: {t.as_text()}")
        print(f"    - Subject: {t.subject}")
        print(f"    - Verb: {t.verb}")
        print(f"    - Object: {t.object}")
        print(f"    - Extraction Method: {t.extraction_method}")
        if t.is_negated: print(f"    - Negated: {t.is_negated}")
        if t.modality: print(f"    - Modality: {t.modality}")
        if t.condition: print(f"    - Condition: {t.condition}")

if __name__ == "__main__":
    test_markdown_extraction()
