import pymupdf4llm, re
from shared.extractor import extract_source_triples

def diagnostic():
    pdf_path = "Apple-1.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    
    for page in md_text:
        text = page.get('text', '')
        if "China" in text and "tons" in text.lower():
            print(f"--- Found Page with China and Tons ---")
            print(text[:800])
            print("-" * 30)
            triples = extract_source_triples(text)
            for t in triples:
                if "china" in t.subject.lower() or "china" in t.object.lower():
                    print(f"S: {t.subject} | V: {t.verb} | O: {t.object} | A: {t.temporal_anchors}")
            break

if __name__ == "__main__":
    diagnostic()
