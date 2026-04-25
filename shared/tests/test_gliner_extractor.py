from shared.extractor import extract_claim_triples

def test_gliner_smoke():
    text = (
        "The Apollo 11 mission launched on a Saturn V rocket. "
        "Neil Armstrong and Buzz Aldrin descended to the lunar "
        "surface in the Eagle module. Michael Collins remained "
        "in lunar orbit aboard the Command Module Columbia."
    )

    print(f"Extracting triples from text:\n{text}\n")
    triples = extract_claim_triples(text)

    print(f"Found {len(triples)} triples:")
    for i, t in enumerate(triples, 1):
        print(f" {i}. {t.subject} | {t.verb} | {t.object} ({t.extraction_method})")

    # 2. Assert that len(triples) >= 2
    assert len(triples) >= 2, f"Expected at least 2 triples, found {len(triples)}"

    # 3. Assert that at least one triple has subject containing "Armstrong" OR "Aldrin" (case-insensitive)
    found_crew = any("armstrong" in t.subject.lower() or "aldrin" in t.subject.lower() for t in triples)
    assert found_crew, "Did not find Armstrong or Aldrin in any triple subject"

    # 4. Assert that no triple has a duplicated object span
    for t in triples:
        words = t.object.split()
        # Check for repeated 4+ word sequences
        if len(words) >= 8:
             for i in range(len(words) - 7): # check up to possible repetition
                 seq = " ".join(words[i:i+4])
                 # Find if this sequence repeats later in the string
                 remaining = " ".join(words[i+1:])
                 assert seq not in remaining, f"Found duplicated 4-word sequence '{seq}' in triple: {t.object}"

    print("\nSmoke test PASSED!")

if __name__ == "__main__":
    test_gliner_smoke()
