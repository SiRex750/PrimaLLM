from __future__ import annotations

from types import SimpleNamespace

from shared import extractor as extractor_module
from shared.triple import KnowledgeTriple


class FakeToken:
    def __init__(self, text: str, dep_: str = "", pos_: str = "", lemma_: str = "") -> None:
        self.text = text
        self.dep_ = dep_
        self.pos_ = pos_
        self.lemma_ = lemma_ or text
        self.subtree = SimpleNamespace(text=text)


class FakeSentence:
    def __init__(self, tokens):
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


class FakeDoc:
    def __init__(self, sentences):
        self.sents = sentences


class FakeNLP:
    def __init__(self, doc):
        self._doc = doc

    def __call__(self, text: str):
        return self._doc


def test_extract_knowledge_triples(monkeypatch):
    sent = FakeSentence(
        [
            FakeToken("Caesar", dep_="nsubj"),
            FakeToken("conquered", dep_="ROOT", pos_="VERB", lemma_="conquer"),
            FakeToken("Gaul", dep_="dobj"),
        ]
    )
    monkeypatch.setattr(extractor_module, "_load_spacy_model", lambda: FakeNLP(FakeDoc([sent])))

    triples = extractor_module.extract_knowledge_triples("Caesar conquered Gaul.")

    assert triples == [KnowledgeTriple(subject="Caesar", verb="conquer", object="Gaul")]
