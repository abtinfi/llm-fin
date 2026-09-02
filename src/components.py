"""
The two additive contributions on top of the base LLM:

  RAG          -- TF-IDF retrieval over a small guideline corpus
  SymbolicGate -- fact extraction from the vignette text + constraint check

IMPORTANT (report this honestly): the symbolic extractor reads the vignette
text with regexes. It does NOT read the dataset's structured `facts` field.
Because the vignettes are templated, extraction is near-perfect, so the gate's
measured contribution is an UPPER BOUND. On free-text notes (MIMIC-IV) the
extractor will degrade and the gap will shrink. This is stated as a limitation
rather than hidden.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------- RAG -----------------------------------------

class TfidfRetriever:
    def __init__(self, corpus_path: str, k: int = 3):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = [json.loads(l) for l in Path(corpus_path).open()]
        self.texts = [d["text"] for d in self.docs]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                   stop_words="english")
        self.matrix = self.vec.fit_transform(self.texts)
        self.k = k

    def retrieve(self, query: str) -> List[Dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        q = self.vec.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        order = sims.argsort()[::-1][:self.k]
        return [{"id": self.docs[i]["id"], "text": self.texts[i],
                 "score": float(sims[i])} for i in order]


RAG_TEMPLATE = (
    "You are reviewing a proposed prescription for safety.\n"
    "Use the retrieved clinical guidance below if it is relevant.\n\n"
    "Retrieved guidance:\n{context}\n\n"
    "Answer with exactly one word on the first line: SAFE or UNSAFE.\n"
    "Then give one short sentence of justification.\n\n"
    "Case:\n{vignette}\n\n"
    "Is the proposed prescription safe for this patient?"
)


def build_rag_prompt(vignette: str, retriever: TfidfRetriever):
    docs = retriever.retrieve(vignette)
    ctx = "\n".join(f"- {d['text']}" for d in docs)
    return RAG_TEMPLATE.format(context=ctx, vignette=vignette), docs


# --------------------------- Symbolic gate ---------------------------------

@dataclass
class GateResult:
    fired: bool
    violated: bool          # a hard constraint is violated by the case
    rule: Optional[str]
    extracted: Dict


NUM = r"(\d+(?:\.\d+)?)"

EXTRACTORS = {
    "egfr":         re.compile(r"eGFR (?:is |of )?" + NUM, re.I),
    "potassium":    re.compile(r"potassium is " + NUM, re.I),
    "inr":          re.compile(r"INR is " + NUM, re.I),
    "qtc":          re.compile(r"QTc of " + NUM, re.I),
    "age":          re.compile(r"(\d+)-year-old", re.I),
}

BOOL_EXTRACTORS = {
    "pregnant": (re.compile(r"\bis \d+ weeks pregnant\b", re.I),
                 re.compile(r"\b(not pregnant|post-menopausal|negative pregnancy test)\b", re.I)),
    "severe_asthma": (re.compile(r"\b(severe (persistent )?asthma|brittle asthma)\b", re.I),
                      re.compile(r"\bno (history of asthma|respiratory disease)\b", re.I)),
    "clarithromycin": (re.compile(r"\bclarithromycin\b", re.I),
                       re.compile(r"\b(amoxicillin|doxycycline|no other regular medication)\b", re.I)),
}


def extract_facts(vignette: str) -> Dict:
    facts = {}
    for k, rx in EXTRACTORS.items():
        m = rx.search(vignette)
        if m:
            facts[k] = float(m.group(1))
    for k, (pos, neg) in BOOL_EXTRACTORS.items():
        if pos.search(vignette):
            facts[k] = True
        elif neg.search(vignette):
            facts[k] = False
    return facts


def _check(constraint, facts) -> Optional[bool]:
    var = constraint["var"]
    if var not in facts:
        return None
    v, t, op = facts[var], constraint["threshold"], constraint["op"]
    if op == "<":
        return v < t
    if op == ">":
        return v > t
    if op == "==":
        return v == t
    raise ValueError(op)


class SymbolicGate:
    """
    Checks the case against the constraint graph. Only rules whose drug is
    actually mentioned in the vignette are evaluated.
    """

    def __init__(self, families):
        self.families = families

    def __call__(self, vignette: str) -> GateResult:
        facts = extract_facts(vignette)
        for fam in self.families:
            if fam.drug.lower() not in vignette.lower():
                continue
            res = _check(fam.constraint, facts)
            if res is None:
                continue
            if res:
                return GateResult(True, True, fam.name, facts)
            return GateResult(True, False, fam.name, facts)
        return GateResult(False, False, None, facts)
