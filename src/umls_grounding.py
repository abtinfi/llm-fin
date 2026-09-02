"""
UMLS grounding and the Causal Knowledge Graph (proposal 4.2, Aims 1 and 3).

WHY THIS FILE EXISTS
--------------------
Three sentences in the proposal have had no code behind them until now:

  4.2  "The Structural Causal Model (SCM) is initialized from biomedical
        ontologies (e.g. UMLS) and refined using expert-curated causal
        relations."
  4.4  features are to be mapped to "expert-annotated semantic concepts";
        `sae.py` instead matches 11 regexes written from memory, so
        `S_semantic` currently means "matches my regex", not "selects a
        biomedical concept".
  Fig 1 has a "Causal Knowledge Graph" box. No such object exists anywhere in
        the repository.

This module supplies the missing pieces: a cached UMLS client, CUI-anchored
concept groundings that replace the regex vocabulary, and a typed, provenance-
tagged `CausalKnowledgeGraph` that `constraint_layer.py` can query so that
`L_ontology` consults an ontology rather than a constant.

WHAT UMLS CAN AND CANNOT DO HERE -- read this before believing a number
----------------------------------------------------------------------
1. **UMLS supplies terms, not numbers.** There is no CUI for "eGFR below 30".
   The ontology gives the qualitative edge (metformin is contraindicated with
   kidney failure) and the *threshold* stays expert-curated, sourced from FDA
   labelling via `fetch_openfda.py`. So the regexes do not vanish: the lexical
   alternation is replaced by UMLS synonyms, and the numeric tail that captures
   `2.1 mg/dL` remains a regex, because nothing in UMLS can express it. Any
   claim that this module "replaces the regexes" is only half true and the
   docstring on `ConceptGrounding.matcher` restates the limit at the point of
   use.

2. **MED-RT coverage is partial, and that is a finding, not a bug.** Measured
   against this project's 10 rule families on 2026-09-01 (release 2026AA):

     metformin    -> contraindicated_with_disease -> Kidney Failure     FOUND
     propranolol  -> contraindicated_with_disease -> Asthma             FOUND
                     ... and has_physiologic_effect -> Bronchoconstriction,
                     which is the mechanism the rule is about
     ondansetron  -> nothing about QT prolongation                      ABSENT

   The ondansetron/QTc family -- one of the two held-out families, and the one
   carrying the Aim 3 result -- is NOT recoverable from MED-RT. `coverage`
   below reports this per family the same way the symbolic gate reports its
   42.7% applicability, and for the same reason: a grounding that silently
   falls back to hand-written knowledge while claiming ontology provenance
   would be the single most misleading thing this module could do.

3. **The network is not in any scoring path.** Every response is cached to
   disk as JSON. A run with a warm cache makes zero HTTP calls and is
   byte-reproducible; `S_semantic` computed today and in six months must agree,
   and a UMLS release bump must be a visible, dated cache refresh rather than
   silent drift. The cache records the release it was built from.

USAGE
  # one-time (or after --refresh): fetch everything the 10 rule families need
  python src/umls_grounding.py build --out data/umls

  # what did the ontology actually cover?
  python src/umls_grounding.py coverage --graph data/umls/causal_graph.json

  # inspect one concept
  python src/umls_grounding.py show --term "propranolol"
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = "https://uts-ws.nlm.nih.gov/rest"

# MED-RT is the vocabulary that actually carries drug-disease clinical
# relations. The plain /relations endpoint is dominated by RxNorm formulation
# noise (tradenames, combination products, `ingredient_of`), none of which is
# causal. Restricting the source is what makes the edge set meaningful.
CLINICAL_SABS = "MED-RT"

# Relation labels kept, and the graph edge each becomes. Everything else is
# dropped rather than guessed at.
RELATION_EDGE = {
    "contraindicated_with_disease": "CONTRAINDICATED_WITH",
    "may_treat": "MAY_TREAT",
    "may_prevent": "MAY_PREVENT",
    "has_physiologic_effect": "HAS_PHYSIOLOGIC_EFFECT",
    "has_mechanism_of_action": "HAS_MECHANISM",
    "has_pharmacokinetics": "HAS_PHARMACOKINETICS",
    "induces": "INDUCES",
}

# UMLS semantic-type groups -> the node kind this project reasons about.
# Semantic types come from UMLS itself (TUIs); the grouping is ours.
TUI_KIND = {
    "T109": "drug", "T121": "drug", "T110": "drug", "T200": "drug",
    "T195": "drug", "T125": "drug", "T129": "drug",
    "T047": "disease", "T046": "disease", "T191": "disease",
    "T019": "disease", "T020": "disease", "T037": "disease",
    "T184": "finding", "T033": "finding", "T032": "finding",
    "T040": "process", "T042": "process", "T039": "process",
    "T043": "process", "T044": "process", "T045": "process",
    "T034": "lab", "T059": "lab", "T060": "lab",
    "T201": "lab",                      # Clinical Attribute, e.g. QT interval
}


def load_env(path: Path = None) -> None:
    """
    Read KEY=VALUE lines from .env into os.environ without adding a dependency.

    Existing environment variables win, so an explicit `UMLS_API_KEY=... python
    ...` on the command line overrides the file.
    """
    if path is None:
        here = Path(__file__).resolve()
        for cand in (here.parent.parent, here.parent.parent.parent,
                     here.parent.parent.parent.parent):
            if (cand / ".env").is_file():
                path = cand / ".env"
                break
    if path is None or not Path(path).is_file():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# 1. Cached REST client
# ---------------------------------------------------------------------------

class UMLSClient:
    """
    Thin UTS REST wrapper with a mandatory on-disk cache.

    The cache is keyed by the full request path plus sorted query parameters,
    minus the API key -- so a cache directory can be committed or shared
    without leaking the credential. `offline=True` turns a cache miss into an
    error instead of a network call, which is what every downstream scoring
    script should use.
    """

    def __init__(self, api_key: Optional[str] = None,
                 cache_dir: Path = Path("data/umls/cache"),
                 offline: bool = False, sleep: float = 0.05,
                 timeout: int = 30):
        load_env()
        self.api_key = api_key or os.environ.get("UMLS_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.sleep = sleep
        self.timeout = timeout
        self.n_hit = self.n_miss = 0
        if not self.offline and not self.api_key:
            raise SystemExit(
                "UMLS_API_KEY is not set. Put it in .env or the environment, "
                "or construct UMLSClient(offline=True) to run from cache.")

    # -- cache ------------------------------------------------------------
    def _key(self, path: str, params: Dict[str, str]) -> str:
        safe = {k: v for k, v in params.items() if k != "apiKey"}
        raw = path + "?" + urllib.parse.urlencode(sorted(safe.items()))
        return re.sub(r"[^A-Za-z0-9._-]", "_", raw)[:180]

    def _get(self, path: str, **params) -> Optional[dict]:
        cf = self.cache_dir / (self._key(path, params) + ".json")
        if cf.is_file():
            self.n_hit += 1
            return json.loads(cf.read_text())
        if self.offline:
            raise SystemExit(
                f"offline=True and no cached response for {path} {params}.\n"
                f"Run `python src/umls_grounding.py build` with network access "
                f"first.")
        self.n_miss += 1
        q = dict(params, apiKey=self.api_key)
        url = f"{BASE}{path}?" + urllib.parse.urlencode(q)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            # 404 is a legitimate answer ("no such concept / no relations"),
            # and is cached as null so it is not re-fetched forever.
            if e.code == 404:
                data = None
            else:
                raise
        cf.write_text(json.dumps(data))
        time.sleep(self.sleep)
        return data

    # -- endpoints --------------------------------------------------------
    def search(self, term: str, page_size: int = 5) -> List[dict]:
        d = self._get("/search/current", string=term, pageSize=page_size)
        return ((d or {}).get("result") or {}).get("results") or []

    def best_cui(self, term: str) -> Optional[Tuple[str, str]]:
        """First non-placeholder hit. Returns (cui, preferred name)."""
        for r in self.search(term):
            if r.get("ui") and r["ui"] != "NONE":
                return r["ui"], r.get("name", term)
        return None

    def concept(self, cui: str) -> Optional[dict]:
        d = self._get(f"/content/current/CUI/{cui}")
        return (d or {}).get("result")

    def semantic_types(self, cui: str) -> List[Tuple[str, str]]:
        """[(TUI, name)] straight from UMLS -- never inferred."""
        c = self.concept(cui) or {}
        out = []
        for st in c.get("semanticTypes") or []:
            tui = (st.get("uri") or "").rsplit("/", 1)[-1]
            out.append((tui, st.get("name", "")))
        return out

    def atoms(self, cui: str, page_size: int = 100) -> List[str]:
        """
        English surface forms for a concept -- the synonym list that replaces a
        hand-written regex alternation.
        """
        d = self._get(f"/content/current/CUI/{cui}/atoms",
                      pageSize=page_size, language="ENG")
        names = {a.get("name", "").strip()
                 for a in ((d or {}).get("result") or []) if a.get("name")}
        return sorted(n for n in names if n)

    def relations(self, cui: str, sabs: str = CLINICAL_SABS,
                  page_size: int = 100) -> List[dict]:
        d = self._get(f"/content/current/CUI/{cui}/relations",
                      pageSize=page_size, sabs=sabs)
        return (d or {}).get("result") or []

    def source_to_cui(self, sab: str, ui: str) -> Optional[Tuple[str, str]]:
        """
        MED-RT relation targets are source atom ids (MSH/M0001885), not CUIs.
        Resolve to a CUI so graph nodes are CUI-keyed throughout.
        """
        d = self._get("/search/current", string=ui, sabs=sab,
                      searchType="exact", inputType="sourceUi")
        for r in ((d or {}).get("result") or {}).get("results") or []:
            if r.get("ui") and r["ui"] != "NONE":
                return r["ui"], r.get("name", ui)
        return None

    def release(self) -> str:
        """The UMLS release the cache was built from, for provenance."""
        for r in self.search("atrial fibrillation", page_size=1):
            m = re.search(r"/content/(\d{4}[A-Z]{2})/", r.get("uri", ""))
            if m:
                return m.group(1)
        return "unknown"


# ---------------------------------------------------------------------------
# 2. Concept grounding -- what replaces CONCEPT_PATTERNS in sae.py
# ---------------------------------------------------------------------------

# Numeric tails UMLS cannot express. Keyed by the *project* concept slug, not
# by CUI, because the unit is a property of how the note is written rather than
# of the concept. Kept deliberately small and explicit.
NUMERIC_TAIL = {
    "creatinine":  r"[^.\n;]{0,30}?(?P<val>\d+(?:\.\d+)?)\s*(?:mg\s*/\s*d[lL]|[µu]mol\s*/\s*L)",
    "qt_interval": r"[^.\n;]{0,25}?(?P<val>\d+(?:\.\d+)?)\s*(?:ms|msec|milliseconds)\b",
    "egfr":        r"\s*(?:is|of)?\s*(?P<val>\d+(?:\.\d+)?)",
    "heart_rate":  r"[^.\n;]{0,25}?(?P<val>\d+(?:\.\d+)?)",
    "potassium":   r"\s*(?:is |of )?(?P<val>\d+(?:\.\d+)?)",
    "inr":         r"\s*(?:is |of )?(?P<val>\d+(?:\.\d+)?)",
    "age":         None,     # handled by its own pattern below
}

# A few concepts are lexical patterns rather than named entities; UMLS has no
# atom that matches "74-year-old". Declared here so the fallback is visible.
NON_UMLS_PATTERNS = {
    "age": r"\d+(?:\.\d+)?[\s-]*(?:year|yr)s?[\s-]*old",
}

# ---------------------------------------------------------------------------
# Curated lexical variants -- the second stage of grounding, and the reason
# this module does not simply delete the regexes.
#
# MEASURED, not assumed: on 120 real prompts, matching on UMLS atoms ALONE
# collapses recall against the hand-written regexes -- eGFR 16 hits -> 0,
# pregnancy 20 -> 2, potassium 4 -> 0, QT 4 -> 0. The cause is visible in the
# atom lists themselves:
#
#   egfr        UMLS has only "Estimated Glomerular Filtration Rate".
#               Clinical notes write "eGFR".
#   qt_interval UMLS has "QT interval". Notes write "QTc".
#   pregnancy   UMLS has "Pregnancy"/"Gestation" (nouns). Notes write
#               "pregnant" (adjective) -- UMLS carries no inflected forms.
#   potassium   UMLS has "Serum potassium". Notes write "potassium is 5.8".
#
# So UMLS surface forms are terminology-normalised and do not cover clinical
# prose morphology. The grounding is therefore a UNION of two sources with the
# same two-stage structure as the causal graph itself: ontology-initialised,
# expert-refined. `ConceptGrounding` counts each source separately and
# `cmd_audit` reports the split, so "UMLS-grounded" can never be claimed for a
# concept whose matches actually came from this table.
#
# UMLS is not merely decoration here: it contributes real terms the
# hand-written regexes lacked ("Gestation", "Cardiac rate", "Q-T interval",
# and 192 drug surface forms including brand names).
LEXICAL_VARIANTS = {
    "egfr":        ["eGFR", "GFR", "estimated GFR"],
    "qt_interval": ["QTc", "QT", "QT duration", "corrected QT"],
    "pregnancy":   ["pregnant", "hCG", "menstrual", "gravid"],
    "potassium":   ["potassium", "K+"],
    "heart_rate":  ["pulse", "heart rate", "HR"],
    "creatinine":  ["creatinine", "creatine", "SCr"],
    "inr":         ["INR"],
    "renal_disease": ["renal failure", "renal impairment", "renal insufficiency",
                      "kidney disease", "kidney injury", "dialysis", "CKD"],
    "asthma":      ["asthma", "asthmatic", "bronchospasm", "salbutamol",
                    "inhaler", "wheeze", "wheezing"],
}

# Synonyms UMLS returns that are too generic to match on. A concept's atom list
# includes single letters and bare abbreviations that would fire on unrelated
# text; matching on them would inflate S_semantic for free.
SYNONYM_MIN_LEN = 3
SYNONYM_STOP = {"age", "drug", "test", "rate", "level", "value", "product",
                "agent", "other", "unit", "mass", "total", "index"}


@dataclass
class ConceptGrounding:
    """One clinical concept, anchored to a CUI rather than to a regex."""
    slug: str                       # project-local name, e.g. "creatinine"
    cui: str
    preferred_name: str
    tuis: List[str] = field(default_factory=list)
    tui_names: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    lexical_variants: List[str] = field(default_factory=list)
    kind: str = "unknown"
    numeric: bool = False
    source: str = "umls"            # "umls" | "lexical-fallback"

    @property
    def n_umls_terms(self) -> int:
        return len(self.synonyms)

    @property
    def n_curated_terms(self) -> int:
        return len(self.lexical_variants)

    def matcher(self) -> "re.Pattern":
        """
        Compile a matcher from UMLS synonyms, plus a numeric tail where the
        concept is a measurement.

        THE HONEST BIT: the alternation is UMLS-derived, the numeric tail is
        not and cannot be. A match therefore means "a UMLS surface form of this
        CUI, optionally followed by a number in the expected unit" -- which is
        a strictly stronger claim than the old regexes made, but is not "UMLS
        parsed the value".
        """
        if self.slug in NON_UMLS_PATTERNS:
            return re.compile(NON_UMLS_PATTERNS[self.slug], re.I)
        terms = [t for t in self.synonyms
                 if len(t) >= SYNONYM_MIN_LEN and t.lower() not in SYNONYM_STOP]
        # The curated layer is exempt from the stop-list: "QT" and "K+" are
        # short and generic in general text but unambiguous in a clinical note,
        # and they are here precisely because a human decided that.
        terms = terms + list(self.lexical_variants)
        if not terms:
            terms = [self.preferred_name]
        # longest first so "renal failure" wins over "failure"
        alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
        tail = NUMERIC_TAIL.get(self.slug) if self.numeric else None
        pat = rf"\b(?:{alt})\b" + (tail or "")
        return re.compile(pat, re.I)


def ground_concept(cli: UMLSClient, slug: str, term: str,
                   numeric: bool = False,
                   extra_terms: Tuple[str, ...] = ()) -> ConceptGrounding:
    """
    Resolve one project concept to a CUI and pull its semantic types + synonyms.

    `extra_terms` lets a project concept span more than one CUI's vocabulary
    (the `drug` concept is the union of ten drugs). Their atoms are merged; the
    CUI recorded is the primary term's, and the merge is visible in `synonyms`.
    """
    if slug in NON_UMLS_PATTERNS:
        return ConceptGrounding(slug=slug, cui="", preferred_name=term,
                                kind="lexical", numeric=numeric,
                                source="lexical-fallback")
    hit = cli.best_cui(term)
    if hit is None:
        return ConceptGrounding(slug=slug, cui="", preferred_name=term,
                                kind="unknown", numeric=numeric,
                                source="lexical-fallback")
    cui, name = hit
    sts = cli.semantic_types(cui)
    syns = set(cli.atoms(cui))
    for t in extra_terms:
        h = cli.best_cui(t)
        if h:
            syns |= set(cli.atoms(h[0]))
    kinds = [TUI_KIND.get(t) for t, _ in sts if TUI_KIND.get(t)]
    return ConceptGrounding(
        slug=slug, cui=cui, preferred_name=name,
        tuis=[t for t, _ in sts], tui_names=[n for _, n in sts],
        synonyms=sorted(syns),
        lexical_variants=list(LEXICAL_VARIANTS.get(slug, [])),
        kind=(kinds[0] if kinds else "unknown"),
        numeric=numeric, source="umls")


# ---------------------------------------------------------------------------
# 3. The Causal Knowledge Graph (proposal Figure 1, section 4.2)
# ---------------------------------------------------------------------------

def norm_atom(s: str) -> str:
    """
    Normalise a UMLS surface form for concordance testing.

    Strips the qualifier suffixes UMLS appends per source ("(disorder)",
    "[disease/finding]") and collapses whitespace/case. Nothing clever: the
    point is only to decide whether two CUIs share a name, not to do
    terminology normalisation.
    """
    s = re.sub(r"\((?:disorder|finding|procedure|observable entity|"
               r"substance|product|qualifier value)\)", " ", s, flags=re.I)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    return " ".join(s.lower().split())


@dataclass
class Node:
    cui: str
    name: str
    kind: str                       # drug | disease | lab | finding | process
    tuis: List[str] = field(default_factory=list)
    # Normalised English surface forms, stored so `coverage()` can test
    # concordance between two CUIs without any network access.
    atom_keys: List[str] = field(default_factory=list)


@dataclass
class Edge:
    src: str                        # CUI
    dst: str                        # CUI
    rel: str                        # CONTRAINDICATED_WITH | MAY_TREAT | ...
    provenance: str                 # "umls" | "curated"
    source_vocab: str = ""          # MED-RT, or the curated rule family name
    threshold: Optional[dict] = None   # curated numeric constraint, if any


class CausalKnowledgeGraph:
    """
    The SCM object of proposal 4.2, built in the two stages the proposal names.

        initialized from biomedical ontologies   -> provenance="umls"
        refined using expert-curated relations   -> provenance="curated"

    Every edge carries its provenance, so "what does the ontology actually
    know, as opposed to what did we write down" is a query rather than an
    argument. `coverage()` answers it.

    The clinical pathway of section 4 --
        Disease -> Physiological mechanism -> Biomarker -> Treatment response
    -- is representable because MED-RT supplies HAS_PHYSIOLOGIC_EFFECT and
    HAS_MECHANISM alongside CONTRAINDICATED_WITH. The metformin and propranolol
    families realise it fully; see coverage() for the ones that do not.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.groundings: Dict[str, ConceptGrounding] = {}
        self.meta: Dict[str, str] = {}

    # -- construction -----------------------------------------------------
    def add_node(self, cui: str, name: str, kind: str, tuis=None,
                 atom_keys=None) -> None:
        if not cui:
            return
        if cui not in self.nodes:
            self.nodes[cui] = Node(cui, name, kind, list(tuis or []),
                                   sorted(set(atom_keys or [])))
        elif atom_keys and not self.nodes[cui].atom_keys:
            self.nodes[cui].atom_keys = sorted(set(atom_keys))

    def add_edge(self, src: str, dst: str, rel: str, provenance: str,
                 source_vocab: str = "", threshold: dict = None) -> None:
        if not (src and dst):
            return
        for e in self.edges:
            if (e.src, e.dst, e.rel, e.provenance) == (src, dst, rel, provenance):
                return
        self.edges.append(Edge(src, dst, rel, provenance, source_vocab,
                               threshold))

    # -- queries ----------------------------------------------------------
    def out_edges(self, cui: str, rel: str = None) -> List[Edge]:
        return [e for e in self.edges
                if e.src == cui and (rel is None or e.rel == rel)]

    def contraindications(self, drug_cui: str) -> List[Edge]:
        return self.out_edges(drug_cui, "CONTRAINDICATED_WITH")

    def is_contraindicated(self, drug_cui: str, finding_cui: str) -> Optional[Edge]:
        """
        The contraindication edge for this pair, preferring the ontology-
        attested one when both exist.

        Both CAN exist: stage A writes a curated edge for every rule family and
        stage B independently writes a UMLS edge where MED-RT agrees. Returning
        whichever came first in the list would make `ontology_margin` depend on
        insertion order, which is how this silently returned the curated margin
        for propranolol/asthma even though MED-RT attests it.
        """
        hits = [e for e in self.contraindications(drug_cui)
                if e.dst == finding_cui]
        if not hits:
            return None
        return next((e for e in hits if e.provenance == "umls"), hits[0])

    def mechanism_annotations(self, drug_cui: str) -> List[Edge]:
        """
        The drug's physiologic effects and mechanisms of action, as MED-RT
        asserts them.

        DELIBERATELY NOT CALLED A PATH. The proposal's section 4 chain is
        Disease -> Physiological mechanism -> Biomarker -> Treatment response,
        and it is tempting to read propranolol -> Bronchoconstriction ->
        Asthma out of this. MED-RT does not license that: it asserts
        `propranolol has_physiologic_effect Bronchoconstriction` and
        `propranolol contraindicated_with_disease Asthma` as two independent
        facts, with NO edge connecting Bronchoconstriction to Asthma. Joining
        them is a clinician's inference, not the ontology's.

        So this returns the annotations available for a drug, unordered and
        unfiltered. An earlier version returned them as a "path" alongside the
        contraindication edge, which made propranolol/asthma look like a
        derived mechanism when four of the ten effects returned (arterial
        vasodilation, negative inotropy, ...) have nothing to do with asthma.
        """
        return [e for e in self.edges
                if e.src == drug_cui
                and e.rel in ("HAS_PHYSIOLOGIC_EFFECT", "HAS_MECHANISM")]

    def ontology_margin(self, drug_cui: str, finding_cui: str,
                        base: float = 1.0) -> float:
        """
        What `L_ontology` in constraint_layer.py should consult.

        Returns the required SAFE-minus-UNSAFE margin for this (drug, finding)
        pair, or 0.0 when the graph licenses no contraindication -- in which
        case the term must NOT fire. Today the hinge applies `base` to every
        pair unconditionally, which is why the term's name outruns its content.

        The scale is deliberately coarse: an ontology-attested edge with a
        mechanism behind it earns a wider required margin than one asserted
        only by a curated rule. It is not a probability and must not be
        reported as one.
        """
        if self.is_contraindicated(drug_cui, finding_cui) is None:
            return 0.0
        # An ontology-attested contraindication earns a wider required margin
        # than one resting on curation alone. The scale is coarse and is NOT a
        # probability; it must never be reported as one. It does not depend on
        # mechanism annotations, because MED-RT does not connect an effect to
        # the finding (see `mechanism_annotations`).
        #
        # Attestation uses the SAME concordance test as `coverage()`. Testing
        # only for an exact-CUI UMLS edge here made this method contradict
        # coverage() on spironolactone, which MED-RT attests under C0020461
        # ("Increased circulating potassium concentration") while the curated
        # rule names C5979967 ("Hyperkalemia"). One of the two answers had to
        # be wrong; the concordant one is right.
        return base * (1.25 if self.attestation(drug_cui, finding_cui)
                       in ("exact", "synonymous") else 1.0)

    def attestation(self, drug_cui: str, finding_cui: str) -> str:
        """
        How the ontology stands on this drug/finding pair:
        "exact" | "synonymous" | "curated" (edge exists but MED-RT is silent)
        | "none" (no edge at all). Shared by `coverage` and `ontology_margin`
        so the two can never disagree.
        """
        umls_targets = [e.dst for e in self.edges
                        if e.rel == "CONTRAINDICATED_WITH"
                        and e.provenance == "umls" and e.src == drug_cui]
        if finding_cui in umls_targets:
            return "exact"
        if any(self.concordant(finding_cui, t) for t in umls_targets):
            return "synonymous"
        return ("curated"
                if self.is_contraindicated(drug_cui, finding_cui) else "none")

    def concordant(self, cui_a: str, cui_b: str) -> bool:
        """
        Do two CUIs denote the same clinical concept?

        UMLS routinely carries the same finding under more than one CUI --
        MED-RT points spironolactone at C0020461 "Increased circulating
        potassium concentration", while searching "hyperkalemia" returns
        C5979967 "Hyperkalemia (disorder)". They share six English surface
        forms (hyperkalemia, hyperkalaemia, hyperpotassemia, ...). Testing
        CUI equality alone reports that rule as ontology-absent, which is
        false.

        The test is deliberately *lexical concordance between two UMLS
        concepts' own atom sets*, not a hierarchy walk and not string
        similarity. It says "these two CUIs are called the same thing by some
        source vocabulary". It is conservative in the right direction: Kidney
        Failure (C0035078) and Kidney Diseases (C0022658) share zero surface
        forms and stay distinct, which is correct, because the MED-RT edge
        really is about a broader concept than the curated rule.
        """
        if cui_a == cui_b:
            return True
        a = self.nodes.get(cui_a), self.nodes.get(cui_b)
        if not all(a) or not (a[0].atom_keys and a[1].atom_keys):
            return False
        return bool(set(a[0].atom_keys) & set(a[1].atom_keys))

    def coverage(self) -> dict:
        """
        How much of the rule set the ontology actually attests, family by
        family, in three states rather than two:

          exact       MED-RT asserts the contraindication against the same CUI
          synonymous  ... against a CUI that shares a surface form (see
                      `concordant`) -- the same clinical claim under a
                      different identifier
          absent      MED-RT asserts nothing about this drug/finding pair

        This fraction is the module's uncircular number, and it is the one to
        quote -- exactly as the symbolic gate's 42.7% applicability is quoted
        beside its 0.986 accuracy. An `absent` family rests entirely on the
        curated threshold, precisely as it did before this module existed.
        """
        fams = {}
        for e in self.edges:
            if e.rel != "CONTRAINDICATED_WITH" or e.provenance != "curated":
                continue
            umls_targets = [x.dst for x in self.edges
                            if x.rel == "CONTRAINDICATED_WITH"
                            and x.provenance == "umls" and x.src == e.src]
            st = self.attestation(e.src, e.dst)
            if st == "exact":
                state, matched = "exact", e.dst
            elif st == "synonymous":
                matched = next((t for t in umls_targets
                                if self.concordant(e.dst, t)), None)
                state = "synonymous"
            else:
                state, matched = "absent", None
            fams[e.source_vocab] = {
                "state": state,
                "drug": self.nodes[e.src].name if e.src in self.nodes else e.src,
                "curated_finding": (self.nodes[e.dst].name
                                    if e.dst in self.nodes else e.dst),
                "curated_finding_cui": e.dst,
                "umls_matched_cui": matched,
                "umls_matched_name": (self.nodes[matched].name
                                      if matched in self.nodes else None),
                "threshold": e.threshold,
                "n_umls_contraindications": len(umls_targets),
            }
        n = len(fams)
        k = sum(1 for r in fams.values() if r["state"] != "absent")
        return {"families": fams, "n_families": n, "n_umls_attested": k,
                "fraction_umls_attested": (k / n if n else 0.0)}

    # -- io ---------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "meta": self.meta,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
            "groundings": {k: asdict(v) for k, v in self.groundings.items()},
        }

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CausalKnowledgeGraph":
        d = json.loads(Path(path).read_text())
        g = cls()
        g.meta = d.get("meta", {})
        for n in d["nodes"]:
            g.nodes[n["cui"]] = Node(**n)
        for e in d["edges"]:
            g.edges.append(Edge(**e))
        for k, v in (d.get("groundings") or {}).items():
            g.groundings[k] = ConceptGrounding(**v)
        return g

    def concept_patterns(self) -> Dict[str, "re.Pattern"]:
        """
        Drop-in replacement for `sae.CONCEPT_PATTERNS`, so `sae.py` can be
        switched from hand-written regexes to UMLS groundings without any other
        change to its scoring code.
        """
        return {slug: g.matcher() for slug, g in self.groundings.items()}


# ---------------------------------------------------------------------------
# 4. Building the graph from this project's rule families
# ---------------------------------------------------------------------------

# The project's concept vocabulary, expressed as UMLS lookups. `numeric` marks
# the ones whose regex tail survives because UMLS cannot express a value.
CONCEPT_TERMS = {
    "creatinine":    ("serum creatinine measurement", True, ()),
    "qt_interval":   ("QT interval", True, ()),
    "egfr":          ("estimated glomerular filtration rate", True, ()),
    "heart_rate":    ("heart rate", True, ()),
    "potassium":     ("serum potassium measurement", True, ()),
    "inr":           ("international normalized ratio", True, ()),
    "age":           ("age", False, ()),          # lexical fallback
    "drug":          ("metformin", False,
                      ("ibuprofen", "lisinopril", "propranolol", "aspirin",
                       "spironolactone", "warfarin", "simvastatin",
                       "nitrofurantoin", "ondansetron")),
    "renal_disease": ("kidney failure", False, ("renal insufficiency",
                                                "chronic kidney disease")),
    "pregnancy":     ("pregnancy", False, ()),
    "asthma":        ("asthma", False, ()),
}

# The finding each rule family turns on, as a UMLS-searchable term. This is the
# expert-curated half of proposal 4.2: a human asserts that `metformin_renal`
# is about kidney failure. UMLS is then asked whether it agrees.
FAMILY_FINDING = {
    "metformin_renal":             "kidney failure",
    "nsaid_renal":                 "kidney failure",
    "acei_pregnancy":              "pregnancy",
    "betablocker_asthma":          "asthma",
    "aspirin_reye":                "Reye syndrome",
    "spironolactone_hyperkalaemia": "hyperkalemia",
    "warfarin_inr":                "hemorrhage",
    "statin_macrolide":            "rhabdomyolysis",
    "nitrofurantoin_renal":        "kidney failure",
    "ondansetron_qt":              "long QT syndrome",
}


def build_graph(cli: UMLSClient, verbose: bool = True) -> CausalKnowledgeGraph:
    """
    Two-stage construction, in the order proposal 4.2 specifies.

    Stage A (curated): every rule family in `rules.py` becomes a
      drug --CONTRAINDICATED_WITH--> finding edge carrying its numeric
      threshold. This is the expert-curated layer, and it is what the pipeline
      already relies on; making it an explicit graph is the point.

    Stage B (ontology): MED-RT relations for each drug CUI are pulled and added
      with provenance="umls". Where stage B independently reproduces a stage A
      edge, the rule is ontology-attested. Where it does not, `coverage()` says
      so.
    """
    import rules

    g = CausalKnowledgeGraph()
    g.meta = {"umls_release": cli.release(),
              "built": time.strftime("%Y-%m-%d"),
              "clinical_sabs": CLINICAL_SABS}

    # -- concept groundings (replaces sae.CONCEPT_PATTERNS) ---------------
    for slug, (term, numeric, extra) in CONCEPT_TERMS.items():
        gr = ground_concept(cli, slug, term, numeric, extra)
        g.groundings[slug] = gr
        if gr.cui:
            g.add_node(gr.cui, gr.preferred_name, gr.kind, gr.tuis)
        if verbose:
            print(f"  {slug:14s} {gr.cui or '(lexical)':10s} "
                  f"{gr.preferred_name[:38]:40s} "
                  f"{len(gr.synonyms):4d} synonyms  [{gr.source}]")

    # -- stage A: curated edges from rules.py -----------------------------
    if verbose:
        print("\n  stage A: curated edges from rules.py")
    drug_cui: Dict[str, str] = {}
    for fam in rules.RULE_FAMILIES:
        d = cli.best_cui(fam.drug)
        f_term = FAMILY_FINDING.get(fam.name)
        f = cli.best_cui(f_term) if f_term else None
        if not (d and f):
            if verbose:
                print(f"    {fam.name:30s} UNRESOLVED "
                      f"(drug={bool(d)} finding={bool(f)})")
            continue
        drug_cui[fam.drug] = d[0]
        g.add_node(d[0], d[1], "drug", [t for t, _ in cli.semantic_types(d[0])])
        # Atoms are fetched for findings because coverage() needs them to tell
        # "the ontology is silent" from "the ontology says it under a
        # different CUI".
        g.add_node(f[0], f[1], "disease",
                   [t for t, _ in cli.semantic_types(f[0])],
                   atom_keys=[norm_atom(a) for a in cli.atoms(f[0])])
        g.add_edge(d[0], f[0], "CONTRAINDICATED_WITH", "curated",
                   source_vocab=fam.name, threshold=dict(fam.constraint))
        if verbose:
            print(f"    {fam.name:30s} {d[1][:18]:20s} -| {f[1][:26]:28s} "
                  f"{fam.constraint.get('var')} "
                  f"{fam.constraint.get('op')} {fam.constraint.get('threshold')}")

    # -- stage B: ontology edges from MED-RT ------------------------------
    if verbose:
        print("\n  stage B: MED-RT relations")
    for drug, cui in sorted(drug_cui.items()):
        rels = cli.relations(cui)
        kept = 0
        for r in rels:
            lab = r.get("additionalRelationLabel")
            edge = RELATION_EDGE.get(lab)
            if not edge:
                continue
            rid = (r.get("relatedId") or "")
            m = re.search(r"/source/([^/]+)/([^/?]+)", rid)
            if not m:
                continue
            tgt = cli.source_to_cui(m.group(1), m.group(2))
            if tgt is None:
                continue
            kind = ("disease" if edge in ("CONTRAINDICATED_WITH", "MAY_TREAT",
                                          "MAY_PREVENT") else "process")
            # Only contraindication targets need atoms -- they are the only
            # ones coverage() compares against a curated finding.
            ak = ([norm_atom(a) for a in cli.atoms(tgt[0])]
                  if edge == "CONTRAINDICATED_WITH" else None)
            g.add_node(tgt[0], tgt[1], kind, atom_keys=ak)
            g.add_edge(cui, tgt[0], edge, "umls", source_vocab=CLINICAL_SABS)
            kept += 1
        if verbose:
            print(f"    {drug:16s} {len(rels):4d} MED-RT relations, "
                  f"{kept:3d} kept as causal edges")
    return g


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def cmd_build(args):
    cli = UMLSClient(cache_dir=Path(args.out) / "cache",
                     offline=args.offline)
    if args.refresh and not args.offline:
        n = 0
        for f in (Path(args.out) / "cache").glob("*.json"):
            f.unlink(); n += 1
        print(f"cleared {n} cached responses")
    print(f"building causal knowledge graph (cache={Path(args.out)/'cache'})\n")
    g = build_graph(cli, verbose=True)
    out = Path(args.out) / "causal_graph.json"
    g.save(out)
    cov = g.coverage()
    print(f"\n  nodes {len(g.nodes)}  edges {len(g.edges)}  "
          f"(umls {sum(1 for e in g.edges if e.provenance=='umls')}, "
          f"curated {sum(1 for e in g.edges if e.provenance=='curated')})")
    print(f"  UMLS release {g.meta['umls_release']}  "
          f"cache: {cli.n_hit} hits / {cli.n_miss} fetches")
    print(f"  ontology-attested rule families: {cov['n_umls_attested']}"
          f"/{cov['n_families']} ({cov['fraction_umls_attested']:.1%})")
    print(f"\nwrote {out}")


def cmd_coverage(args):
    g = CausalKnowledgeGraph.load(args.graph)
    cov = g.coverage()
    print(f"UMLS release {g.meta.get('umls_release')}, "
          f"built {g.meta.get('built')}\n")
    print(f"{'rule family':30s} {'state':>11s}  {'curated finding':26s} "
          f"{'MED-RT says':28s} threshold")
    print("-" * 118)
    for fam, r in sorted(cov["families"].items(),
                         key=lambda kv: (kv[1]["state"], kv[0])):
        th = r["threshold"] or {}
        thr = f"{th.get('var','')} {th.get('op','')} {th.get('threshold','')}"
        says = r["umls_matched_name"] or (
            f"(nothing; {r['n_umls_contraindications']} other CIs)")
        print(f"{fam:30s} {r['state']:>11s}  {r['curated_finding'][:25]:26s} "
              f"{says[:27]:28s} {thr}")
    print(f"\nontology-attested: {cov['n_umls_attested']}/{cov['n_families']} "
          f"({cov['fraction_umls_attested']:.1%})")
    print("""
This fraction is the uncircular number for the grounding, and is the one to
quote. Three things it does NOT say:

  - `exact`/`synonymous` does not mean UMLS supplied the threshold. It never
    can: there is no CUI for "eGFR below 30". Every number in the `threshold`
    column is expert-curated and FDA-sourced, before and after this module.
  - `absent` does not mean the rule is wrong. It means MED-RT asserts nothing
    about that drug/finding pair, so the rule rests entirely on curation --
    which is where it rested before, only now it is visible.
  - A drug-drug interaction (statin_macrolide) cannot be attested by a
    drug-DISEASE relation at all. That row is a category mismatch, not a
    coverage failure.""")


def cmd_show(args):
    cli = UMLSClient(cache_dir=Path(args.cache), offline=args.offline)
    hit = cli.best_cui(args.term)
    if not hit:
        print(f"no CUI for {args.term!r}"); return
    cui, name = hit
    print(f"{args.term!r} -> {cui}  {name}")
    for tui, tn in cli.semantic_types(cui):
        print(f"  semantic type  {tui}  {tn}  -> kind={TUI_KIND.get(tui,'?')}")
    syn = cli.atoms(cui)
    print(f"  {len(syn)} English atoms, first 12: {syn[:12]}")
    print(f"  MED-RT relations:")
    for r in cli.relations(cui):
        lab = r.get("additionalRelationLabel")
        if RELATION_EDGE.get(lab):
            print(f"    {lab:32s} {r.get('relatedIdName')}")


def cmd_audit(args):
    """
    Old regexes vs UMLS grounding, on real prompts, per concept.

    Exists because swapping a matching layer silently changes every
    `S_semantic` in Aim 1. The comparison must be a reported number, not an
    assumption -- especially since the UMLS-only version LOST recall badly and
    the curated lexical layer exists to repair exactly that.
    """
    import sae
    g = CausalKnowledgeGraph.load(args.graph)
    new = g.concept_patterns()
    old = sae.CONCEPT_PATTERNS
    recs = []
    for f, n in ((args.data, args.limit), (args.data2, args.limit)):
        if f and Path(f).is_file():
            recs += [json.loads(l) for l in Path(f).open()][:n]
    print(f"{len(recs)} prompts from {args.data} + {args.data2}\n")
    print(f"{'concept':14s} {'CUI':10s} {'UMLS':>5s} {'cur':>4s} "
          f"{'old hits':>9s} {'new hits':>9s} {'umls-only':>10s}  note")
    print("-" * 96)
    tot_o = tot_n = 0
    for k in old:
        gr = g.groundings.get(k)
        o = sum(len(old[k].findall(r["prompt"])) for r in recs)
        n = sum(len(new[k].findall(r["prompt"])) for r in recs) if k in new else 0
        # what UMLS atoms alone would have matched, to expose the split
        uo = 0
        if gr and gr.cui and k not in NON_UMLS_PATTERNS:
            bare = ConceptGrounding(**{**asdict(gr), "lexical_variants": []})
            uo = sum(len(bare.matcher().findall(r["prompt"])) for r in recs)
        tot_o += o; tot_n += n
        note = ("same" if o == n else
                f"{'+' if n > o else ''}{n - o}")
        print(f"{k:14s} {(gr.cui if gr else '-'):10s} "
              f"{(gr.n_umls_terms if gr else 0):5d} "
              f"{(gr.n_curated_terms if gr else 0):4d} "
              f"{o:9d} {n:9d} {uo:10d}  {note}")
    print("-" * 96)
    print(f"{'TOTAL':14s} {'':10s} {'':5s} {'':4s} {tot_o:9d} {tot_n:9d}")
    print("""
Read the `umls-only` column before quoting anything as UMLS-grounded: it is
what the ontology's own surface forms match without the curated lexical layer.
Where it is far below `new hits`, the concept is grounded to a CUI but matched
largely by curated morphology, and S_semantic for it should be described as
CUI-anchored, not UMLS-matched.""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)

    b = sub.add_parser("build", help="fetch UMLS and build the causal graph")
    b.add_argument("--out", default="data/umls")
    b.add_argument("--offline", action="store_true",
                   help="build from cache only; error on any cache miss")
    b.add_argument("--refresh", action="store_true",
                   help="clear the cache first (a dated re-fetch, not silent "
                        "drift)")
    b.set_defaults(fn=cmd_build)

    c = sub.add_parser("coverage", help="what the ontology actually attests")
    c.add_argument("--graph", default="data/umls/causal_graph.json")
    c.set_defaults(fn=cmd_coverage)

    a = sub.add_parser("audit", help="old regexes vs UMLS grounding, measured")
    a.add_argument("--graph", default="data/umls/causal_graph.json")
    a.add_argument("--data",
                   default="data/synthetic_control/counterfactual_test.jsonl")
    a.add_argument("--data2", default="data/medcalc/counterfactual_test.jsonl")
    a.add_argument("--limit", type=int, default=60)
    a.set_defaults(fn=cmd_audit)

    s = sub.add_parser("show", help="inspect one concept")
    s.add_argument("--term", required=True)
    s.add_argument("--cache", default="data/umls/cache")
    s.add_argument("--offline", action="store_true")
    s.set_defaults(fn=cmd_show)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
