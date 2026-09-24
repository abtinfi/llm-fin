"""
Five sanity audits of the v3b pipeline: are the unusual results (the collapse
under swapped answer options, 100% deferral under UQ) the models' behaviour,
or a bug, a leak, or a coding error in the pipeline?

  python src/audit_pipeline_sanity.py      # -> results/mimic_v3b/AUDIT_SANITY.md

  1 ORACLE / CONTROLS  the REAL UQ engine (run_eval.run_variant: calibration
                       pass, calibrate_threshold, the `unc > tau` abstention,
                       the gate) driven by stub models on the real calib/test
                       splits. An oracle must be answered 100% with error 0; a
                       coin flip must be deferred 100%; a model that knows 60%
                       of items must be answered on ~60%; a model confident
                       exactly when WRONG must be deferred (a flipped
                       comparator would answer it instead).
  2 TOKEN ALIGNMENT    (population part; the live generate() check is
                       audit_token_alignment.py) on every stored prediction,
                       sign(logit_margin) must agree with the answer parsed
                       from the generated text: greedy decoding emits the
                       argmax token, so a margin read off the wrong ids would
                       disagree. And uq_uncertainty must equal the binary
                       entropy recomputed from the margin.
  3 DISJOINTNESS       subject_id intersections between every pair of splits,
                       for the main, note and cl datasets.
  4 MANUAL TRACE       10 stratified rows of BioMistral test nsai_uq traced
                       through every step with INDEPENDENT code (own regexes,
                       own CKD-EPI 2021, own thresholds), then the same checks
                       over every row of every UQ-bearing prediction file.
  5 OPTION SWAP        same ids and labels, prompts differing only in the
                       options line, label compared as a WORD (no index), and
                       the swap effect measured again on outputs that name
                       exactly one answer (so an echoed "UNSAFE or SAFE" cannot
                       be driving it).

COUNTS ONLY in the tracked report: v3.1 rows are credentialed data. The one
row-level view (audit 4's trace, with prompts) goes to logs/mimic_v3b/, which
is gitignored, for a human to read locally.
"""

import collections
import io
import json
import math
import random
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CSAI = Path("/home/asosoft/abtin/paper/csai")
sys.path.insert(0, str(CSAI / "src"))

import run_eval as RE                              # noqa: E402
from components import SymbolicGate, TfidfRetriever               # noqa: E402
from metrics import score                          # noqa: E402
from model import Generation, parse_answer         # noqa: E402
from rules import RULE_FAMILIES                    # noqa: E402

D = HERE / "data/mimic_v3b"
RES = HERE / "results/mimic_v3b"
MODELS = ["biomistral-7b", "llama3-openbiollm-8b", "mistral-7b-instruct-v0-2"]
OUT = RES / "AUDIT_SANITY.md"
TRACE = HERE / "logs/mimic_v3b/AUDIT_TRACE_LOCAL.md"
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def read(p):
    return [json.loads(l) for l in open(p)]


# ---------------------------------------------------------------- 1. oracle --
class StubLM:
    """generate() returns a canned Generation per RECORD. run_variant asks for
    the calibration split, then the test split, each in record order, so the
    stub walks those lists by position (a RAG prompt is not the record's
    `prompt`, so keying on text would not work). The length check makes a
    mismatch fail loudly. Nothing else of the pipeline is replaced."""

    def __init__(self, fn, splits):
        self.fn, self.queue = fn, list(splits)

    def attach_adapter(self, *a, **k):
        pass

    def detach_adapter(self):
        pass

    def generate(self, prompts, max_new_tokens=64, batch_size=8):
        recs = self.queue.pop(0)
        assert len(recs) == len(prompts)
        return [self.fn(r) for r in recs]


def gen(answer, margin):
    return Generation(text=answer, entropy=0.0, max_entropy=0.0, n_tokens=1,
                      logit_margin=margin)


def audit_oracle():
    say("## 1. Oracle and controls through the real UQ engine\n")
    calib, test = read(D / "counterfactual_calib.jsonl"), read(D / "counterfactual_test.jsonl")
    truth = lambda r: r["label"]  # noqa: E731
    other = {"SAFE": "UNSAFE", "UNSAFE": "SAFE"}
    sgn = {"SAFE": 1.0, "UNSAFE": -1.0}

    def h(r):  # a deterministic per-item coin
        return random.Random(r["id"]).random()

    stubs = {
        "oracle (always right, P = 1.00)":
            lambda r: gen(truth(r), sgn[truth(r)] * 50.0),
        "random coin, random confidence":
            lambda r: (lambda a: gen(a, sgn[a] * abs(random.Random(r["id"] + "m").gauss(0, 3))))(
                "SAFE" if h(r) < 0.5 else "UNSAFE"),
        "knows 60% (3% of those confidently wrong), unsure on the rest":
            lambda r: (gen(truth(r), sgn[truth(r)] * 8.0) if h(r) < 0.582 else
                       gen(other[truth(r)], sgn[other[truth(r)]] * 8.0) if h(r) < 0.6 else
                       (lambda a: gen(a, sgn[a] * 0.2))(
                           "SAFE" if random.Random(r["id"] + "c").random() < 0.5 else "UNSAFE")),
        "inverted (confident when wrong, unsure when right)":
            lambda r: (gen(other[truth(r)], sgn[other[truth(r)]] * 8.0) if h(r) < 0.5 else
                       gen(truth(r), sgn[truth(r)] * 0.1)),
    }
    gate = SymbolicGate(RULE_FAMILIES)
    retriever = TfidfRetriever(str(D / "rag_corpus.jsonl"), k=3)
    say("| stub model | variant | τ | certifiable | answered, UQ-governed items "
        "| selective error, UQ-governed | coverage, all items | expected |")
    say("|---|---|---|---|---|---|---|---|")
    expect = {"oracle": "answer 100%, error 0",
              "random": "defer 100%", "knows": "answer ≈ 60%, error ≤ 0.10",
              "inverted": "defer ≈ 100%"}
    verdicts = []
    for name, fn in stubs.items():
        for variant in ("uq", "nsai_uq"):
            with redirect_stdout(io.StringIO()):
                recs, tau, diag = RE.run_variant(
                    variant, StubLM(fn, [calib, test]), calib, test, retriever,
                    gate, 0.10, 8,
                    uq_signal="decision_entropy")
            s = score(recs)
            gov = [r for r in recs if not r["gate_fired"]]
            ans = [r for r in gov if not r["abstained"]]
            cov_g = len(ans) / len(gov) if gov else float("nan")
            err_g = (1 - sum(r["pred"] == r["label"] for r in ans) / len(ans)
                     if ans else float("nan"))
            key = name.split()[0]
            ok = {"oracle": cov_g == 1.0 and err_g == 0.0,
                  "random": cov_g == 0.0,
                  "knows": 0.5 <= cov_g <= 0.65 and err_g <= 0.10,
                  "inverted": cov_g <= 0.01}[key]
            verdicts.append(ok)
            say(f"| {name} | {variant} | {tau:.4g} | {diag.get('calib_certifiable')} "
                f"| {cov_g:.4f} | {err_g:.4f} | {s['coverage']:.4f} "
                f"| {expect[key]}: **{'PASS' if ok else 'FAIL'}** |")
    say(f"\nAll {len(verdicts)} checks pass: **{all(verdicts)}**. "
        "Every number is from the unmodified run_variant / calibrate_threshold / "
        "score on the real calibration (66,938 items) and test (166,264 items) "
        "splits; only the language model is replaced.\n")

    say("**The real rows that defer 100%**, from the calibration diagnostics "
        "each run recorded:\n")
    say("| model | split | variant | τ | calib pool n | certifiable | min retained "
        "items needed | coverage here |")
    say("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for split in ("test", "heldout"):
            p = RES / m / f"summary_{split}_mimic3b.json"
            if not p.exists():
                continue
            for r in json.load(p.open()):
                if r["variant"] in ("uq", "nsai_uq"):
                    say(f"| {m} | {split} | {r['variant']} | {r['tau']:.4g} | "
                        f"{r.get('calib_n')} | {r.get('calib_certifiable')} | "
                        f"{r.get('calib_min_prefix_needed')} | {r['coverage']:.3f} |")
    say("")


# --------------------------------------------------- 2. token alignment -------
def binary_entropy_from_margin(m):
    p = 1 / (1 + math.exp(-m))
    return -sum(q * math.log(q) for q in (p, 1 - p) if q > 0)


def audit_margin_agreement():
    say("## 2. Token alignment, population part\n")
    say("greedy decoding emits the argmax token, so at the step the margin is "
        "read from, sign(logit(SAFE ids) − logit(UNSAFE ids)) must equal the "
        "answer parsed from the text. A margin read off the wrong token ids "
        "(e.g. `SAFE` vs `▁SAFE`) would disagree.\n")
    say("| model | file tag | rows | margin present | sign agrees with parsed text "
        "| disagree | H recomputed = stored (|Δ|<1e-6) |")
    say("|---|---|---|---|---|---|---|")
    total_dis = 0
    for m in MODELS:
        for tag in ("_mimic3b", "_mimic3bswap"):
            n = pres = agree = dis = hn = hok = 0
            for pf in sorted((RES / m).glob(f"preds_*_seed0{tag}.jsonl")):
                for r in read(pf):
                    n += 1
                    mg = r.get("logit_margin")
                    a = parse_answer(r.get("raw") or "")
                    if mg is None or a is None or mg == 0:
                        continue
                    pres += 1
                    if (mg > 0) == (a == "SAFE"):
                        agree += 1
                    else:
                        dis += 1
                    if r.get("uq_signal") == "decision_entropy" and r.get("uq_uncertainty") is not None:
                        hn += 1
                        hok += abs(binary_entropy_from_margin(mg) - r["uq_uncertainty"]) < 1e-6
            total_dis += dis
            say(f"| {m} | {tag} | {n:,} | {pres:,} | {agree:,} | {dis:,} | {hok:,}/{hn:,} |")
    say(f"\nTotal disagreements: **{total_dis}**.\n")

    say("**Rows whose text names an answer but that carry NO margin** (the "
        "margin step search found no generated id in the SAFE/UNSAFE sets). "
        "`src/audit_token_alignment.py` (live `model.generate`, log in "
        "`logs/mimic_v3b/audit_token_alignment.log`) shows why for "
        "OpenBioLLM: it writes UNSAFE as the non-canonical pieces `UNS`+`AFE`, "
        "while the pipeline's UNSAFE ids are `UN` and `ĠUNS` only (the "
        "tokenizer itself encodes the word as `UN`+`SAFE`). Such a row gets "
        "uncertainty ∞: deferred by UQ and dropped from calibration.\n")
    say("| model | file | model answers UNSAFE | of which no margin | UQ-governed rows deferred only for this |")
    say("|---|---|---|---|---|")
    any_row = False
    for m in MODELS:
        for pf in sorted((RES / m).glob("preds_*_seed0_mimic3b*.jsonl")):
            u = nm = d = 0
            for r in read(pf):
                a = parse_answer(r.get("raw") or "")
                if a is not None and r.get("logit_margin") is None:
                    nm += 1
                    d += r["abstained"] and not r["gate_fired"]
                u += a == "UNSAFE"
            if nm:
                any_row = True
                say(f"| {m} | {pf.name} | {u:,} | {nm:,} | {d:,} |")
    if not any_row:
        say("| — | none | | | |")
    say("\nNo main-arm (`_mimic3b`) file of any model is affected: OpenBioLLM "
        "never answers UNSAFE there. Accuracy, CC and every prediction are "
        "parsed from the text, so the swap result is unaffected; only the "
        "margin (hence UQ) is.\n")

    say("**Non-answers, by model and file (main arm)**: the share of rows with "
        "no parsable SAFE/UNSAFE. A non-answer counts as wrong in accuracy and "
        "is always deferred by UQ.\n")
    say("| model | variant | test | held-out |")
    say("|---|---|---|---|")
    for m in MODELS:
        for v in ("base", "rag", "nsai", "nsai_uq"):
            cells = []
            for split in ("test", "heldout"):
                pf = RES / m / f"preds_{split}_{v}_seed0_mimic3b.jsonl"
                if not pf.exists():
                    cells.append("—")
                    continue
                P = read(pf)
                cells.append(f"{sum(parse_answer(r.get('raw') or '') is None for r in P) / len(P):.3f}")
            say(f"| {m} | {v} | {cells[0]} | {cells[1]} |")
    say("")


# ------------------------------------------------------- 3. disjointness ------
def audit_disjoint():
    say("## 3. Patient-level disjointness (subject_id)\n")
    say("| dataset | split A | split B | patients A | patients B | shared |")
    say("|---|---|---|---|---|---|")
    bad = []
    sets = {}
    for ds in ("mimic_v3b", "mimic_v3b_note", "mimic_v3b_cl"):
        for split in ("train", "calib", "test", "heldout"):
            p = HERE / "data" / ds / f"counterfactual_{split}.jsonl"
            if p.exists():
                sets[ds, split] = {r["subject_id"] for r in read(p)}
    pairs = [(("mimic_v3b", a), ("mimic_v3b", b)) for a, b in
             [("train", "calib"), ("train", "test"), ("calib", "test"),
              ("train", "heldout"), ("calib", "heldout"), ("test", "heldout")]]
    pairs += [(("mimic_v3b_note", a), ("mimic_v3b_note", b)) for a, b in
              [("train", "test"), ("calib", "test"), ("train", "calib")]]
    pairs += [(("mimic_v3b_cl", "train"), ("mimic_v3b", "test")),
              (("mimic_v3b_cl", "train"), ("mimic_v3b", "calib")),
              (("mimic_v3b_cl", "train"), ("mimic_v3b", "heldout"))]
    for A, B in pairs:
        if A not in sets or B not in sets:
            continue
        sh = len(sets[A] & sets[B])
        say(f"| {A[0] if A[0] == B[0] else A[0] + ' vs ' + B[0]} | {A[1]} | {B[1]} "
            f"| {len(sets[A]):,} | {len(sets[B]):,} | **{sh}** |")
        if sh and "heldout" not in (A[1], B[1]):
            bad.append((A, B, sh))
    # prompt text (not patients) shared between calibration and test
    cp = {r["prompt"] for r in read(D / "counterfactual_calib.jsonl")}
    tp = {r["prompt"] for r in read(D / "counterfactual_test.jsonl")}
    say(f"\nAssertions (train∩test, calib∩test, train∩calib = 0): "
        f"**{'PASS' if not bad else 'FAIL ' + str(bad)}**.")
    say(f"Not a patient leak but reported: {len(cp & tp):,} of {len(tp):,} "
        f"distinct test prompts also occur, as text, in calibration (different "
        f"patients whose rendered vignette is identical: same age, sex and "
        f"printed value).\n")


# --------------------------------------------------------- 4. manual trace ----
RX = {"creat": re.compile(r"serum creatinine of (\d+\.\d+) mg/dL"),
      "k": re.compile(r"Serum potassium is (\d+\.\d+) mmol/L"),
      "inr": re.compile(r"Today's INR is (\d+\.\d+)"),
      "who": re.compile(r"A (\d+)-year-old (woman|man)")}
THRESH = {"metformin_egfr30": ("creat", 30.0, "<"),
          "metformin_egfr45": ("creat", 45.0, "<"),
          "spironolactone_k5_5": ("k", 5.5, ">"),
          "warfarin_inr4": ("inr", 4.0, ">")}


def my_ckd_epi_2021(scr, age, female):
    """Written here from the published equation, not imported."""
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    return (142 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.200
            * 0.9938 ** age * (1.012 if female else 1.0))


def independent(item):
    kind, thr, op = THRESH[item["family"]]
    tail = item["vignette"].strip().split("\n")[-1]
    vs = RX[kind].search(tail).group(1)
    v = float(vs)
    w = RX["who"].search(tail)
    age, female = int(w.group(1)), w.group(2) == "woman"
    q = my_ckd_epi_2021(v, age, female) if kind == "creat" else v
    lab = "UNSAFE" if (q < thr if op == "<" else q > thr) else "SAFE"
    raw = item["facts"]["lab_raw"]
    nd = len(vs.split(".")[1])              # decimals AS PRINTED
    return {"value_in_prompt": v, "facts_value": raw,
            "value_matches": abs(raw - v) <= 0.5 * 10 ** -nd + 1e-9,
            "quantity": q,
            "stored_quantity": item.get("factor_value"),
            "my_label": lab, "label_ok": lab == item["label"],
            "age_ok": age == item["facts"]["age"],
            "sex_ok": female == str(item["facts"]["sex"]).lower().startswith("f")}


def check_row(p, item, tau, gate, use_gate=True):
    ind = independent(item)
    mg = p.get("logit_margin")
    H = binary_entropy_from_margin(mg) if mg is not None else math.inf
    g = gate(item["vignette"])
    if not use_gate:
        g = type(g)(False, False, None, g.extracted)
    own = parse_answer(p.get("raw") or "")
    exp_pred = ("UNSAFE" if g.violated else "SAFE") if g.fired else own
    exp_abst = (not g.fired) and (exp_pred is None or H > tau)
    c = {**ind, "P_SAFE": None if mg is None else 1 / (1 + math.exp(-mg)),
         "H": H, "H_ok": (p.get("uq_uncertainty") is None and mg is None) or
         (mg is not None and abs(H - p["uq_uncertainty"]) < 1e-6),
         "gate_fired": p["gate_fired"], "gate_ok": g.fired == p["gate_fired"],
         "gate_equals_rule": (not g.fired) or exp_pred == ind["my_label"],
         "pred_ok": p["pred"] == exp_pred, "abstain_ok": p["abstained"] == exp_abst,
         "abstained": p["abstained"], "tau": tau, "own": own}
    c["all_ok"] = all(c[k] for k in ("value_matches", "label_ok", "age_ok", "sex_ok",
                                     "H_ok", "gate_ok", "gate_equals_rule",
                                     "pred_ok", "abstain_ok"))
    return c


def audit_trace():
    say("## 4. Manual trace, with independent code\n")
    items = {r["id"]: r for s in ("test", "heldout")
             for r in read(D / f"counterfactual_{s}.jsonl")}
    gate = SymbolicGate(RULE_FAMILIES)
    m, split, v = "biomistral-7b", "test", "nsai_uq"
    tau = next(r["tau"] for r in json.load(open(RES / m / f"summary_{split}_mimic3b.json"))
               if r["variant"] == v)
    preds = read(RES / m / f"preds_{split}_{v}_seed0_mimic3b.jsonl")
    rng = random.Random(0)
    strata = [("gate fired", lambda p: p["gate_fired"], 3),
              ("answered by the model", lambda p: not p["gate_fired"] and not p["abstained"], 3),
              ("deferred by UQ", lambda p: p["abstained"], 4)]
    pick = []
    for name, f, k in strata:
        pool = [p for p in preds if f(p)]
        pick += [(name, p) for p in rng.sample(pool, k)]
    say(f"BioMistral-7B, test, NS-AI + UQ, τ = {tau:.6f}. Ten rows, stratified. "
        "Lab value, age and sex are re-read from the prompt with this file's own "
        "regexes; eGFR is recomputed with this file's own CKD-EPI 2021; the "
        "threshold comes from the family name. Row-level values stay in "
        f"`{TRACE.relative_to(HERE)}` (gitignored); this table is flags and "
        "model outputs only.\n")
    say("| # | stratum | family | value in prompt = record | label = independent rule "
        "| P(SAFE) | H | H = stored | gate as expected | gate = rule | abstain = (H > τ) rule | all |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    tl = ["# Local trace (credentialed rows; do not commit, do not paste into an online service)\n"]
    yn = {True: "yes", False: "**NO**"}
    for i, (name, p) in enumerate(pick, 1):
        it = items[p["id"]]
        c = check_row(p, it, tau, gate)
        say(f"| {i} | {name} | {it['family']} | {yn[c['value_matches']]} "
            f"| {yn[c['label_ok']]} | {c['P_SAFE']:.4f} | {c['H']:.4f} | {yn[c['H_ok']]} "
            f"| {yn[c['gate_ok'] and c['pred_ok']]} | {yn[c['gate_equals_rule']]} "
            f"| {yn[c['abstain_ok']]} | {'PASS' if c['all_ok'] else '**FAIL**'} |")
        tl += [f"## {i}. {name} — {p['id']}", "", "```", it["prompt"], "```", "",
               f"- value in prompt {c['value_in_prompt']} / record {c['facts_value']}; "
               f"quantity (my calc) {c['quantity']:.3f} / stored {c['stored_quantity']}",
               f"- label stored {it['label']} / independent {c['my_label']}",
               f"- raw `{p['raw'][:200]!r}` -> own answer {c['own']}",
               f"- margin {p['logit_margin']} -> P(SAFE) {c['P_SAFE']} -> H {c['H']:.6f} "
               f"(stored {p['uq_uncertainty']}); tau {tau:.6f}",
               f"- gate fired {p['gate_fired']}; final pred {p['pred']}; "
               f"abstained {p['abstained']}", ""]
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    TRACE.write_text("\n".join(tl) + "\n")

    say("\nThe same checks over EVERY row of every UQ-bearing prediction file "
        "(main arm):\n")
    say("| model | split | variant | rows | failures | which checks failed |")
    say("|---|---|---|---|---|---|")
    for m in MODELS:
        for split in ("test", "heldout"):
            summ = RES / m / f"summary_{split}_mimic3b.json"
            if not summ.exists():
                continue
            taus = {r["variant"]: r["tau"] for r in json.load(summ.open())}
            for v in ("uq", "nsai_uq"):
                pf = RES / m / f"preds_{split}_{v}_seed0_mimic3b.jsonl"
                if not pf.exists():
                    continue
                fails, n = collections.Counter(), 0
                for p in read(pf):
                    n += 1
                    c = check_row(p, items[p["id"]], taus[v], gate,
                                  use_gate=(v == "nsai_uq"))
                    if not c["all_ok"]:
                        for k in ("value_matches", "label_ok", "age_ok", "sex_ok",
                                  "H_ok", "gate_ok", "gate_equals_rule",
                                  "pred_ok", "abstain_ok"):
                            if not c[k]:
                                fails[k] += 1
                say(f"| {m} | {split} | {v} | {n:,} | {sum(fails.values()):,} "
                    f"| {dict(fails) or '—'} |")
    say("")


# ------------------------------------------------------------ 5. swap ----------
ORIG = "Answer with exactly one word on the first line: SAFE or UNSAFE."
SWAP = "Answer with exactly one word on the first line: UNSAFE or SAFE."
WORDS = re.compile(r"\b(UNSAFE|SAFE)\b", re.I)


def audit_swap():
    say("## 5. Option-swap validation\n")
    o = {r["id"]: r for r in read(D / "counterfactual_test.jsonl")}
    s = {r["id"]: r for r in read(HERE / "data/mimic_v3b_swap/counterfactual_test.jsonl")}
    same_ids = o.keys() == s.keys()
    same_labels = all(o[i]["label"] == s[i]["label"] for i in o)
    only_line = all(o[i]["prompt"].replace(ORIG, SWAP) == s[i]["prompt"]
                    and o[i]["prompt"] != s[i]["prompt"] for i in o)
    say(f"- same item ids in both files: **{same_ids}** ({len(o):,})")
    say(f"- ground-truth label identical per id: **{same_labels}**. It is the "
        "word SAFE/UNSAFE, not an option index, so there is no position to "
        "update; `metrics.score` compares `pred == label` as strings.")
    say(f"- prompts differ ONLY in the options line: **{only_line}**\n")
    say("| model | order | acc (all) | echoes both words | rows naming exactly one answer "
        "| acc on those | UNSAFE share on those | margin sign = text |")
    say("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for key, tag in (("original", "_mimic3b"), ("swapped", "_mimic3bswap")):
            pf = RES / m / f"preds_test_base_seed0{tag}.jsonl"
            if not pf.exists():
                continue
            P = read(pf)
            both = one = one_ok = one_uns = sign_ok = sign_n = 0
            for p in P:
                w = {x.upper() for x in WORDS.findall(p.get("raw") or "")}
                if len(w) == 2:
                    both += 1
                elif len(w) == 1:
                    one += 1
                    a = w.pop()
                    one_ok += a == p["label"]
                    one_uns += a == "UNSAFE"
                if p.get("logit_margin") not in (None, 0) and p["pred"]:
                    sign_n += 1
                    sign_ok += (p["logit_margin"] > 0) == (p["pred"] == "SAFE")
            acc = sum((not p["abstained"]) and p["pred"] == p["label"] for p in P) / len(P)
            say(f"| {m} | {key} | {acc:.3f} | {both / len(P):.3f} | {one / len(P):.3f} "
                f"| {one_ok / max(one, 1):.3f} | {one_uns / max(one, 1):.3f} "
                f"| {sign_ok}/{sign_n} |")
    say("")


def main():
    say("# Pipeline sanity audits (v3b)\n")
    say("Generated by `src/audit_pipeline_sanity.py`. Counts and model outputs "
        "only; see the docstring for what each audit can and cannot show.\n")
    audit_oracle()
    audit_margin_agreement()
    audit_disjoint()
    audit_trace()
    audit_swap()
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
