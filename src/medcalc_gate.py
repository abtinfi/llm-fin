"""
Symbolic gate for the MedCalc-Bench benchmark.

DIFFERENCE FROM `components.SymbolicGate`
-----------------------------------------
The original gate reads a templated vignette that states the decisive quantity
outright ("an eGFR of 27 mL/min/1.73m2"), so its only job is to find a number
and compare it. `components.py` says so in its own header, and calls its
measured contribution an upper bound for exactly that reason.

Here the notes are real PMC case reports. They state **creatinine**, not eGFR;
**QT**, not QTc. So this gate has to do three things instead of one:

    1. extract several facts from unconstrained clinical prose
       (creatinine + its unit + age + sex, or QT + heart rate),
    2. run the clinical formula itself,
    3. compare the result against a published threshold.

Step 2 is what makes this a symbolic reasoning component rather than a regex
lookup, and step 1 is where it is expected to lose ground. Both are measured
rather than assumed: `python src/medcalc_gate.py --data ...` reports extraction
accuracy against the dataset's structured `facts`, which are ground truth.

THE GATE NEVER GUESSES (HANDOFF.md section 7). If any required fact is missing,
ambiguous, or physiologically impossible, it declines and the neural answer
stands. No default units, no "probably mg/dL", no picking the first number near
the word. Its zero-error record depends on this and must not be traded away for
a higher firing rate.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from components import GateResult                       # noqa: E402
from renal import (ckd_epi, mdrd, to_mgdl, qtc_bazett,   # noqa: E402
                   PLAUSIBLE, implausible)

NUM = r"(\d+(?:\.\d+)?)"

# --- creatinine -------------------------------------------------------------
# Unit is REQUIRED. MedCalc notes state creatinine in mg/dL and in µmol/L, and
# the two differ by 88x -- assuming a default unit would not be a small error,
# it would invert the decision. "creatine" is included because that misspelling
# occurs in the source notes.
CREAT_UNIT = r"(mg\s*/\s*d[lL]|mg\s*/\s*L|[µu]mol\s*/\s*L|mmol\s*/\s*L)"
CREAT_PATTERNS = [
    # value AFTER the word: "serum creatinine was 3.1 mg/dL",
    # "creatinine: 1.3 mg/dl", "serum creatinine (0.57 mg/dL)"
    re.compile(r"creatin(?:ine|e)\b[^.\n;]{0,30}?" + NUM + r"\s*" + CREAT_UNIT,
               re.I),
    # value BEFORE the word: "0.5 mg/dL creatinine". Real notes do this and
    # the original post-position-only patterns silently read the next lab
    # along instead -- e.g. picking up "71 mg/dL HDL".
    re.compile(NUM + r"\s*" + CREAT_UNIT + r"[^.\n;]{0,12}?creatin(?:ine|e)\b",
               re.I),
    # "Cr 2.4 mg/dL" -- only with an explicit unit, never bare
    re.compile(r"\bCr\b[^.\n;]{0,12}?" + NUM + r"\s*" + CREAT_UNIT),
]


def creatinine_mentions(note):
    """
    Every place the note states a creatinine WITH a unit, de-duplicated by
    character span.

    This is the single definition of "where the creatinine is", used both by
    the dataset builder (to reject notes where it is stated more than once) and
    by the gate (to read it). Keeping one definition means the gate can never
    be reading a different quantity from the one the label was derived from.

    Why more than one mention is disqualifying: a note reading "creatinine 0.7
    on admission ... creatinine 3.3 on day 5" has no single eGFR. MedCalc-Bench
    resolves that by picking one, but nothing in the text tells a reader which,
    so the item would have no unambiguous ground truth. Same failure class as
    bug #2 in HANDOFF.md -- a case whose text contradicts its own label.
    """
    seen, out = set(), []
    for rx in CREAT_PATTERNS:
        for m in rx.finditer(note):
            key = (m.start(), m.end())
            if key in seen:
                continue
            seen.add(key)
            g = m.groups()
            val, unit = (g[0], g[1])
            try:
                out.append({"value": float(val), "unit": unit,
                            "span": key, "num_span": m.span(1),
                            "mgdl": to_mgdl(float(val), unit)})
            except Exception:
                continue
    # Drop spans nested inside a longer match of another pattern.
    out.sort(key=lambda d: d["span"])
    keep = []
    for d in out:
        if any(k["span"][0] <= d["span"][0] and d["span"][1] <= k["span"][1]
               and k is not d for k in out):
            continue
        keep.append(d)
    return keep

AGE_PATTERNS = [
    re.compile(NUM + r"[\s-]*(?:year|yr)s?[\s-]*old", re.I),
    re.compile(r"aged?\s+" + NUM + r"\s*(?:years|yrs)?", re.I),
]
MALE = re.compile(r"\b(man|male|gentleman|boy|his)\b", re.I)
FEMALE = re.compile(r"\b(woman|female|lady|girl|her)\b", re.I)

QT_PATTERNS = [
    re.compile(r"QT\s*(?:interval|duration)?[^.\n;]{0,25}?" + NUM
               + r"\s*(?:ms|msec|milliseconds)\b", re.I),
    re.compile(r"QT\s*(?:interval|duration)?\s*(?:of|was|is|:|=)\s*" + NUM,
               re.I),
]
HR_PATTERNS = [
    re.compile(r"(?:heart rate|pulse rate|pulse)[^.\n;]{0,25}?" + NUM
               + r"\s*(?:beats?\s*(?:per|/)\s*min|bpm|/min)?", re.I),
    re.compile(NUM + r"\s*(?:beats?\s*(?:per|/)\s*min|bpm)\b", re.I),
]


def _first(patterns, text, groups=1):
    for rx in patterns:
        m = rx.search(text)
        if m:
            return m.groups() if groups > 1 else m.group(1)
    return None


def extract_renal(note):
    """
    creatinine (mg/dL), age, sex -- absent keys mean "not established".

    Declines on ambiguity: if the note states creatinine more than once with
    differing values, there is no single answer and the gate must not pick one.
    """
    out = {}
    men = creatinine_mentions(note)
    vals = {round(m["mgdl"], 6) for m in men}
    if len(men) >= 1 and len(vals) == 1:
        m = men[0]
        out["creatinine_mgdl"] = m["mgdl"]
        out["creatinine_raw"] = m["value"]
        out["creatinine_unit"] = m["unit"]
    age = _first(AGE_PATTERNS, note)
    if age:
        out["age"] = float(age)
    # Sex: decide only when the evidence is one-sided. A note containing both
    # "his" and "her" (a patient and a relative) is not a basis for a decision.
    head = note[:400]
    m, f = bool(MALE.search(head)), bool(FEMALE.search(head))
    if m and not f:
        out["sex"] = "Male"
    elif f and not m:
        out["sex"] = "Female"
    return out


def extract_qt(note):
    out = {}
    qt = _first(QT_PATTERNS, note)
    if qt:
        out["qt_ms"] = float(qt)
    hr = _first(HR_PATTERNS, note)
    if hr:
        out["heart_rate"] = float(hr)
    return out


class MedCalcGate:
    """
    Extract -> compute -> compare. Declines unless every input is present and
    physiologically possible.

    `calculator` selects which eGFR equation to run. The dataset records which
    one MedCalc used for each note; passing "ckd_epi" (the current clinical
    standard) for everything is the honest default when that is unknown at
    inference time, and `--oracle_calc` exists only to quantify how much that
    choice costs.
    """

    EGFR_THRESHOLD = 30.0     # FDA metformin label
    QTC_THRESHOLD = 500.0     # FDA ondansetron label

    def __init__(self, calculator="ckd_epi"):
        self.calculator = calculator

    def __call__(self, vignette, calculator=None):
        low = vignette.lower()

        if "metformin" in low:
            f = extract_renal(vignette)
            need = ("creatinine_mgdl", "age", "sex")
            if any(k not in f for k in need):
                return GateResult(False, False, None, f)
            kind = calculator or self.calculator
            fn = ckd_epi if kind == "ckd_epi" else mdrd
            try:
                egfr = (fn(f["creatinine_mgdl"], f["age"], f["sex"])
                        if kind == "ckd_epi"
                        else fn(f["creatinine_mgdl"], f["age"], f["sex"]))
            except Exception:
                return GateResult(False, False, None, f)
            if implausible(creatinine_mgdl=f["creatinine_mgdl"], eGFR=egfr):
                return GateResult(False, False, None, f)
            f["eGFR"] = egfr
            return GateResult(True, egfr < self.EGFR_THRESHOLD,
                              "metformin_renal", f)

        if "ondansetron" in low:
            f = extract_qt(vignette)
            if "qt_ms" not in f or "heart_rate" not in f:
                return GateResult(False, False, None, f)
            try:
                qtc = qtc_bazett(f["qt_ms"], f["heart_rate"])
            except Exception:
                return GateResult(False, False, None, f)
            if implausible(qt_ms=f["qt_ms"], heart_rate=f["heart_rate"],
                           QTc=qtc):
                return GateResult(False, False, None, f)
            f["QTc"] = qtc
            return GateResult(True, qtc > self.QTC_THRESHOLD,
                              "ondansetron_qt", f)

        return GateResult(False, False, None, {})


# --------------------------------------------------------------------------
# Extraction diagnostic. The dataset carries the true values, so how well the
# regexes read real prose is measurable rather than a matter of opinion.
# --------------------------------------------------------------------------

def diagnose(path, oracle_calc=False):
    rows = [json.loads(l) for l in Path(path).open()]
    gate = MedCalcGate()
    n = len(rows)
    fired = correct = 0
    ext = {"creatinine": [0, 0], "age": [0, 0], "sex": [0, 0],
           "qt_ms": [0, 0], "heart_rate": [0, 0]}
    decline_reason = {}

    for r in rows:
        truth = r["facts"]
        calc = r.get("calculator") if oracle_calc else None
        g = gate(r["vignette"], calculator=(calc if calc != "qtc_bazett"
                                            else None))
        if r["family"] == "metformin_renal":
            f = extract_renal(r["vignette"])
            for key, tkey in (("creatinine_raw", "creatinine_raw"),
                              ("age", "age"), ("sex", "sex")):
                slot = {"creatinine_raw": "creatinine"}.get(key, key)
                if key in f:
                    ext[slot][1] += 1
                    tv = truth.get(tkey)
                    ok = (str(f[key]).lower() == str(tv).lower()
                          if key == "sex" else
                          abs(float(f[key]) - float(tv)) < 1e-6)
                    ext[slot][0] += int(ok)
            missing = [k for k in ("creatinine_mgdl", "age", "sex")
                       if k not in f]
        else:
            f = extract_qt(r["vignette"])
            for key in ("qt_ms", "heart_rate"):
                if key in f:
                    ext[key][1] += 1
                    ext[key][0] += int(abs(f[key] - float(truth[key])) < 1e-6)
            missing = [k for k in ("qt_ms", "heart_rate") if k not in f]

        if g.fired:
            fired += 1
            pred = "UNSAFE" if g.violated else "SAFE"
            correct += int(pred == r["label"])
        else:
            key = "+".join(missing) if missing else "implausible/other"
            decline_reason[key] = decline_reason.get(key, 0) + 1

    print(f"\n=== {Path(path).name}  n={n} "
          f"{'(oracle calculator)' if oracle_calc else ''} ===")
    print(f"gate fires on {fired}/{n} = {fired/n:.1%}")
    if fired:
        print(f"when it fires: {correct}/{fired} correct = {correct/fired:.1%}"
              f"   ({fired-correct} wrong)")
    print("extraction accuracy (of the values it did extract):")
    for k, (ok, tot) in ext.items():
        if tot:
            print(f"  {k:14s} {ok}/{tot} = {ok/tot:.1%}")
    if decline_reason:
        print("declines, by what was missing:")
        for k, v in sorted(decline_reason.items(), key=lambda kv: -kv[1]):
            print(f"  {v:5d}  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--oracle_calc", action="store_true",
                    help="use the calculator MedCalc actually used, to "
                         "quantify the cost of defaulting to CKD-EPI")
    args = ap.parse_args()
    for p in args.data:
        diagnose(p, args.oracle_calc)


if __name__ == "__main__":
    main()
