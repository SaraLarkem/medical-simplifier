"""
evaluate_pipeline.py
Project: Medical Information Simplifier
Purpose: Measure the pipeline stages against the synthetic ground truth made by
         generate_synthetic.py.

Two subcommands:

  ocr      OCR pre-processing ablation. For each pre-processing config and image variant:
           character error rate (CER), word error rate (WER), and how many PII / clinically
           meaningful spans survive OCR exactly.

  redact   PII-redaction evaluation. Per detector:
             * leakage        = PII strings still present after redaction      (privacy risk)
             * over-redaction = clinically meaningful strings destroyed        (utility loss)
             * false positives on look-alikes (eponyms, business phone numbers)
             * relaxed precision / recall (ground-truth text only)
           Input text is either the ground truth (isolates the redactor) or real OCR output
           (end-to-end: shows how OCR errors affect redaction).

Examples:
  python evaluate_pipeline.py ocr    --data data/synthetic --workers 4
  python evaluate_pipeline.py redact --data data/synthetic --detectors mvp regex_ids
  python evaluate_pipeline.py redact --data data/synthetic --source ocr --ocr-config deskew_bgnorm --variant phone

Add your own detector: write a function  detect(text) -> [(start, end, entity_type), ...]
in e.g. my_redactor.py and pass  --detectors my_redactor:detect
"""

import argparse
import csv
import importlib
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytesseract

try:
    from rapidfuzz.distance import Levenshtein as _RF
except ImportError:  # pure-Python fallback below
    _RF = None

VARIANTS = ["clean", "scan", "phone"]


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #
def norm_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def _lev_bitparallel(a, b):
    """Hyyro's bit-parallel Levenshtein distance. Works on strings or lists of hashables."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m = len(a)
    full = (1 << m) - 1
    last = 1 << (m - 1)
    peq = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    vp, vn, dist = full, 0, m
    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | vn
        xh = (((eq & vp) + vp) ^ vp) | eq
        hp = vn | (~(xh | vp) & full)
        hn = vp & xh
        if hp & last:
            dist += 1
        elif hn & last:
            dist -= 1
        hp = ((hp << 1) | 1) & full
        hn = (hn << 1) & full
        vp = hn | (~(xv | hp) & full)
        vn = hp & xv
    return dist


def levenshtein(a, b):
    if _RF is not None:
        return _RF.distance(a, b)
    return _lev_bitparallel(a, b)


def cer(ref, hyp):
    ref, hyp = norm_ws(ref), norm_ws(hyp)
    return levenshtein(ref, hyp) / max(1, len(ref))


def wer(ref, hyp):
    r, h = norm_ws(ref).split(), norm_ws(hyp).split()
    return levenshtein(r, h) / max(1, len(r))


# --------------------------------------------------------------------------- #
# OCR pre-processing configs (each maps a grayscale uint8 image -> image for Tesseract)
# --------------------------------------------------------------------------- #
def upscale(g, f=2):
    return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def estimate_skew(g, max_angle=5.0, step=0.25):
    """Projection-profile skew estimate: the rotation that makes text rows sharpest."""
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    bw = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = bw.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-6, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def deskew(g):
    angle = estimate_skew(g)
    h, w = g.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(g, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def adaptive(g):
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)


def bg_normalise(g):
    """Divide by an estimate of the paper brightness: removes shadows / uneven lighting."""
    bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((15, 15), np.uint8)), (0, 0), 21)
    return cv2.divide(g, bg, scale=255)


def otsu(g):
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


PREPROCESS = {
    "baseline": lambda g: upscale(g),                                  # what mvp_test.py does
    "adaptive": lambda g: adaptive(upscale(g)),
    "bgnorm_otsu": lambda g: otsu(bg_normalise(upscale(g))),
    "deskew_adaptive": lambda g: adaptive(upscale(deskew(g))),
    "deskew_bgnorm": lambda g: otsu(bg_normalise(upscale(deskew(g)))),
}


def configure_tesseract():
    cmd = os.getenv("TESSERACT_CMD")
    if not cmd and os.name == "nt":
        default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default):
            cmd = default
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def _ocr_job(job):
    img_path, config, psm, cache_path = job
    configure_tesseract()
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(PREPROCESS[config](g), config=f"--oem 3 --psm {psm}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return cache_path


def cache_path(results, psm, config, doc_id, variant):
    return Path(results) / "ocr_cache" / f"psm{psm}" / config / f"{doc_id}_{variant}.txt"


def ensure_ocr(docs, data_dir, variants, configs, psm, results, workers):
    """Run OCR for any (doc, variant, config) not already cached."""
    jobs = []
    for d in docs:
        for v in variants:
            for c in configs:
                cp = cache_path(results, psm, c, d["id"], v)
                if not cp.exists():
                    jobs.append((Path(data_dir) / d["images"][v], c, psm, cp))
    if not jobs:
        return
    print(f"Running OCR on {len(jobs)} images ({workers} worker(s)); results are cached...", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, _ in enumerate(ex.map(_ocr_job, jobs), 1):
                if i % 20 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            _ocr_job(job)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)


# --------------------------------------------------------------------------- #
# Span bookkeeping
# --------------------------------------------------------------------------- #
def load_docs(data_dir, limit=None):
    files = sorted(Path(data_dir).glob("doc_*.json"))
    if not files:
        sys.exit(f"No doc_*.json found in {data_dir}. Run generate_synthetic.py first.")
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return docs[:limit] if limit else docs


def span_survival(doc, text):
    """How many ground-truth spans appear (exactly) in `text`. String-level, so it works on OCR output."""
    t = norm_ws(text)
    out = defaultdict(lambda: [0, 0])  # (group, type) -> [found, total]
    for group in ("pii", "keep"):
        by_val = defaultdict(list)
        for s in doc[group]:
            by_val[s["text"]].append(s["type"])
        for val, types in by_val.items():
            found = min(len(types), t.count(val))
            out[(group, types[0])][0] += found
            out[(group, types[0])][1] += len(types)
    return out


def apply_spans(text, spans):
    """Replace detected spans with <TYPE> placeholders (overlaps are merged)."""
    merged = []
    for st, en, ty in sorted(spans, key=lambda s: (s[0], -(s[1] - s[0]))):
        if merged and st < merged[-1][1]:
            if en > merged[-1][1]:
                merged[-1] = (merged[-1][0], en, merged[-1][2])
            continue
        merged.append((st, en, ty))
    out, pos = [], 0
    for st, en, ty in merged:
        out += [text[pos:st], f"<{ty}>"]
        pos = en
    out.append(text[pos:])
    return "".join(out)


def score_redaction(doc, input_text, redacted_text):
    """String-level scoring, robust to OCR shifting character offsets.

    Returns {(group, type): {"total", "present", "bad"}} where
      total   = ground-truth spans of that type
      present = of those, how many are still readable in the INPUT text (OCR may have lost some)
      bad     = pii:      present spans whose string is STILL in the redacted text  (leak)
                keep:     present spans whose string is GONE from the redacted text (destroyed)
                not_pii:  present spans whose string is GONE from the redacted text (false positive)
    """
    inp, red = norm_ws(input_text), norm_ws(redacted_text)
    res = defaultdict(lambda: {"total": 0, "present": 0, "bad": 0})
    for group in ("pii", "keep", "not_pii"):
        by_val = defaultdict(list)
        for s in doc[group]:
            by_val[s["text"]].append(s["type"])
        for val, types in by_val.items():
            present = min(len(types), inp.count(val))
            remaining = min(present, red.count(val))
            bad = remaining if group == "pii" else present - remaining
            r = res[(group, types[0])]
            r["total"] += len(types)
            r["present"] += present
            r["bad"] += bad
    return res


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def relaxed_prf_counts(doc, detected):
    """Type-agnostic overlap matching; only valid when `detected` offsets refer to the ground-truth text."""
    true = [(s["start"], s["end"]) for s in doc["pii"]]
    det = [(d[0], d[1]) for d in detected]
    return {
        "det_total": len(det),
        "det_tp": sum(any(overlaps(d, t) for t in true) for d in det),
        "true_total": len(true),
        "true_hit": sum(any(overlaps(t, d) for d in det) for t in true),
    }


# --------------------------------------------------------------------------- #
# Detectors: detect(text) -> [(start, end, entity_type)]
# --------------------------------------------------------------------------- #
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
REGEX_PATTERNS = [
    ("EMAIL_ADDRESS", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    ("PPS_NUMBER", r"\b\d{7}[A-Za-z]{1,2}\b"),
    ("EIRCODE", r"\b(?:[ACDEFHKNPRTVWXY]\d{2}|D6W)\s?[0-9ACDEFHKNPRTVWXY]{4}\b"),
    ("PHONE_NUMBER", r"\+353\s?\d{2}\s?\d{3}\s?\d{4}|\b0\d{2}[\s-]?\d{3}[\s-]?\d{4}\b|\(0\d{1,2}\)\s?\d{7}"),
    ("DATE_TIME", rf"\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}} (?:{_MONTHS}) \d{{4}}\b"),
    ("MRN", r"(?<=MRN: )\d{7}"),
]


def detect_regex_ids(text):
    """Naive baseline: regexes for structured identifiers only (no names, no addresses)."""
    return [(m.start(), m.end(), label) for label, pat in REGEX_PATTERNS for m in re.finditer(pat, text)]


_PRESIDIO = {}


def presidio_detector(entities):
    def detect(text):
        if "engine" not in _PRESIDIO:
            from presidio_analyzer import AnalyzerEngine  # imported lazily: heavy, optional
            _PRESIDIO["engine"] = AnalyzerEngine()
        results = _PRESIDIO["engine"].analyze(text=text, entities=entities, language="en")
        return [(r.start, r.end, r.entity_type) for r in results]
    return detect

def detect_hybrid(text):
    """Combines Presidio NLP (names/addresses) with strict Regex (dates/IDs)."""
    # 1. Run the precise regexes (preserves clinical dates, catches Irish IDs)
    spans = detect_regex_ids(text)
    
    # 2. Run Presidio ONLY for natural language entities 
    if "engine" not in _PRESIDIO:
        from presidio_analyzer import AnalyzerEngine
        _PRESIDIO["engine"] = AnalyzerEngine()
        
    # By omitting DATE_TIME here, we stop Presidio from destroying medical durations
    results = _PRESIDIO["engine"].analyze(
        text=text, 
        entities=["PERSON", "LOCATION"], 
        language="en"
    )
    
    for r in results:
        # Map Presidio's LOCATION to the evaluation script's ADDRESS type
        entity_type = "ADDRESS" if r.entity_type == "LOCATION" else r.entity_type
        spans.append((r.start, r.end, entity_type))
        
    return spans

DETECTORS = {
    "regex_ids": detect_regex_ids,
    # exactly the configuration used in mvp_test.py:
    "mvp": presidio_detector(["PERSON", "DATE_TIME", "PHONE_NUMBER"]),
    "hybrid": detect_hybrid,
}


def resolve_detector(name):
    if name in DETECTORS:
        return DETECTORS[name]
    if ":" in name:  # module:function
        sys.path.insert(0, os.getcwd())
        mod, fn = name.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    sys.exit(f"Unknown detector {name!r}. Built-ins: {', '.join(DETECTORS)}; or use module:function")


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Subcommand: ocr
# --------------------------------------------------------------------------- #
def cmd_ocr(args):
    docs = load_docs(args.data, args.limit)
    ensure_ocr(docs, args.data, args.variants, args.configs, args.psm, args.results, args.workers)
    per_doc, agg = [], defaultdict(lambda: defaultdict(list))
    for c in args.configs:
        for v in args.variants:
            for d in docs:
                text = cache_path(args.results, args.psm, c, d["id"], v).read_text(encoding="utf-8")
                sv = span_survival(d, text)
                pf = sum(x[0] for k, x in sv.items() if k[0] == "pii")
                pt = sum(x[1] for k, x in sv.items() if k[0] == "pii")
                kf = sum(x[0] for k, x in sv.items() if k[0] == "keep")
                kt = sum(x[1] for k, x in sv.items() if k[0] == "keep")
                row = (d["id"], d["doc_type"], v, c, round(cer(d["text"], text), 4),
                       round(wer(d["text"], text), 4), pf, pt, kf, kt)
                per_doc.append(row)
                a = agg[(c, v)]
                a["cer"].append(row[4]); a["wer"].append(row[5])
                a["pf"].append(pf); a["pt"].append(pt); a["kf"].append(kf); a["kt"].append(kt)
    write_csv(Path(args.results) / "ocr_per_doc.csv",
              ["doc_id", "doc_type", "variant", "config", "cer", "wer", "pii_found", "pii_total", "keep_found", "keep_total"],
              per_doc)
    rows = []
    for (c, v), a in agg.items():
        rows.append((c, v, f"{100 * np.mean(a['cer']):.1f}%", f"{100 * np.mean(a['wer']):.1f}%",
                     pct(sum(a["pf"]), sum(a["pt"])), pct(sum(a["kf"]), sum(a["kt"]))))
    table = md_table(["Pre-processing", "Image", "CER", "WER", "PII spans read exactly", "Keep spans read exactly"], rows)
    print(f"\nOCR ablation ({len(docs)} documents, Tesseract --psm {args.psm})\n")
    print(table)
    (Path(args.results) / "ocr_summary.md").write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved: {Path(args.results) / 'ocr_summary.md'} and ocr_per_doc.csv")


# --------------------------------------------------------------------------- #
# Subcommand: redact
# --------------------------------------------------------------------------- #
def cmd_redact(args):
    docs = load_docs(args.data, args.limit)
    texts = {}
    if args.source == "ocr":
        ensure_ocr(docs, args.data, [args.variant], [args.ocr_config], args.psm, args.results, args.workers)
        for d in docs:
            texts[d["id"]] = cache_path(args.results, args.psm, args.ocr_config, d["id"], args.variant).read_text(encoding="utf-8")
        label = f"OCR output ({args.ocr_config}, {args.variant} images)"
    else:
        texts = {d["id"]: d["text"] for d in docs}
        label = "ground-truth text"

    summary, detail_md, detail_rows = [], [], []
    for name in args.detectors:
        detect = resolve_detector(name)
        tot = defaultdict(lambda: {"total": 0, "present": 0, "bad": 0})
        prf = defaultdict(int)
        for d in docs:
            text = texts[d["id"]]
            spans = detect(text)
            scored = score_redaction(d, text, apply_spans(text, spans))
            for k, v in scored.items():
                for f in v:
                    tot[k][f] += v[f]
            if args.source == "gt":
                for k, v in relaxed_prf_counts(d, spans).items():
                    prf[k] += v

        def group_sum(g, field):
            return sum(v[field] for k, v in tot.items() if k[0] == g)

        p = prf["det_tp"] / prf["det_total"] if prf["det_total"] else 0.0
        r = prf["true_hit"] / prf["true_total"] if prf["true_total"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        summary.append([name,
                        pct(group_sum("pii", "bad"), group_sum("pii", "present")),
                        pct(group_sum("keep", "bad"), group_sum("keep", "present")),
                        pct(group_sum("not_pii", "bad"), group_sum("not_pii", "present")),
                        f"{p:.2f} / {r:.2f} / {f1:.2f}" if args.source == "gt" else "n/a (needs gt text)",
                        pct(group_sum("pii", "present"), group_sum("pii", "total"))])
        rows = []
        for (g, t), v in sorted(tot.items()):
            meaning = {"pii": "leaked", "keep": "destroyed", "not_pii": "false positive"}[g]
            rows.append((g.upper(), t, v["total"], v["present"], v["bad"], meaning, pct(v["bad"], v["present"])))
            detail_rows.append((name, args.source, g, t, v["total"], v["present"], v["bad"]))
        detail_md.append(f"### Detail: `{name}`\n\n" + md_table(
            ["Group", "Type", "Spans", "Readable in input", "Bad", "Meaning of 'bad'", "Rate"], rows))

    out = [f"Redaction evaluation on {label} ({len(docs)} documents)\n",
           md_table(["Detector", "PII leakage", "Clinical text destroyed", "Look-alike false positives",
                     "Relaxed precision / recall / F1", "PII readable in input"], summary),
           "\nLower is better for the first three columns. Leakage = privacy risk; destroyed = utility loss.\n"]
    report = "\n".join(out) + "\n" + "\n\n".join(detail_md) + "\n"
    print("\n" + report)
    tag = "gt" if args.source == "gt" else f"ocr_{args.ocr_config}_{args.variant}"
    Path(args.results).mkdir(parents=True, exist_ok=True)
    (Path(args.results) / f"redaction_{tag}.md").write_text(report, encoding="utf-8")
    write_csv(Path(args.results) / f"redaction_{tag}.csv",
              ["detector", "source", "group", "type", "total", "present", "bad"], detail_rows)
    print(f"Saved: {Path(args.results) / f'redaction_{tag}.md'} and .csv")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default="data/synthetic")
        p.add_argument("--results", default="results")
        p.add_argument("--limit", type=int, help="only the first N documents (quick runs)")
        p.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    o = sub.add_parser("ocr", help="OCR pre-processing ablation")
    common(o)
    o.add_argument("--configs", nargs="+", default=list(PREPROCESS), choices=list(PREPROCESS))
    o.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    o.set_defaults(fn=cmd_ocr)

    r = sub.add_parser("redact", help="PII redaction evaluation")
    common(r)
    r.add_argument("--detectors", nargs="+", default=["mvp", "regex_ids"])
    r.add_argument("--source", choices=["gt", "ocr"], default="gt")
    r.add_argument("--ocr-config", default="baseline", choices=list(PREPROCESS))
    r.add_argument("--variant", default="phone", choices=VARIANTS)
    r.set_defaults(fn=cmd_redact)

    args = ap.parse_args()
    configure_tesseract()
    args.fn(args)


if __name__ == "__main__":
    main()