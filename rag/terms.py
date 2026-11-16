"""
terms.py
Project: Medical Information Simplifier
Purpose: Find medical terms in REDACTED text by matching it against the local knowledge
         base. Runs entirely on the local machine; no text is sent anywhere.

Matching, in order of preference:
  1. exact / alias   longest phrase first, so "Iron deficiency anaemia" beats "anaemia".
                     Abbreviation-like names (OD, TSH, CRP) are matched case-sensitively;
                     everything else after rag.kb.normalise() (case, UK/US spelling, accents).
  2. fuzzy           only for single words nothing else matched, to recover OCR damage:
                     a cheap digit->letter repair ("Haemog1obin"), then words OCR glued
                     together ("HepatitisB" -> Hepatitis B), then rapidfuzz.

Redaction placeholders such as <PERSON> are never matched and no phrase spans across one.
"""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from rag.kb import normalise

DEFAULT_FUZZY_THRESHOLD = 88   # rapidfuzz ratio (0-100) needed for a fuzzy match
DEFAULT_FUZZY_MIN_LEN = 5      # shorter words are never fuzzy-matched (too many false hits)
FUZZY_LEN_TOLERANCE = 1        # OCR usually swaps, drops or adds ONE character; a wider window
                               # matched different words (Prednisolone -> Prednisone)

PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]*>")
TOKEN = re.compile(r"[A-Za-z0-9]+")
# Characters allowed between the words of one phrase. Anything else (newline, colon,
# comma, brackets, a placeholder) ends the phrase.
SOFT_SEPARATOR = re.compile(r"[ \t\-'’/&.+]*")
# Common OCR digit-for-letter swaps inside words
OCR_REPAIR = str.maketrans({"1": "l", "0": "o", "5": "s"})


@dataclass
class Term:
    term: str             # the text exactly as it appears in the document
    span: tuple           # (start, end) character offsets into the input text
    match_type: str       # "exact" | "alias" | "fuzzy"
    entry_ids: list       # candidate knowledge-base entries, best first
    score: float          # 100 for exact/alias; rapidfuzz ratio for fuzzy


# --------------------------------------------------------------------------- #
# Tokenising
# --------------------------------------------------------------------------- #
def segments(text):
    """Split text into runs of tokens that a single phrase may span.

    Each token is (start, end, text). Placeholders are blanked out first so their
    letters are never tokens, and they act as a hard boundary.
    """
    masked = PLACEHOLDER.sub(lambda m: "\n" * len(m.group()), text)
    runs, current, prev_end = [], [], None
    for m in TOKEN.finditer(masked):
        if current and not SOFT_SEPARATOR.fullmatch(masked[prev_end:m.start()]):
            runs.append(current)
            current = []
        current.append((m.start(), m.end(), m.group()))
        prev_end = m.end()
    if current:
        runs.append(current)
    return runs


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _rank(kb, hits):
    """Order candidate entries: titles before aliases, then MedlinePlus before curated."""
    best = {}
    for entry_id, is_title in hits:
        best[entry_id] = best.get(entry_id, False) or is_title
    order = sorted(best, key=lambda e: (not best[e], kb.entries[e].origin != "medlineplus", e))
    return order, any(best.values())


def _lookup(kb, text, tokens):
    """Exact/alias lookup for one candidate phrase. Returns ranked hits or None."""
    phrase = text[tokens[0][0]:tokens[-1][1]]
    hits = list(kb.cs_index.get(" ".join(t[2] for t in tokens), []))
    key = normalise(phrase)
    hits += kb.ci_index.get(key, [])
    if not hits and "-" in phrase:      # hyphenated compound written as one word in the KB
        hits += kb.compact_index.get(key.replace(" ", ""), [])
    return hits or None


def _fuzzy_buckets(kb):
    """Single-word, case-insensitive names grouped by length (built once per KB)."""
    if kb.fuzzy_buckets is None:
        buckets = {}
        for name in kb.ci_index:
            if " " not in name and name.isalpha():
                buckets.setdefault(len(name), []).append(name)
        kb.fuzzy_buckets = buckets
    return kb.fuzzy_buckets


def _fuzzy(kb, word, threshold, min_len):
    """Match one OCR-damaged word. Returns (index hits, score) or None."""
    if len(word) < min_len or not any(c.isalpha() for c in word):
        return None
    norm = normalise(word)
    repaired = normalise(word.translate(OCR_REPAIR))
    if repaired != norm and repaired in kb.ci_index:
        return kb.ci_index[repaired], fuzz.ratio(norm, repaired)
    # OCR often drops the space between words ("HepatitisB", "LymphNodes"):
    # compare against names with their spaces removed before trying fuzzy matching.
    if repaired in kb.compact_index:
        return kb.compact_index[repaired], 100.0
    if not repaired.isalpha():
        return None
    buckets = _fuzzy_buckets(kb)
    candidates = [n for size in range(len(repaired) - FUZZY_LEN_TOLERANCE, len(repaired) + FUZZY_LEN_TOLERANCE + 1)
                  for n in buckets.get(size, [])]
    best = process.extractOne(repaired, candidates, scorer=fuzz.ratio, score_cutoff=threshold)
    return (kb.ci_index[best[0]], best[1]) if best else None


def extract_terms(text, kb, fuzzy=True, fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD,
                  fuzzy_min_len=DEFAULT_FUZZY_MIN_LEN):
    """Return every knowledge-base term found in `text`, in document order."""
    found = []
    for run in segments(text):
        i = 0
        while i < len(run):
            for n in range(min(kb.max_words, len(run) - i), 0, -1):
                tokens = run[i:i + n]
                hits = _lookup(kb, text, tokens)
                if hits:
                    ids, is_title = _rank(kb, hits)
                    start, end = tokens[0][0], tokens[-1][1]
                    found.append(Term(text[start:end], (start, end), "exact" if is_title else "alias", ids, 100.0))
                    i += n
                    break
            else:
                start, end, word = run[i]
                match = _fuzzy(kb, word, fuzzy_threshold, fuzzy_min_len) if fuzzy else None
                if match:
                    ids, _ = _rank(kb, match[0])
                    found.append(Term(word, (start, end), "fuzzy", ids, float(match[1])))
                i += 1
    return found
