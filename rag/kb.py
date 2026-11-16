"""
kb.py
Project: Medical Information Simplifier
Purpose: Load the local knowledge base (MedlinePlus topics + curated glossary) into
         memory and index every title/alias for fast lookup by the term extractor.

Two lookup tables are built:
  * case-insensitive:  normalised name -> entries   (most medical words)
  * case-sensitive:    exact name      -> entries   (abbreviations such as OD, TSH, CRP)
"""

import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_JSONL = "data/kb/medlineplus.jsonl"
DEFAULT_CSV = "data/kb/curated_glossary.csv"


@dataclass
class Entry:
    id: str
    title: str
    aliases: list
    summary: str
    url: str
    source: str
    category: str = "health_topic"
    ambiguous: bool = False
    origin: str = "medlineplus"     # which part of the KB: "medlineplus" | "curated"


@dataclass
class KB:
    entries: dict = field(default_factory=dict)          # id -> Entry
    ci_index: dict = field(default_factory=dict)         # normalised name -> [(entry_id, is_title)]
    cs_index: dict = field(default_factory=dict)         # exact name -> [(entry_id, is_title)]
    compact_index: dict = field(default_factory=dict)    # normalised name without spaces -> ...
    max_words: int = 1                                   # longest name, in tokens
    missing: list = field(default_factory=list)          # files that could not be loaded
    fuzzy_buckets: dict = None                           # built lazily by rag.terms

    def __len__(self):
        return len(self.entries)


# --------------------------------------------------------------------------- #
# Normalisation (applied identically to KB names and document text)
# --------------------------------------------------------------------------- #
_SEPARATORS = re.compile(r"[-/&.+,()]")


def _fold_uk(word):
    """Fold UK 'ae'/'oe' spellings to US 'e' in one lower-case word.

    Only where UK spellings actually use them, so genuine US clusters survive:
      ae -> e  everywhere (haemoglobin, anaemia, paediatric) except a leading "aer-" (aerosol)
      oe -> e  at the start (oesophagus, oedema) or after h/c/f/n/m (diarrhoea, coeliac,
               foetus, dyspnoea, amoeba), but not in gastroesophageal, bioethics, roentgen
      neither  for a word-final ae/oe/oes (pneumoniae, toe, shoes, tornadoes)
    """
    if not word.startswith("aer"):
        word = re.sub(r"ae(?!s?$)", "e", word)
    return re.sub(r"(?:^|(?<=[hcfnm]))oe(?!s?$)", "e", word)


def normalise(text):
    """Lower-case, fold UK spellings to US, and turn punctuation into spaces.

    The same function is used on both sides, so a rule can never create a mismatch;
    at worst two different words collapse to the same form.
    """
    # Strip accents (Sjögren -> sjogren), then case-fold and drop apostrophes (Crohn's -> crohns)
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.casefold().replace("'", "").replace("’", "")
    t = re.sub(r"[a-z]+", lambda m: _fold_uk(m.group()), t)
    t = re.sub(r"isation\b", "ization", t)
    t = re.sub(r"ised\b", "ized", t)
    t = _SEPARATORS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def raw_tokens(text):
    """Split a name into word tokens the same way documents are tokenised."""
    return re.findall(r"[A-Za-z0-9]+", text)


def is_case_sensitive(name, category):
    """Abbreviation-like names must match exactly, so 'on' never matches ON."""
    if category == "abbreviation":
        return True
    compact = re.sub(r"[^A-Za-z0-9]", "", name)
    if len(compact) <= 2:          # Hb, PR (but "Flu" stays case-insensitive)
        return True
    return len(compact) <= 6 and sum(c.isupper() for c in compact) >= 2


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(Entry(r["id"], r["title"], r["aliases"], r["summary"], r["url"], r["source"]))
    return out


def _read_csv(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            out.append(Entry(
                id=f"cur-{i:03d}",
                title=r["term"],
                aliases=[a.strip() for a in r["aliases"].split("|") if a.strip()],
                summary=r["definition"],
                url=r["url"],
                source=r["source"],
                category=r["category"],
                ambiguous=r["ambiguous"].strip().lower() == "true",
                origin="curated",
            ))
    return out


def _add(index, key, entry_id, is_title):
    bucket = index.setdefault(key, [])
    if (entry_id, is_title) not in bucket:
        bucket.append((entry_id, is_title))


def build_index(entries, stoplist=()):
    """Index entries by every title and alias, skipping stoplisted names."""
    kb = KB()
    stop = {normalise(s) for s in stoplist}
    for e in entries:
        kb.entries[e.id] = e
        for name, is_title in [(e.title, True)] + [(a, False) for a in e.aliases]:
            if not raw_tokens(name) or normalise(name) in stop:
                continue
            if is_case_sensitive(name, e.category):
                key = " ".join(raw_tokens(name))
                _add(kb.cs_index, key, e.id, is_title)
            else:
                key = normalise(name)
                _add(kb.ci_index, key, e.id, is_title)
                # Lets "Gastro-oesophageal reflux" match "Gastroesophageal reflux"
                _add(kb.compact_index, key.replace(" ", ""), e.id, is_title)
            kb.max_words = max(kb.max_words, len(raw_tokens(name)))
    return kb


def load_stoplist(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def load_kb(jsonl=DEFAULT_JSONL, csv_path=DEFAULT_CSV, stoplist_path=None):
    """Load whatever parts of the KB exist. Missing files are listed in kb.missing."""
    if stoplist_path is None:
        stoplist_path = Path(__file__).with_name("stoplist.txt")
    entries, missing = [], []
    for path, reader in ((jsonl, _read_jsonl), (csv_path, _read_csv)):
        if Path(path).exists():
            entries += reader(path)
        else:
            missing.append(str(path))
    kb = build_index(entries, load_stoplist(stoplist_path))
    kb.missing = missing
    if missing:
        print(f"Warning: knowledge-base file(s) not found: {', '.join(missing)}", file=sys.stderr)
    return kb
