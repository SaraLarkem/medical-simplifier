"""
retriever.py
Project: Medical Information Simplifier
Purpose: Turn the terms found in a document into a short, ranked list of trusted
         definitions ("passages") that can be given to the language model.

The retriever sits behind a small interface (Retriever) so another implementation,
e.g. an embedding-based one, can be swapped in without changing the pipeline.

Version 1 (LexicalRetriever) works from the terms found by rag.terms:
  * one passage per knowledge-base entry (repeats and synonyms are merged)
  * ranked exact -> alias -> fuzzy, then by match score, then by position in the document
  * MedlinePlus summaries are shortened to about two sentences
  * ambiguous abbreviations are marked as ambiguous; no meaning is chosen here
  * a passage without a source URL is never returned
"""

import re
import sys
from dataclasses import dataclass
from typing import Protocol

MAX_SENTENCES = 2
MAX_CHARS = 400
TYPE_ORDER = {"exact": 0, "alias": 1, "fuzzy": 2}
TYPE_WEIGHT = {"exact": 1.0, "alias": 0.9, "fuzzy": 0.8}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass
class Passage:
    id: str            # knowledge-base entry id, e.g. "mlp-34" or "cur-063"
    term: str          # the term as written in the document
    title: str         # the knowledge-base title it matched
    text: str          # the (shortened) definition
    source: str
    url: str
    score: float       # 0-1, used for ranking
    match_type: str    # "exact" | "alias" | "fuzzy"
    origin: str        # "medlineplus" | "curated"
    ambiguous: bool
    verified: bool


class Retriever(Protocol):
    """Anything that can turn a document's terms (and/or its text) into passages."""

    def retrieve(self, terms, kb, max_items=10, text=None):
        ...


# --------------------------------------------------------------------------- #
# Summary shortening
# --------------------------------------------------------------------------- #
def shorten(summary, max_sentences=MAX_SENTENCES, max_chars=MAX_CHARS):
    """Keep the first couple of real sentences of a MedlinePlus summary.

    Skips opening heading questions ("What is asthma?") and stops before a sentence
    that introduces a list ("Symptoms include: Fever Cough ..."), because the list
    structure was flattened when the HTML was stripped.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(summary) if s.strip()]
    while len(sentences) > 1 and sentences[0].endswith("?"):
        sentences.pop(0)
    kept = []
    for s in sentences:
        if ":" in s:
            break
        kept.append(s)
        if len(kept) == max_sentences:
            break
    if kept:
        text = " ".join(kept)
    elif sentences:   # the very first sentence introduces a list: keep the part before it,
        lead = sentences[0].split(":", 1)[0].rstrip()   # unless that is too short to make
        text = lead + "." if len(lead.split()) >= 6 else sentences[0]   # sense on its own
    else:
        text = ""
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return text


# --------------------------------------------------------------------------- #
# Lexical retriever
# --------------------------------------------------------------------------- #
class LexicalRetriever:
    """Looks up the knowledge-base entries that rag.terms already matched."""

    def retrieve(self, terms, kb, max_items=10, text=None):
        # Merge repeats/synonyms: per entry keep the best match and its first position
        best, first_pos = {}, {}
        for term in terms:
            if not term.entry_ids:
                continue
            entry_id = term.entry_ids[0]
            quality = (TYPE_ORDER[term.match_type], -term.score)
            if entry_id not in best or quality < (TYPE_ORDER[best[entry_id].match_type], -best[entry_id].score):
                best[entry_id] = term
            first_pos[entry_id] = min(first_pos.get(entry_id, term.span[0]), term.span[0])

        def rank(entry_id):
            t = best[entry_id]
            return TYPE_ORDER[t.match_type], -t.score, first_pos[entry_id]

        passages = []
        for entry_id in sorted(best, key=rank):
            term = best[entry_id]
            entry = kb.entries[entry_id]
            if not entry.url:
                print(f"Warning: skipping '{entry.title}' ({entry_id}): no source URL", file=sys.stderr)
                continue
            summary = shorten(entry.summary) if entry.origin == "medlineplus" else entry.summary
            if not summary:
                continue
            passages.append(Passage(
                id=entry_id, term=term.term, title=entry.title, text=summary,
                source=entry.source, url=entry.url,
                score=round(TYPE_WEIGHT[term.match_type] * term.score / 100, 3),
                match_type=term.match_type, origin=entry.origin,
                ambiguous=entry.ambiguous, verified=entry.verified,
            ))
            if len(passages) == max_items:
                break
        return passages


def retrieve(terms, kb, max_items=10, text=None, retriever=None):
    """Convenience wrapper: lexical retrieval unless another Retriever is given."""
    return (retriever or LexicalRetriever()).retrieve(terms, kb, max_items=max_items, text=text)
