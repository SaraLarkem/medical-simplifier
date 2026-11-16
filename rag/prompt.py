"""
prompt.py
Project: Medical Information Simplifier
Purpose: Build the language-model request from the redacted text and the retrieved
         trusted definitions, add the Sources section in code, and check the [S#]
         citations in the model's answer.

Kept separate from mvp_pipeline.py so it can be unit-tested without Presidio/Tesseract.
"""

import re

# Added AFTER the existing system prompt; the original rules stay unchanged.
RAG_RULES = """
TRUSTED DEFINITIONS: The user message ends with numbered trusted definitions [S1], [S2], ... taken from MedlinePlus and a curated medical glossary. When trusted definitions are provided, these rules replace step 2 above:
- For a term that has a trusted definition, explain it using ONLY that definition. You may simplify the wording, but do not add medical facts that are not in the definition. Put its citation, e.g. [S1], straight after the explanation.
- For any other difficult medical term that has no trusted definition, write the term followed by "no trusted definition found". Do not explain it from your own knowledge.
- If a definition is marked AMBIGUOUS, list its possible meanings, do not choose one, and suggest confirming the meaning with the pharmacist or doctor.
- Only cite numbers that appear in the list. Do not write a Sources section; it is added automatically."""

ATTRIBUTION = ("Definitions from MedlinePlus (U.S. National Library of Medicine) "
               "and a curated glossary.")

_CITATION = re.compile(r"\[S(\d+)\]")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_context(passages):
    """Numbered definitions block: [S1] Term: definition (Source, URL)."""
    lines = []
    for i, p in enumerate(passages, 1):
        flag = " [AMBIGUOUS - do not choose a meaning]" if p.ambiguous else ""
        lines.append(f"[S{i}] {p.term}{flag}: {p.text} ({p.source}, {p.url})")
    return "\n".join(lines)


def build_system_prompt(base_prompt, passages):
    return base_prompt + "\n" + RAG_RULES if passages else base_prompt


def build_messages(base_prompt, safe_text, passages):
    """Chat messages for the LLM. Without passages this is exactly the original request."""
    user = f"Simplify this medical text: {safe_text}"
    if passages:
        user += "\n\nTrusted definitions:\n" + build_context(passages)
    return [
        {"role": "system", "content": build_system_prompt(base_prompt, passages)},
        {"role": "user", "content": user},
    ]


def format_sources(passages):
    """The Sources section, generated from the passages (never written by the model)."""
    if not passages:
        return ""
    lines = ["Sources"]
    for i, p in enumerate(passages, 1):
        lines.append(f"[S{i}] {p.term}: {p.source}, {p.url}")
    lines.append(ATTRIBUTION)
    return "\n".join(lines)


def strip_reasoning(answer):
    """Remove <think>...</think> blocks some models emit before the answer."""
    return _THINK.sub("", answer).strip()


def check_citations(answer, n_sources):
    """Which [S#] the answer cites, which do not exist, and which sources it ignored."""
    cited = {int(n) for n in _CITATION.findall(strip_reasoning(answer))}
    valid = set(range(1, n_sources + 1))
    return {
        "cited": sorted(cited & valid),
        "invalid": sorted(cited - valid),
        "uncited": sorted(valid - cited),
    }
