"""Tests for the RAG glossary stage (rag/). Run with:  pytest -q

All tests use a small in-memory fixture knowledge base, so the MedlinePlus download is
not needed. The only real data file used is the committed curated_glossary.csv.
"""
import ast
import csv
import json
from pathlib import Path

import pytest

from rag.build_kb import find_latest_xml, html_to_text, parse_topics
from rag.kb import Entry, build_index, load_kb, normalise
from rag.prompt import (ATTRIBUTION, RAG_RULES, build_context, build_messages,
                        build_system_prompt, check_citations, format_sources)
from rag.retriever import retrieve, shorten
from rag.terms import extract_terms

ROOT = Path(__file__).resolve().parents[1]


def mlp(i, title, aliases=(), summary=None, url="auto"):
    return Entry(id=f"mlp-{i}", title=title, aliases=list(aliases),
                 summary=summary or f"{title} is a health topic. It has a summary.",
                 url=f"https://medlineplus.gov/{i}.html" if url == "auto" else url,
                 source="MedlinePlus")


def cur(i, term, category, aliases=(), ambiguous=False, verified=True):
    return Entry(id=f"cur-{i}", title=term, aliases=list(aliases), summary=f"Definition of {term}.",
                 url=f"https://example.org/{i}", source="Curated test source", category=category,
                 ambiguous=ambiguous, origin="curated", verified=verified)


@pytest.fixture(scope="module")
def kb():
    entries = [
        mlp(1, "Anemia", ["Iron deficiency anemia"]),
        mlp(2, "Asthma", summary="What is asthma? Asthma is a chronic lung disease. It affects your "
                                 "airways. Symptoms include: Wheezing Coughing"),
        mlp(3, "High Blood Pressure", ["Hypertension", "Blood Pressure", "HTN"]),
        mlp(4, "Hepatitis B"),
        mlp(5, "Hemoglobin"),
        mlp(6, "Platelets"),
        mlp(7, "GERD", ["Gastroesophageal reflux disease"]),
        mlp(8, "Steroids", ["Prednisone"]),
        mlp(9, "Sarcoidosis", url=""),                # no source URL: must never be returned
        mlp(10, "Location"),                          # same word as the <LOCATION> placeholder
        mlp(11, "Sjögren's Syndrome"),
        mlp(12, "Celiac Disease"),
        cur(1, "OD", "abbreviation", ["o.d."], ambiguous=True),
        cur(2, "BD", "abbreviation", ["b.d."]),
        cur(3, "TSH", "lab_test", ["Thyroid stimulating hormone"], verified=False),
        cur(4, "Haemoglobin", "lab_test"),
    ]
    return build_index(entries, stoplist=["Blood Pressure"])


def found(text, kb, **kw):
    """{(term as written, KB title of best entry)} for convenience."""
    return {(t.term, kb.entries[t.entry_ids[0]].title) for t in extract_terms(text, kb, **kw)}


# --------------------------------------------------------------------------- #
# Term extraction: known terms and placeholders
# --------------------------------------------------------------------------- #
def test_known_terms_are_found(kb):
    hits = found("Patient <PERSON> has Asthma and Hypertension.\nIron deficiency anaemia noted.", kb)
    assert ("Asthma", "Asthma") in hits
    assert ("Hypertension", "High Blood Pressure") in hits
    # longest match wins: one term for the whole phrase, not "anaemia" on its own
    assert ("Iron deficiency anaemia", "Anemia") in hits
    assert not any(term == "anaemia" for term, _ in hits)


def test_spans_point_back_into_the_text(kb):
    text = "Seen by <PERSON>. Asthma, TSH 4.1, Hepatitis B vaccine."
    for t in extract_terms(text, kb):
        assert text[t.span[0]:t.span[1]] == t.term


def test_redaction_placeholders_are_never_matched(kb):
    assert extract_terms("<LOCATION> <PERSON> <DATE_TIME>", kb) == []
    text = "Lives at <LOCATION>. Location of clinic changed."
    terms = extract_terms(text, kb)
    assert [t.term for t in terms] == ["Location"]
    assert all("<" not in t.term and ">" not in t.term for t in terms)


def test_phrases_do_not_cross_placeholders_or_line_breaks(kb):
    for text in ("Iron <PERSON> deficiency anaemia", "Iron deficiency\nanaemia"):
        terms = extract_terms(text, kb)
        assert all("\n" not in t.term and "<" not in t.term for t in terms)
        assert not any(t.term.startswith("Iron") for t in terms)


# --------------------------------------------------------------------------- #
# Term extraction: OCR damage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("damaged, title", [
    ("Haemog1obin", "Hemoglobin"),     # digit-for-letter repair
    ("Platelcts", "Platelets"),        # rapidfuzz
    ("HepatitisB", "Hepatitis B"),     # words merged by OCR
])
def test_fuzzy_matching_recovers_ocr_damage(kb, damaged, title):
    terms = extract_terms(f"Result: {damaged} low", kb)
    assert len(terms) == 1
    assert terms[0].match_type == "fuzzy"
    assert kb.entries[terms[0].entry_ids[0]].title == title


def test_fuzzy_threshold_min_length_and_switch_are_configurable(kb):
    assert found("Platelcts", kb, fuzzy_threshold=99) == set()
    assert found("Platelcts", kb, fuzzy=False) == set()
    assert ("Astma", "Asthma") in found("Astma", kb)                 # 5 letters: allowed by default
    assert found("Astma", kb, fuzzy_min_len=6) == set()


def test_fuzzy_does_not_match_a_different_drug(kb):
    # Regression: with a +/-2 length window, Prednisolone matched Prednisone
    assert found("Prednisolone 30 mg", kb) == set()


# --------------------------------------------------------------------------- #
# Term extraction: abbreviations
# --------------------------------------------------------------------------- #
def test_abbreviations_match_exactly(kb):
    assert ("OD", "OD") in found("Take one tablet OD", kb)
    assert ("b.d", "BD") in found("One tab b.d. with food", kb)
    assert ("TSH", "TSH") in found("TSH 4.1 mIU/L", kb)


@pytest.mark.parametrize("text", ["take one tablet od", "ODOUR", "PERIOD", "Code", "GOOD",
                                  "tsh", "BDS"])
def test_abbreviations_do_not_match_inside_words_or_other_case(kb, text):
    assert found(text, kb) == set()


# --------------------------------------------------------------------------- #
# Normalisation and stoplist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word, expected", [
    ("Haemoglobin", "hemoglobin"), ("anaemia", "anemia"), ("paediatric", "pediatric"),
    ("oesophageal", "esophageal"), ("diarrhoea", "diarrhea"), ("coeliac", "celiac"),
    ("dyspnoea", "dyspnea"),
    # genuine US spellings that must NOT be folded
    ("gastroesophageal", "gastroesophageal"), ("aerosol", "aerosol"),
    ("pneumoniae", "pneumoniae"), ("bioethics", "bioethics"), ("shoes", "shoes"),
    # accents, apostrophes, hyphens
    ("Sjögren's", "sjogrens"), ("Gastro-oesophageal", "gastro esophageal"),
])
def test_normalise(word, expected):
    assert normalise(word) == expected


def test_shoes_and_shes_do_not_collide():
    assert normalise("shoes") != normalise("she's")


def test_spelling_variants_reach_us_topics(kb):
    assert ("Gastro-oesophageal reflux disease", "GERD") in found("Gastro-oesophageal reflux disease", kb)
    assert ("Coeliac disease", "Celiac Disease") in found("Coeliac disease", kb)
    assert ("Sjogren's syndrome", "Sjögren's Syndrome") in found("Sjogren's syndrome", kb)


def test_stoplist_removes_a_name_but_not_its_topic(kb):
    assert found("Blood Pressure: 120/80", kb) == set()
    assert ("Hypertension", "High Blood Pressure") in found("Hypertension", kb)


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
def passages_for(text, kb, **kw):
    return retrieve(extract_terms(text, kb), kb, **kw)


def test_every_passage_has_a_source_url(kb, capsys):
    ps = passages_for("Sarcoidosis, Asthma, OD, TSH, Hypertension, Platelcts", kb)
    assert ps
    assert all(p.url.startswith("https://") and p.source for p in ps)
    assert "Sarcoidosis" not in {p.title for p in ps}
    assert "no source URL" in capsys.readouterr().err


def test_repeats_and_synonyms_become_one_passage(kb):
    ps = passages_for("Asthma. Asthma again. Hypertension, HTN.", kb)
    assert [p.title for p in ps] == ["Asthma", "High Blood Pressure"]


def test_ranking_exact_then_alias_then_fuzzy(kb):
    ps = passages_for("Platelcts low. Hypertension. Asthma.", kb)
    assert [p.match_type for p in ps] == ["exact", "alias", "fuzzy"]
    assert [p.title for p in ps] == ["Asthma", "High Blood Pressure", "Platelets"]
    assert ps[0].score > ps[1].score > ps[2].score


def test_max_items(kb):
    assert len(passages_for("Asthma, Hypertension, Platelcts", kb, max_items=2)) == 2


def test_medlineplus_preferred_over_curated_on_a_tie(kb):
    (p,) = passages_for("Haemoglobin 9.1", kb)
    assert p.origin == "medlineplus" and p.title == "Hemoglobin"


def test_ambiguous_and_verified_flags_pass_through(kb):
    by_title = {p.title: p for p in passages_for("OD, TSH, Asthma", kb)}
    assert by_title["OD"].ambiguous is True
    assert by_title["TSH"].verified is False
    assert by_title["Asthma"].verified is True and by_title["Asthma"].ambiguous is False


def test_shorten_skips_heading_questions_and_list_lead_ins():
    s = "What is asthma? Asthma is a chronic lung disease. It affects your airways. More here."
    assert shorten(s) == "Asthma is a chronic lung disease. It affects your airways."
    assert shorten("Measles is caused by a virus. Symptoms include: Fever Cough") == \
        "Measles is caused by a virus."


def test_shorten_when_the_first_sentence_introduces_a_list():
    long_lead = "Tetanus, diphtheria, and pertussis are serious bacterial infections: Tetanus is rare."
    assert shorten(long_lead) == "Tetanus, diphtheria, and pertussis are serious bacterial infections."
    short_lead = "The B vitamins are: B1 B2 B3. They help."
    assert shorten(short_lead) == "The B vitamins are: B1 B2 B3."


def test_shorten_caps_length():
    s = shorten(" ".join(["word"] * 200) + ".")
    assert len(s) <= 401 and s.endswith("…")


# --------------------------------------------------------------------------- #
# Prompt building and citations
# --------------------------------------------------------------------------- #
def original_system_prompt():
    """SYSTEM_PROMPT from mvp_pipeline.py, read without importing Presidio/Tesseract."""
    tree = ast.parse((ROOT / "mvp_pipeline.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SYSTEM_PROMPT":
            return ast.literal_eval(node.value)
    raise AssertionError("SYSTEM_PROMPT not found in mvp_pipeline.py")


def test_without_passages_the_request_is_unchanged():
    base = original_system_prompt()
    assert build_messages(base, "Hb 9.1", []) == [
        {"role": "system", "content": base},
        {"role": "user", "content": "Simplify this medical text: Hb 9.1"},
    ]


def test_original_safety_rules_are_kept_with_rag(kb):
    base = original_system_prompt()
    system = build_system_prompt(base, passages_for("Asthma", kb))
    assert system.startswith(base)
    assert "strictly non-diagnostic" in system and "Do not diagnose" in system
    assert RAG_RULES in system


def test_context_has_numbered_citations_with_sources(kb):
    ps = passages_for("Asthma and OD", kb)
    context = build_context(ps)
    lines = context.splitlines()
    assert len(lines) == len(ps)
    for i, (line, p) in enumerate(zip(lines, ps), 1):
        assert line.startswith(f"[S{i}] {p.term}")
        assert line.endswith(f"({p.source}, {p.url})")
    assert "[AMBIGUOUS" in next(line for line in lines if "OD" in line)
    user = build_messages("base", "Asthma and OD", ps)[1]["content"]
    assert "Trusted definitions:\n" + context in user


def test_sources_section(kb):
    ps = passages_for("Asthma, Hypertension, OD", kb)
    sources = format_sources(ps).splitlines()
    assert sources[0] == "Sources"
    for i, p in enumerate(ps, 1):
        assert sources[i] == f"[S{i}] {p.term}: {p.source}, {p.url}"
    assert sources[-1] == ATTRIBUTION
    assert format_sources([]) == ""


def test_check_citations_finds_invalid_and_uncited():
    answer = "<think>maybe [S2]</think> Anaemia means ... [S1]. OD ... [S3] [S3]. Other [S9]."
    assert check_citations(answer, 3) == {"cited": [1, 3], "invalid": [9], "uncited": [2]}


# --------------------------------------------------------------------------- #
# Knowledge-base building and loading
# --------------------------------------------------------------------------- #
XML = """<?xml version="1.0" encoding="UTF-8"?>
<health-topics total="2" date-generated="11/13/2026 08:58:25">
<health-topic title="Anemia" url="https://medlineplus.gov/anemia.html" id="139" language="English">
<also-called>Iron poor blood</also-called>
<also-called>anemia</also-called>
<see-reference>Anaemia</see-reference>
<see-reference>Iron poor blood</see-reference>
<full-summary>&lt;p&gt;Anemia means too few &lt;a href="x"&gt;red cells&lt;/a&gt; &amp;amp; low iron.&lt;/p&gt;</full-summary>
</health-topic>
<health-topic title="Anemia" url="https://medlineplus.gov/spanish/anemia.html" id="140" language="Spanish">
<full-summary>&lt;p&gt;La anemia&lt;/p&gt;</full-summary>
</health-topic>
</health-topics>"""


def test_html_to_text():
    assert html_to_text("<p>A &amp; B</p><ul><li>x</li><li>y</li></ul>") == "A & B x y"


def test_parse_topics(tmp_path):
    path = tmp_path / "mplus_topics_2026-11-13.xml"
    path.write_text(XML, encoding="utf-8")
    records, stats = parse_topics(path)
    assert stats["skipped_non_english"] == 1 and len(records) == 1
    r = records[0]
    assert r["id"] == "mlp-139" and r["source"] == "MedlinePlus"
    assert r["aliases"] == ["Iron poor blood", "Anaemia"]          # de-duplicated, title removed
    assert r["summary"] == "Anemia means too few red cells & low iron."


def test_find_latest_xml(tmp_path):
    for name in ("mplus_topics_2026-01-01.xml", "mplus_topics_2026-11-13.xml",
                 "mplus_topics_compressed_2026-12-01.zip"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert find_latest_xml(tmp_path).name == "mplus_topics_2026-11-13.xml"
    with pytest.raises(SystemExit):
        find_latest_xml(tmp_path / "empty")


def test_load_kb_reports_missing_files_instead_of_crashing(tmp_path):
    kb = load_kb(tmp_path / "none.jsonl", tmp_path / "none.csv", tmp_path / "none.txt")
    assert len(kb) == 0 and len(kb.missing) == 2


def test_load_kb_reads_both_sources(tmp_path):
    jsonl = tmp_path / "kb.jsonl"
    jsonl.write_text(json.dumps({"id": "mlp-2", "title": "Asthma", "aliases": [], "summary": "S.",
                                 "url": "https://medlineplus.gov/asthma.html",
                                 "source": "MedlinePlus"}) + "\n", encoding="utf-8")
    csv_path = tmp_path / "glossary.csv"
    csv_path.write_text("term,aliases,category,definition,source,url,ambiguous,verified\n"
                        'OD,o.d.,abbreviation,"Once a day, usually.",BNF,https://x.org,true,false\n',
                        encoding="utf-8")
    kb = load_kb(jsonl, csv_path, tmp_path / "none.txt")
    od = next(e for e in kb.entries.values() if e.title == "OD")
    assert len(kb) == 2 and kb.missing == []
    assert od.origin == "curated" and od.ambiguous is True and od.verified is False
    assert od.summary == "Once a day, usually."


def test_committed_curated_glossary_is_well_formed():
    path = ROOT / "data" / "kb" / "curated_glossary.csv"
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["term", "aliases", "category", "definition", "source",
                                     "url", "ambiguous", "verified"]
        rows = list(reader)
    assert rows
    terms = [r["term"] for r in rows]
    assert len(terms) == len(set(terms)), "duplicate terms"
    for r in rows:
        assert None not in r and None not in r.values(), f"bad column count: {r}"
        assert r["term"] and r["definition"] and r["source"]
        assert r["url"].startswith("https://"), r["term"]
        assert r["category"] in {"abbreviation", "lab_test", "vaccine"}, r["term"]
        assert r["ambiguous"] in {"true", "false"} and r["verified"] in {"true", "false"}, r["term"]
