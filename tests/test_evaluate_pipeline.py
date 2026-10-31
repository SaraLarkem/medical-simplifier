"""Tests for evaluate_pipeline.py. Run with:  pytest -q"""
import random

import numpy as np
from PIL import Image

import evaluate_pipeline as ev
import generate_synthetic as gen


def _dp_levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_bitparallel_levenshtein_matches_reference():
    rng = random.Random(0)
    for _ in range(500):
        a = "".join(rng.choice("abc ") for _ in range(rng.randint(0, 40)))
        b = "".join(rng.choice("abc ") for _ in range(rng.randint(0, 40)))
        assert ev._lev_bitparallel(a, b) == _dp_levenshtein(a, b), (a, b)
    words_a, words_b = "the cat sat on the mat".split(), "the dog sat on mat today".split()
    assert ev._lev_bitparallel(words_a, words_b) == _dp_levenshtein(words_a, words_b)


def test_cer_and_wer():
    assert ev.cer("hello world", "hello world") == 0
    assert ev.cer("hello   world\n", "hello world") == 0          # whitespace-insensitive
    assert abs(ev.cer("abcd", "abxd") - 0.25) < 1e-9
    assert abs(ev.wer("a b c d", "a b x d") - 0.25) < 1e-9


def test_apply_spans_merges_overlaps_and_replaces():
    text = "John Smith lives here"
    out = ev.apply_spans(text, [(0, 4, "PERSON"), (0, 10, "PERSON"), (11, 16, "X")])
    assert out == "<PERSON> <X> here"


def _doc():
    return {
        "pii": [{"type": "PPS_NUMBER", "text": "1234567T"}, {"type": "PERSON", "text": "Aoife Kelly"}],
        "keep": [{"type": "DOSE", "text": "500 mg"}, {"type": "DURATION", "text": "7 days"}],
        "not_pii": [{"type": "EPONYM", "text": "Crohn's"}],
    }


def test_score_redaction_counts_leaks_and_destroyed_text():
    text = "Aoife Kelly 1234567T takes 500 mg for 7 days. Crohn's disease."
    # detector redacts the name and (wrongly) the duration + eponym; misses the PPS number
    redacted = "<PERSON> 1234567T takes 500 mg for <DATE_TIME>. <PERSON> disease."
    r = ev.score_redaction(_doc(), text, redacted)
    assert r[("pii", "PPS_NUMBER")]["bad"] == 1          # leaked
    assert r[("pii", "PERSON")]["bad"] == 0              # redacted properly
    assert r[("keep", "DOSE")]["bad"] == 0               # survived
    assert r[("keep", "DURATION")]["bad"] == 1           # destroyed
    assert r[("not_pii", "EPONYM")]["bad"] == 1          # false positive


def test_ocr_lost_pii_is_not_counted_as_a_leak():
    text = "Aoife Kelly 12345G7T"                         # OCR garbled the PPS number
    r = ev.score_redaction(_doc(), text, "<PERSON> 12345G7T")
    assert r[("pii", "PPS_NUMBER")]["present"] == 0
    assert r[("pii", "PPS_NUMBER")]["bad"] == 0


def test_relaxed_prf_counts():
    doc = {"pii": [{"start": 0, "end": 5}, {"start": 10, "end": 15}]}
    c = ev.relaxed_prf_counts(doc, [(0, 4, "A"), (20, 25, "B")])
    assert c == {"det_total": 2, "det_tp": 1, "true_total": 2, "true_hit": 1}


def test_regex_baseline_finds_every_structured_identifier():
    """Also guards the generator: formats it emits must match the documented formats."""
    structured = {"EMAIL_ADDRESS", "PPS_NUMBER", "EIRCODE", "PHONE_NUMBER", "DATE_OF_BIRTH", "DATE_TIME", "MRN"}
    for i in range(60):
        rng = random.Random(f"t-{i}")
        doc, _ = gen.BUILDERS[gen.DOC_TYPES[i % 4]](rng)
        found = [(s, e) for s, e, _ in ev.detect_regex_ids(doc.text)]
        for span in doc.spans["pii"]:
            if span["type"] in structured:
                assert any(s <= span["start"] and span["end"] <= e for s, e in found), (doc.doc_type, span)


def test_regex_baseline_leaves_clinical_text_alone():
    """Doses, durations, lab values and eponyms must survive. (The business phone number is
    deliberately excluded: a regex cannot tell it from a personal number, which is exactly the
    look-alike false positive the harness reports.)"""
    for i in range(40):
        doc, _ = gen.BUILDERS[gen.DOC_TYPES[i % 4]](random.Random(f"k-{i}"))
        redacted = ev.apply_spans(doc.text, ev.detect_regex_ids(doc.text))
        for span in doc.spans["keep"] + [s for s in doc.spans["not_pii"] if s["type"] == "EPONYM"]:
            assert span["text"] in redacted, span


def test_regex_baseline_flags_business_phone_as_known_false_positive():
    doc, _ = gen.build_prescription_label(random.Random(3))
    redacted = ev.apply_spans(doc.text, ev.detect_regex_ids(doc.text))
    assert "021 555 0100" not in redacted


def test_estimate_skew_has_correct_sign():
    doc, _ = gen.build_discharge_summary(random.Random(1))
    g = np.asarray(gen.render(doc))
    tilted = np.asarray(Image.fromarray(g).rotate(2.0, resample=Image.BICUBIC, fillcolor=255))
    assert abs(ev.estimate_skew(tilted) - (-2.0)) <= 0.5


def test_all_preprocess_configs_run():
    g = np.full((300, 400), 255, np.uint8)
    g[100:110, 50:350] = 0
    for name, fn in ev.PREPROCESS.items():
        out = fn(g)
        assert out.ndim == 2 and out.shape[0] >= 300, name
