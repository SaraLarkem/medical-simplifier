"""
generate_synthetic.py
Project: Medical Information Simplifier
Purpose: Generate SYNTHETIC medical documents (images + ground truth) so that every
         pipeline stage can be measured: OCR accuracy, PII-redaction precision/recall,
         over-redaction of clinically meaningful text, and output faithfulness.

All people, identifiers and numbers are randomly generated. No real patient data is used.
Requires only: Pillow, numpy.

Usage:
    python generate_synthetic.py --n 40 --seed 42 --out data/synthetic
"""

import argparse
import io
import json
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PAGE_W = 1240   # A4 width at ~150 dpi
MARGIN = 80
DOC_TYPES = ["lab_report", "discharge_summary", "prescription_label", "clinic_form"]
VARIANTS = ["clean", "scan", "phone"]


# --------------------------------------------------------------------------- #
# Annotated text segments
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Seg:
    text: str
    kind: str    # "pii" | "keep" | "not_pii"
    label: str


def P(text, label):
    """Personal data that SHOULD be redacted before leaving the device."""
    return Seg(text, "pii", label)


def K(text, label):
    """Clinically meaningful text that must SURVIVE redaction (doses, durations, lab values)."""
    return Seg(text, "keep", label)


def N(text, label):
    """Text that looks like PII but is not (e.g. medical eponyms, business phone numbers)."""
    return Seg(text, "not_pii", label)


class Doc:
    """Builds ground-truth text, span annotations and render instructions together,
    so the offsets can never drift out of sync with what is drawn."""

    def __init__(self, doc_type):
        self.doc_type = doc_type
        self.text = ""
        self.rows = []
        self.spans = {"pii": [], "keep": [], "not_pii": []}
        self.terms = []
        self.checkboxes = []

    def _flatten(self, parts):
        out = ""
        for part in parts:
            if isinstance(part, Seg):
                start = len(self.text)
                self.text += part.text
                self.spans[part.kind].append(
                    {"type": part.label, "text": part.text, "start": start, "end": start + len(part.text)}
                )
                out += part.text
            else:
                self.text += part
                out += part
        return out

    def row(self, cells, style="body", cols=None, check=None):
        rendered = []
        for i, cell in enumerate(cells):
            if i:
                self.text += "  "
            parts = cell if isinstance(cell, list) else [cell]
            rendered.append(self._flatten(parts))
        self.text += "\n"
        self.rows.append({"style": style, "cells": rendered, "cols": cols, "check": check})

    def line(self, *parts, style="body"):
        self.row([list(parts)], style=style)

    def blank(self):
        self.text += "\n"
        self.rows.append({"style": "blank", "cells": [], "cols": None, "check": None})

    def term(self, t):
        if t not in self.terms:
            self.terms.append(t)


# --------------------------------------------------------------------------- #
# Fake identity data (all synthetic)
# --------------------------------------------------------------------------- #
FIRST = ["Aoife", "Seán", "Ciara", "Conor", "Niamh", "Oisín", "Saoirse", "Cian", "Emma", "Liam",
         "Fatima", "Tomasz", "Priya", "Mateus", "Grace", "Adaeze", "Ahmed", "Sophie", "Daniel", "Mei"]
LAST = ["Murphy", "O'Sullivan", "Kelly", "Byrne", "Ryan", "O'Brien", "Walsh", "Doyle", "McCarthy",
        "Gallagher", "Kavanagh", "Nolan", "Okafor", "Hassan", "Nowak", "Silva", "Chen", "Patel",
        "Ibrahim", "Kowalski"]
STREETS = ["Willow Park", "Oakfield Road", "Church Street", "Mill Lane", "Riverview Close",
           "Harbour Road", "Elm Grove", "Castle Avenue"]
TOWNS = [("Ballincollig", "Co. Cork"), ("Naas", "Co. Kildare"), ("Salthill", "Co. Galway"),
         ("Terenure", "Dublin 6W"), ("Raheny", "Dublin 5"), ("Ennis", "Co. Clare"),
         ("Tralee", "Co. Kerry"), ("Athlone", "Co. Westmeath")]
ROUTING_KEYS = ["D02", "D05", "D6W", "T12", "H91", "P85", "V94", "R14", "A83", "T23"]
EIRCODE_CHARS = "0123456789ACDEFHKNPRTVWXY"


def rand_date(rng, y0, y1):
    return date(rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28))


def fmt_date(d, style=0):
    return d.strftime({0: "%d/%m/%Y", 1: "%d %B %Y", 2: "%Y-%m-%d"}[style])


def make_phone(rng):
    r = rng.random()
    if r < 0.5:
        return f"08{rng.choice('35679')} {rng.randrange(100, 999)} {rng.randrange(1000, 9999)}"
    if r < 0.75:
        return f"+353 8{rng.choice('35679')} {rng.randrange(100, 999)} {rng.randrange(1000, 9999)}"
    return f"({rng.choice(['021', '091', '01', '061', '045'])}) {rng.randrange(100, 999)}{rng.randrange(1000, 9999)}"


def make_eircode(rng):
    return f"{rng.choice(ROUTING_KEYS)} {''.join(rng.choice(EIRCODE_CHARS) for _ in range(4))}"


def make_pps(rng):
    pps = f"{rng.randrange(1_000_000, 9_999_999)}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVW')}"
    if rng.random() < 0.3:
        pps += rng.choice(["A", "B", "H", "X"])
    return pps


def make_person(rng):
    first, last = rng.choice(FIRST), rng.choice(LAST)
    town, county = rng.choice(TOWNS)
    slug = re.sub(r"[^a-z]", "", f"{first}.{last}".lower().replace("á", "a").replace("í", "i"))
    return {
        "first": first,
        "last": last,
        "name": f"{first} {last}",
        "dob": fmt_date(rand_date(rng, 1935, 2005), rng.randint(0, 2)),
        "mrn": str(rng.randrange(1_000_000, 9_999_999)),
        "pps": make_pps(rng),
        "address": f"{rng.randint(1, 120)} {rng.choice(STREETS)}, {town}, {county}",
        "eircode": make_eircode(rng),
        "phone": make_phone(rng),
        "email": f"{slug}{rng.randint(1, 99)}@example.com",
    }


def clinician(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def patient_block(d, p):
    d.line("Patient name: ", P(p["name"], "PERSON"))
    d.line("Date of birth: ", P(p["dob"], "DATE_OF_BIRTH"), "     MRN: ", P(p["mrn"], "MRN"))
    d.line("PPS number: ", P(p["pps"], "PPS_NUMBER"))
    d.line("Address: ", P(p["address"], "ADDRESS"), ", ", P(p["eircode"], "EIRCODE"))
    d.line("Phone: ", P(p["phone"], "PHONE_NUMBER"), "     Email: ", P(p["email"], "EMAIL_ADDRESS"))


# --------------------------------------------------------------------------- #
# Clinical content banks
# --------------------------------------------------------------------------- #
# name, unit, lo, hi, displayed range, decimals
LAB_TESTS = [
    ("Haemoglobin", "g/dL", 12.0, 17.0, "12.0 - 17.0", 1),
    ("White cell count", "x10^9/L", 4.0, 11.0, "4.0 - 11.0", 1),
    ("Platelets", "x10^9/L", 150, 400, "150 - 400", 0),
    ("Sodium", "mmol/L", 135, 145, "135 - 145", 0),
    ("Potassium", "mmol/L", 3.5, 5.3, "3.5 - 5.3", 1),
    ("Creatinine", "umol/L", 60, 110, "60 - 110", 0),
    ("HbA1c", "mmol/mol", 20, 42, "20 - 42", 0),
    ("Total cholesterol", "mmol/L", None, 5.0, "< 5.0", 1),
    ("TSH", "mIU/L", 0.4, 4.0, "0.4 - 4.0", 1),
    ("CRP", "mg/L", None, 5, "< 5", 0),
]
LAB_JARGON = {"Haemoglobin", "HbA1c", "TSH", "CRP", "Creatinine", "Platelets"}

# name, dose, [abbreviation, plain phrase], duration, form
MEDS = [
    ("Metformin", "500 mg", ["BD", "twice daily"], None, "tablet"),
    ("Atorvastatin", "20 mg", ["ON", "once daily at night"], None, "tablet"),
    ("Ramipril", "5 mg", ["OD", "once daily"], None, "capsule"),
    ("Amoxicillin", "500 mg", ["TDS", "three times daily"], "7 days", "capsule"),
    ("Levothyroxine", "50 micrograms", ["OD", "once daily"], None, "tablet"),
    ("Omeprazole", "20 mg", ["OD", "once daily"], "8 weeks", "capsule"),
    ("Salbutamol inhaler", "100 micrograms", ["PRN", "as needed"], None, "inhaler"),
    ("Prednisolone", "30 mg", ["OD", "once daily"], "5 days", "tablet"),
    ("Paracetamol", "1 g", ["QDS", "up to four times daily"], None, "tablet"),
    ("Amlodipine", "5 mg", ["OD", "once daily"], None, "tablet"),
]

# parts, jargon terms
DIAGNOSES = [
    (["Community-acquired pneumonia"], ["pneumonia"]),
    (["Type 2 diabetes mellitus"], ["diabetes mellitus"]),
    (["Essential hypertension"], ["hypertension"]),
    ([N("Hashimoto's", "EPONYM"), " thyroiditis"], ["thyroiditis"]),
    ([N("Crohn's", "EPONYM"), " disease"], []),
    ([N("Parkinson's", "EPONYM"), " disease"], []),
    (["Atrial fibrillation"], ["fibrillation"]),
    (["Iron deficiency anaemia"], ["anaemia"]),
    (["Gastro-oesophageal reflux disease"], ["reflux"]),
]
INVESTIGATIONS = ["Chest X-ray", "ECG", "Full blood count", "Urinalysis", "CT thorax"]


def med_parts(rng, d, med):
    """Return annotated parts for 'Name dose frequency [for duration]'."""
    name, dose, freqs, duration, _ = med
    abbr = rng.random() < 0.5
    freq = freqs[0] if abbr else freqs[1]
    if abbr:
        d.term(freq)
    parts = [f"{name} ", K(dose, "DOSE"), " ", K(freq, "FREQUENCY")]
    if duration:
        parts += [" for ", K(duration, "DURATION")]
    return parts


# --------------------------------------------------------------------------- #
# Document builders
# --------------------------------------------------------------------------- #
def build_lab_report(rng):
    d, p = Doc("lab_report"), make_person(rng)
    d.line("Riverside Regional Hospital", style="h1")
    d.line("Department of Clinical Biochemistry and Haematology", style="h2")
    d.blank()
    patient_block(d, p)
    d.line("Requested by: Dr. ", P(clinician(rng), "PERSON"))
    d.line("Date collected: ", P(fmt_date(rand_date(rng, 2025, 2026)), "DATE_TIME"))
    d.blank()
    cols = [MARGIN, 470, 620, 820, 1050]
    d.row(["Test", "Result", "Units", "Reference range", "Flag"], style="bold", cols=cols)
    abnormal = []
    for name, unit, lo, hi, disp, dec in rng.sample(LAB_TESTS, rng.randint(6, 9)):
        r = rng.random()
        low = lo if lo is not None else hi * 0.25
        if r < 0.65:
            v = rng.uniform(low, hi)
        elif r < 0.82 or lo is None:
            v = hi * rng.uniform(1.05, 1.6)
        else:
            v = lo * rng.uniform(0.6, 0.95)
        shown = f"{v:.{dec}f}"
        flag = "H" if float(shown) > hi else ("L" if lo is not None and float(shown) < lo else "")
        if flag:
            abnormal.append(name)
        if name in LAB_JARGON:
            d.term(name)
        d.row([name, K(shown, "LAB_VALUE"), unit, K(disp, "REFERENCE_RANGE"), flag], cols=cols)
    d.blank()
    if abnormal:
        d.line("Comment: results outside the reference range:")
        d.line(", ".join(abnormal[:3]))
        d.line("Clinical correlation advised. Repeat in ", K(f"{rng.choice([2, 4, 6])} weeks", "DURATION"), ".")
    else:
        d.line("Comment: all results are within the reference ranges.")
    return d, p


def build_discharge_summary(rng):
    d, p = Doc("discharge_summary"), make_person(rng)
    d.line("Riverside Regional Hospital", style="h1")
    d.line("DISCHARGE SUMMARY", style="h2")
    d.blank()
    patient_block(d, p)
    admitted = rand_date(rng, 2025, 2026)
    discharged = admitted + timedelta(days=rng.randint(2, 9))
    d.line("Admitted: ", P(fmt_date(admitted), "DATE_TIME"), "     Discharged: ", P(fmt_date(discharged), "DATE_TIME"))
    d.line("Consultant: Dr. ", P(clinician(rng), "PERSON"))
    d.line("GP: Dr. ", P(clinician(rng), "PERSON"))
    d.blank()
    (main_dx, main_terms), (sec_dx, sec_terms) = rng.sample(DIAGNOSES, 2)
    d.line("Principal diagnosis: ", *main_dx)
    d.line("Secondary diagnosis: ", *sec_dx)
    for t in main_terms + sec_terms:
        d.term(t)
    d.blank()
    d.line("Investigations performed:", style="bold")
    for inv in rng.sample(INVESTIGATIONS, rng.randint(2, 3)):
        d.line("- ", K(inv, "INVESTIGATION"))
    d.blank()
    d.line("Medications on discharge:", style="bold")
    for med in rng.sample(MEDS, rng.randint(2, 4)):
        d.line("- ", *med_parts(rng, d, med))
    d.blank()
    d.line("Follow-up:", style="bold")
    d.line("- Outpatient review in ", K(f"{rng.choice([2, 4, 6])} weeks", "DURATION"))
    d.line("- Repeat blood tests in ", K(f"{rng.choice([2, 3])} months", "DURATION"))
    d.line("- Return to the Emergency Department if symptoms worsen.")
    return d, p


def build_prescription_label(rng):
    d, p = Doc("prescription_label"), make_person(rng)
    d.line("Riverside Community Pharmacy", style="h1")
    d.line("Tel: ", N("021 555 0100", "BUSINESS_PHONE"))
    d.blank()
    d.line("Patient: ", P(p["name"], "PERSON"))
    d.line("Address: ", P(p["address"], "ADDRESS"))
    d.line("Date of birth: ", P(p["dob"], "DATE_OF_BIRTH"))
    d.blank()
    name, dose, freqs, duration, form = rng.choice(MEDS)
    d.line("Medicine: ", name)
    d.line("Strength: ", K(dose, "DOSE"))
    freq = rng.choice(freqs)
    if freq in ("BD", "TDS", "OD", "ON", "PRN", "QDS"):
        d.term(freq)
    verb = "Inhale 1 puff " if form == "inhaler" else f"Take one {form} "
    parts = [verb, K(freq, "FREQUENCY")]
    if duration:
        parts += [" for ", K(duration, "DURATION")]
    d.line("Directions: ", *parts)
    d.line("Quantity: ", K(str(rng.choice([14, 28, 56, 60])), "QUANTITY"))
    d.blank()
    d.line("Dispensed: ", P(fmt_date(rand_date(rng, 2025, 2026)), "DATE_TIME"))
    d.line("Pharmacist: ", P(clinician(rng), "PERSON"))
    d.line("Keep out of the reach of children.")
    return d, p


SYSTEMS = ["Nutritional", "Skin/Nails", "Head/Face", "Eyes/Vision", "Ears/Hearing", "Nose/Sinuses",
           "Mouth/Throat", "Lymph Nodes", "Heart", "Lungs", "Abdomen", "Neurologic"]
# history, jargon terms, abnormal system, abnormality text, treatment parts
HISTORIES = [
    ("Asthma since childhood", ["Asthma"], "Lungs", "Exercise-related wheeze, known asthma",
     ["Salbutamol inhaler ", K("100 micrograms", "DOSE"), " ", K("as needed", "FREQUENCY")]),
    ("Eczema", ["Eczema"], "Skin/Nails", "Dry patches on both forearms",
     ["Emollient cream ", K("as needed", "FREQUENCY")]),
    ("Migraine", ["Migraine"], None, None,
     ["Paracetamol ", K("1 g", "DOSE"), " ", K("as needed", "FREQUENCY")]),
    ("No significant history", [], None, None, ["None"]),
]


def build_clinic_form(rng):
    d, p = Doc("clinic_form"), make_person(rng)
    birth_year = rng.randint(2000, 2007)   # college students: keep immunisation dates plausible
    p["dob"] = fmt_date(rand_date(rng, birth_year, birth_year), rng.randint(0, 2))
    guardian = f"{rng.choice(FIRST)} {p['last']}"
    d.line("MEDICAL FORM", style="h1")
    d.line("Riverside College Student Health Centre", style="h2")
    d.blank()
    d.line("1. Name (Last): ", P(p["last"], "PERSON"), "     (First): ", P(p["first"], "PERSON"))
    d.line("2. Date of Birth: ", P(p["dob"], "DATE_OF_BIRTH"), "     3. Sex: ", rng.choice(["F", "M"]))
    d.line("4. Phone: ", P(p["phone"], "PHONE_NUMBER"))
    d.line("5. Home Address: ", P(p["address"], "ADDRESS"), ", ", P(p["eircode"], "EIRCODE"))
    d.line("6. Student Cell Phone: ", P(make_phone(rng), "PHONE_NUMBER"))
    d.line("7. PPS Number: ", P(p["pps"], "PPS_NUMBER"))
    d.line("8. Entrance Date: ", P(fmt_date(rand_date(rng, 2025, 2026)), "DATE_TIME"))
    d.line("10. Name of Parent or Guardian: ", P(guardian, "PERSON"))
    d.blank()
    d.line("11. IMMUNIZATION RECORD (required)", style="bold")
    for t in ("Rubella", "Rubeola", "DPT", "Tetanus"):
        d.term(t)

    def mmyyyy():
        return f"{rng.randint(1, 12):02d}/{rng.randint(birth_year + 1, 2024)}"

    def sort_key(m):   # "MM/YYYY" -> (YYYY, MM)
        return (int(m[3:]), int(m[:2]))

    d.line("German Measles (Rubella): ", K(mmyyyy(), "CLINICAL_DATE"))
    d.line("Measles (Rubeola): ", K(mmyyyy(), "CLINICAL_DATE"))
    d.line("DPT: ", K(mmyyyy(), "CLINICAL_DATE"), "     Polio: ", K(mmyyyy(), "CLINICAL_DATE"))
    d.line("Last Tetanus Booster: ", K(mmyyyy(), "CLINICAL_DATE"))
    hep = sorted((mmyyyy() for _ in range(3)), key=sort_key)
    d.line("Hepatitis B: #1 ", K(hep[0], "CLINICAL_DATE"), "  #2 ", K(hep[1], "CLINICAL_DATE"),
           "  #3 ", K(hep[2], "CLINICAL_DATE"))
    d.blank()
    history, terms, ab_system, ab_text, treatment = rng.choice(HISTORIES)
    for t in terms:
        d.term(t)
    d.line("12. Medical History: ", history)
    d.blank()
    d.line("13. Blood Pressure: ", K(f"{rng.randint(105, 135)}/{rng.randint(65, 85)}", "VITAL"),
           "     14. Pulse: ", K(str(rng.randint(58, 90)), "VITAL"))
    d.line("15. Height: ", K(f"{rng.randint(152, 190)} cm", "VITAL"),
           "     16. Weight: ", K(f"{rng.randint(48, 95)} kg", "VITAL"))
    d.line("17. Drug Sensitivity: None known")
    d.blank()
    d.line("18. Clinical Evaluation (N = normal, A = abnormal)", style="bold")
    cols = [MARGIN, 470, 550]
    d.row(["System", "N", "A"], style="bold", cols=cols)
    for system in SYSTEMS:
        value = "A" if system == ab_system else "N"
        d.row([system], cols=cols, check=value)
        d.checkboxes.append({"item": system, "value": value})
    d.blank()
    d.line("19. Abnormalities: ", ab_text or "None")
    d.line("20. Current treatment: ", *treatment)
    d.line("21. Physical restrictions: None")
    return d, p


BUILDERS = {
    "lab_report": build_lab_report,
    "discharge_summary": build_discharge_summary,
    "prescription_label": build_prescription_label,
    "clinic_form": build_clinic_form,
}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
# style -> (font size, bold, row height)
STYLES = {"h1": (40, True, 64), "h2": (30, True, 48), "bold": (26, True, 40),
          "body": (26, False, 40), "blank": (26, False, 24)}
REG_FONTS = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf"]
BOLD_FONTS = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
_FONTS = {}


def get_font(size, bold):
    key = (size, bold)
    if key not in _FONTS:
        font = None
        for name in (BOLD_FONTS if bold else REG_FONTS):
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:
                font = ImageFont.load_default()
        _FONTS[key] = font
    return _FONTS[key]


def text_width(font, text):
    try:
        return font.getlength(text)
    except AttributeError:
        return font.getsize(text)[0]


def render(doc):
    heights = [STYLES[r["style"]][2] for r in doc.rows]
    img = Image.new("L", (PAGE_W, sum(heights) + 2 * MARGIN), 255)
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for row, h in zip(doc.rows, heights):
        size, bold, _ = STYLES[row["style"]]
        font = get_font(size, bold)
        cols = row["cols"]
        for i, text in enumerate(row["cells"]):
            x = cols[i] if cols else MARGIN
            if text:
                if x + text_width(font, text) > PAGE_W - MARGIN:
                    raise ValueError(f"Line too wide for page: {text!r}")
                draw.text((x, y), text, font=font, fill=0)
        if row["check"]:
            for idx, letter in ((1, "N"), (2, "A")):
                bx, by = cols[idx] - 4, y + 2
                draw.rectangle([bx, by, bx + 24, by + 24], outline=0, width=2)
                if row["check"] == letter:
                    draw.line([bx + 4, by + 4, bx + 20, by + 20], fill=0, width=3)
                    draw.line([bx + 4, by + 20, bx + 20, by + 4], fill=0, width=3)
        y += h
    return img


def degrade(img, variant, rng):
    """clean = as rendered; scan = flatbed-style noise; phone = skew, shadow, JPEG artefacts."""
    if variant == "clean":
        return img
    nrng = np.random.default_rng(rng.randrange(2**32))
    if variant == "scan":
        angle, blur, noise, contrast = rng.uniform(-1.5, 1.5), rng.uniform(0.4, 0.9), 6, 0.85
    else:
        angle, blur, noise, contrast = rng.uniform(-3, 3), rng.uniform(0.8, 1.4), 8, 0.8
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, dtype=np.float32) * contrast + (255 * (1 - contrast))
    if variant == "phone":
        h, w = arr.shape
        shade = np.linspace(1.0, rng.uniform(0.62, 0.85), w)[None, :] * np.linspace(1.0, rng.uniform(0.8, 1.0), h)[:, None]
        arr = arr * shade
    arr = arr + nrng.normal(0, noise, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    if variant == "phone":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(40, 60))
        buf.seek(0)
        img = Image.open(buf).convert("L")
    return img


# --------------------------------------------------------------------------- #
# Validation, output, CLI
# --------------------------------------------------------------------------- #
def check_doc(doc):
    for kind, spans in doc.spans.items():
        for s in spans:
            if doc.text[s["start"]:s["end"]] != s["text"]:
                raise AssertionError(f"{doc.doc_type}: {kind} span offset mismatch for {s}")
    for t in doc.terms:
        if t not in doc.text:
            raise AssertionError(f"{doc.doc_type}: term {t!r} not present in text")


README = """# Synthetic medical documents

Generated by `generate_synthetic.py`. EVERYTHING here is synthetic: names, identifiers,
addresses and results are random. Do not add real patient documents to this folder.

Each document `doc_XXX` has:
- `doc_XXX.json`: ground truth
- `doc_XXX_clean.png`, `_scan.png`, `_phone.png`: the same content at three image-quality levels

## JSON fields
- `text`: ground-truth text. Compare to OCR output AFTER collapsing whitespace.
- `pii`: spans that SHOULD be redacted (`type`, `text`, `start`, `end`).
  Types: PERSON, DATE_OF_BIRTH, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS, ADDRESS, EIRCODE,
  PPS_NUMBER, MRN. Map to Presidio labels in your evaluation (DATE_OF_BIRTH -> DATE_TIME,
  ADDRESS -> LOCATION, ...).
- `keep`: clinically meaningful spans that must SURVIVE redaction and appear correctly in the
  simplified output: DOSE, FREQUENCY, DURATION, LAB_VALUE, REFERENCE_RANGE, INVESTIGATION,
  VITAL, QUANTITY, CLINICAL_DATE. (CLINICAL_DATE, e.g. last tetanus booster, is a design
  decision: you may decide to treat it differently.)
- `not_pii`: text that looks like PII but is not (medical eponyms such as "Crohn's",
  business phone numbers). Redacting these is a false positive.
- `terms`: jargon a good explanation should define.
- `checkboxes`: ground truth for tick-box tables (clinic_form only); not part of `text`.

## Known design choices
- Ground truth uses ASCII units (`umol/L`) to keep OCR comparison clean.
- Some first names contain fadas (e.g. Seán). Tesseract's English model often drops them,
  which is a realistic OCR/PII interaction worth reporting.
- Random identifiers may coincidentally match real ones in format; they are not real.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="number of documents (cycles through 4 types)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(README, encoding="utf-8")
    counts = {}
    for i in range(args.n):
        doc_type = DOC_TYPES[i % len(DOC_TYPES)]
        rng = random.Random(f"{args.seed}-{i}")
        doc, _ = BUILDERS[doc_type](rng)
        check_doc(doc)
        doc_id = f"doc_{i + 1:03d}"
        base = render(doc)
        images = {}
        for v in args.variants:
            name = f"{doc_id}_{v}.png"
            degrade(base, v, rng).save(out / name)
            images[v] = name
        record = {"id": doc_id, "doc_type": doc_type, "synthetic": True, "seed": args.seed,
                  "text": doc.text, "pii": doc.spans["pii"], "keep": doc.spans["keep"],
                  "not_pii": doc.spans["not_pii"], "terms": doc.terms,
                  "checkboxes": doc.checkboxes, "images": images}
        (out / f"{doc_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        for s in doc.spans["pii"]:
            counts[s["type"]] = counts.get(s["type"], 0) + 1
    print(f"Wrote {args.n} documents x {len(args.variants)} variants to {out}/")
    print("PII spans by type:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
