# Architecture Specification: Medical Information Simplifier

**Status:** DRAFT v0.1. Items marked **[CONFIRM]** are decisions still to make.
**Author:** Sara Larkem

---

## 1. Purpose and scope

A privacy-first tool that turns hard-to-read medical documents (reports, leaflets, forms) into plain language at roughly an 8th-grade reading level, while preserving the original medical meaning.

**In scope:** explaining what a document says, defining terms, suggesting questions to ask a clinician.
**Out of scope (by design):** diagnosis, treatment advice, suggesting alternative medication. The tool says "your report says X, here is what that term means", never "you have X, so do Y".

## 2. Research question and contributions

> How much privacy can a layered pipeline (local OCR, local PII redaction, local or cloud inference) provide for medical-document simplification, and what does each layer cost in accuracy and safety?

Intended contributions:
1. A pipeline with a **measured** PII-redaction layer (per-entity precision/recall, including the effect of OCR noise).
2. A **comparison of three configurations**: (A) cloud, no redaction (synthetic data only, baseline); (B) cloud with local redaction; (C) fully local model.
3. A **safety validation layer** that checks outputs against the source (numbers, doses, units, banned diagnostic phrasing).

## 3. Pipeline overview

```
Input (image / PDF / pasted text)
   -> [1] Ingest + pre-process
   -> [2] OCR
   -> [3] PII redaction (local, reversible placeholders)
   -> [4] Inference (pluggable backend: local Ollama | cloud Groq)
   -> [5] Validate (faithfulness + safety checks)
   -> [6] Restore placeholders locally + present
        (original vs simplified view, term definitions, sources, disclaimer)
```

Trust boundary: everything up to and including stage [3], and stages [5]-[6], runs on the user's machine. Only stage [4] may cross the boundary, and only redacted text crosses it.

## 4. Components

| Stage | Current MVP | Planned upgrade | How it is evaluated |
|---|---|---|---|
| Pre-process + OCR | Tesseract, 2x upscale, grayscale, `--psm 3` | Deskew, adaptive threshold, table/checkbox handling, native PDF text layer, compare against a second OCR engine | Character / word error rate vs ground truth |
| PII redaction | Presidio: PERSON, DATE_TIME, PHONE_NUMBER | Custom recognizers (PPS no., Eircode, MRN, email, address), context-aware date handling, medical-eponym allow-list, reversible placeholders | Per-entity precision / recall / F1; leakage rate on OCR output |
| Inference | Groq API, single prompt | `LLMBackend` interface with Groq and Ollama implementations; versioned prompts; structured JSON output | Quality and latency per backend |
| Validation | None | Number/dose/unit preservation, section completeness, banned-phrase checks, glossary lookups (MedlinePlus / NIH) | Hallucination rate; regression tests |
| UI | CLI script | Local web UI served on localhost | Usability (user study) |

## 5. Deployment decision

**[CONFIRM] Desktop / localhost first.**
Rationale: the privacy claim ("data never leaves the device") only holds where OCR, redaction and a local model can all run, which is a laptop. The existing stack is Python. Mobile would force either a rewrite or server-side redaction, which weakens the privacy claim.
Mobile is documented as future work; the core exposes a clean local API so a phone client could be added later.

## 6. Privacy and threat model (skeleton)

**Assets:** the document image/text, extracted PII, the plain-language output, any API key.

| Threat | Mitigation | Residual risk |
|---|---|---|
| PII missed by redaction (e.g. OCR errors) | Custom recognizers; measure leakage; second-detector check | Quasi-identifiers, misses on noisy scans |
| Interception in transit | Enforce minimum TLS version | Provider-side trust |
| Provider retains data | Use provider's zero-retention option; **cite their current terms** | Depends on provider policy |
| Data written to disk | Do not persist by default; if history is added, encrypt locally (AES-256-GCM) | Device compromise |
| Prompt injection via document text | Delimit untrusted text, validate output | Novel attacks |
| API key leakage | `.env` kept out of git; rotate if exposed | User error |
| Hallucinated / unsafe medical content | Validators, non-diagnostic prompt, disclaimer | Subtle errors the checks miss |

Regulatory framing: GDPR first (Art. 5(1)(c) data minimisation, Art. 9 special-category health data, Art. 25 privacy by design). HIPAA only as a comparison. Redaction reduces risk; it is **not** a legal guarantee of anonymisation.

## 7. Evaluation plan

| Area | Method |
|---|---|
| Test data | 30-50 **synthetic** documents (lab report, discharge note, leaflet, form) with known fake PII. No real patient data. |
| OCR | CER / WER across pre-processing ablations |
| Redaction | Per-entity P/R/F1; leakage rate end to end |
| Readability | Flesch-Kincaid / SMOG, original vs simplified |
| Faithfulness | Automated entity/number preservation + manual rubric on a subset |
| Benchmark | SARI / BERTScore on a GEM Cochrane subset (sanity check only) |
| Privacy proof | Traffic capture showing no PII in outgoing payloads |
| User study | 5-10 participants, comprehension questions on original vs simplified (**[CONFIRM]** ethics requirements with supervisor) |
| Performance | Per-stage and end-to-end latency, local vs cloud |

## 8. Proposed repo structure

```
src/
  ingest/      # pre-processing + OCR
  redact/      # Presidio config, custom recognizers, placeholder vault
  inference/   # LLMBackend, groq.py, ollama.py
  validate/    # safety + faithfulness checks
  app/         # local web UI / API
prompts/       # versioned prompts + experiment log
data/synthetic/
eval/          # evaluation scripts + results
tests/
docs/          # threat model, diagrams
```

## 9. Open decisions

- [ ] Desktop-first confirmed? (section 5)
- [ ] Default backend: local or cloud?
- [ ] Local model choice for Ollama (size vs laptop hardware)
- [ ] Ethics approval route for the user study
- [ ] Deadline and marking criteria: map sections 2 and 7 to the rubric