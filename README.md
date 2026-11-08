# Medical Information Simplifier

A privacy-first tool that turns hard-to-read medical documents into plain language at roughly an 8th-grade reading level. Personal data is removed **on your machine** before any text is sent to a language model.

> **Not medical advice.** The tool explains what a document says and defines its terms. It does not diagnose, recommend treatment or suggest alternative medication.

Final-year Software Engineering project by Sara Larkem. Status: **MVP**. The full design and roadmap are in [ARCH_SPEC.md](ARCH_SPEC.md).

## How it works

```
Document image
  -> [1] Pre-process + OCR        (Tesseract, local)
  -> [2] PII redaction            (Microsoft Presidio + custom Irish recognizers, local)
  -> [3] Plain-language rewrite   (LLM via the Groq API; only redacted text is sent)
```

Stages 1 and 2 run locally. Only stage 3 crosses the network, and it only receives text that has already been redacted. As well as Presidio's built-in entities (names, dates, phone numbers, locations), the redaction step has custom recognizers for **PPS numbers** and **Eircodes**.

## Repository layout

| File | Purpose |
|---|---|
| [mvp_pipeline.py](mvp_pipeline.py) | The end-to-end pipeline: OCR -> redaction -> LLM explanation |
| [generate_synthetic.py](generate_synthetic.py) | Generates synthetic medical documents (images + ground truth) for evaluation |
| [evaluate_pipeline.py](evaluate_pipeline.py) | Measures OCR accuracy and redaction quality against the synthetic ground truth |
| [evaluation_analysis-v2.ipynb](evaluation_analysis-v2.ipynb) | Charts the privacy-vs-utility trade-off from the evaluation results |
| [check_models.py](check_models.py) | Lists the models available to your Groq API key |
| [tests/](tests/) | Unit tests for the evaluation harness |
| [ARCH_SPEC.md](ARCH_SPEC.md) | Architecture specification, threat model and evaluation plan |

## Setup

**Requirements:** Python 3 (developed on 3.14) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

1. Install Tesseract. On Windows the default install path (`C:\Program Files\Tesseract-OCR\tesseract.exe`) is detected automatically. Anywhere else, set the `TESSERACT_CMD` environment variable to the binary's path.

2. Install the Python dependencies and the spaCy model that Presidio uses:

   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```

3. Create a `.env` file in the project root with your Groq API key:

   ```
   GROQ_API_KEY=your_key_here
   ```

   `.env` is git-ignored, so never put the key in a source file.

## Usage

### Run the pipeline

```
python mvp_pipeline.py
```

This reads `sample_report.png` and prints the extracted text, the redacted text that gets sent to the API, and the plain-language explanation. The model and prompt are constants at the top of `mvp_pipeline.py`. If you need to check which model names your key can use, run `python check_models.py`.

### Reproduce the evaluation

The evaluation runs entirely on **synthetic** documents: all names, identifiers and results are randomly generated, and no real patient data is used. Generated data (`data/`) and results (`results/`) are git-ignored, so create them locally:

```
# 1. Generate 40 synthetic documents (lab reports, discharge summaries,
#    prescription labels, clinic forms), each as clean / scan / phone images
python generate_synthetic.py --n 40 --seed 42 --out data/synthetic

# 2. OCR pre-processing ablation (CER / WER per config and image quality)
python evaluate_pipeline.py ocr --data data/synthetic

# 3. Redaction evaluation on ground-truth text
python evaluate_pipeline.py redact --data data/synthetic --detectors mvp regex_ids hybrid

# 4. Redaction end-to-end, on real OCR output from phone photos
python evaluate_pipeline.py redact --source ocr --ocr-config deskew_bgnorm --variant phone
```

OCR output is cached in `results/ocr_cache/`, so re-runs are fast. Then open `evaluation_analysis-v2.ipynb` to chart the results from `results/redaction_gt.csv`.

To evaluate your own redactor, write a function `detect(text) -> [(start, end, entity_type), ...]` and pass it as `--detectors my_module:detect`.

## Results so far

**Redaction**, 40 synthetic documents, ground-truth text (lower is better for the first two columns):

| Detector | PII leakage | Clinical text destroyed | Relaxed P / R / F1 |
|---|---|---|---|
| `mvp`: Presidio (PERSON, DATE_TIME, PHONE_NUMBER) | 34.9% | 29.6% | 0.63 / 0.65 / 0.64 |
| `regex_ids`: regexes for structured IDs only | 37.8% | 0.0% | 0.96 / 0.62 / 0.75 |
| `hybrid`: regexes for IDs/dates + Presidio for names/places | **4.6%** | **0.2%** | **0.97 / 0.95 / 0.96** |

- *Leakage* is the share of PII still readable after redaction: the privacy risk.
- *Clinical text destroyed* is the share of doses, durations, lab values and similar spans that were wrongly redacted: the utility loss.

Presidio's `DATE_TIME` entity is the main cause of utility loss, because it also removes durations such as "7 days".

The `mvp` row is the original MVP configuration. `mvp_pipeline.py` has since added PPS, Eircode and LOCATION recognizers, so its current configuration has not been evaluated exactly.

**OCR** (Tesseract `--psm 3`): on degraded phone photos, background normalisation + Otsu thresholding cuts the character error rate from **41.4%** (2x upscale only) to **4.6%**. On clean and scanned images every config stays below 5% CER.

## Tests

```
pytest
```

The tests cover the text metrics (CER/WER, Levenshtein), redaction scoring, the regex baseline against generated documents, skew estimation and the pre-processing configs. pytest only collects `tests/`, as configured in `pytest.ini`.

## Privacy notes

- Redaction lowers risk but is **not** a guarantee of anonymisation. OCR errors and unusual formats can let PII through, which is why leakage is measured rather than assumed.
- Do not add real patient documents to `data/synthetic/`.
- See the threat model in [ARCH_SPEC.md](ARCH_SPEC.md#6-privacy-and-threat-model-skeleton) for more detail.
