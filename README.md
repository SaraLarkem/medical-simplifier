# Medical Information Simplifier

A privacy-first tool that turns hard-to-read medical documents into plain English that a 12–14 year old could follow. Personal data is removed **on your machine** before any text is sent to a language model.

> **Not medical advice.** The tool explains what a document says and defines its terms. It does not diagnose, recommend treatment or suggest alternative medication.

Final-year Software Engineering project by Sara Larkem. Status: **MVP**. The full design and roadmap are in [ARCH_SPEC.md](ARCH_SPEC.md).

## How it works

```
Document image
  -> [1] Pre-process + OCR        (Tesseract, local)
  -> [2] PII redaction            (Microsoft Presidio + custom Irish recognisers, local)
  -> [3] Term lookup              (term extraction + retrieval from MedlinePlus and a curated glossary, local, on redacted text)
  -> [4] Plain-language rewrite   (LLM via the Groq API; only redacted text and the retrieved definitions are sent)
```

Stages 1–3 run locally. Only stage 4 crosses the network, and it only receives text that has already been redacted, plus the definitions found in stage 3. As well as Presidio's built-in entities (names, dates, phone numbers, locations), the redaction step has custom recognisers for **PPS numbers** and **Eircodes**.

Stage 3 finds the medical terms in the redacted text and looks up trusted definitions for them. The model is asked to cite each definition it uses as `[S1]`, `[S2]`, ...; the pipeline then lists the sources and reports any citations that are invalid or missing. If the knowledge base has not been built, the pipeline prints a warning and carries on without definitions.

## Repository layout

| File | Purpose |
|---|---|
| [mvp_pipeline.py](mvp_pipeline.py) | The end-to-end pipeline: OCR -> redaction -> LLM explanation |
| [generate_synthetic.py](generate_synthetic.py) | Generates synthetic medical documents (images + ground truth) for evaluation |
| [evaluate_pipeline.py](evaluate_pipeline.py) | Measures OCR accuracy and redaction quality against the synthetic ground truth |
| [evaluation_analysis.ipynb](evaluation_analysis.ipynb) | Charts the privacy-vs-utility trade-off from the evaluation results |
| [rag/](rag/) | Local term extraction (`terms.py`), retrieval (`retriever.py`), prompt building (`prompt.py`), knowledge-base loading (`kb.py`) and the knowledge-base builder (`build_kb.py`) |
| [data/kb/curated_glossary.csv](data/kb/curated_glossary.csv) | Hand-written definitions, with sources, for abbreviations and terms MedlinePlus does not cover (e.g. OD, BD). The only tracked file under `data/` |
| [check_models.py](check_models.py) | Lists the models available to your Groq API key |
| [tests/](tests/) | Unit tests for the evaluation harness and the RAG module |
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

4. Build the local knowledge base. Download "MedlinePlus Health Topic XML" from <https://medlineplus.gov/xml.html> into `data/kb/raw/`, then run:

   ```
   python -m rag.build_kb
   ```

   This writes `data/kb/medlineplus.jsonl` from the newest `mplus_topics_YYYY-MM-DD.xml` in that folder. The built knowledge base is git-ignored; the curated glossary is already in the repository.

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

OCR output is cached in `results/ocr_cache/`, so re-runs are fast. Then open `evaluation_analysis.ipynb` to chart the results from `results/redaction_gt.csv`.

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

The `mvp` row is the original MVP configuration. `mvp_pipeline.py` has since added PPS, Eircode and LOCATION recognisers, so its current configuration has not been evaluated exactly.

**OCR** (Tesseract `--psm 3`): on degraded phone photos, background normalisation + Otsu thresholding cuts the character error rate from **41.4%** (2x upscale only) to **4.6%**. On clean and scanned images every config stays below 5% CER.

## Tests

```
pytest
```

The tests cover the text metrics (CER/WER, Levenshtein), redaction scoring, the regex baseline against generated documents, skew estimation and the pre-processing configs, as well as RAG term extraction, retrieval, prompt building and knowledge-base parsing (`tests/test_rag.py`). pytest only collects `tests/`, as configured in `pytest.ini`.

## Privacy notes

- Redaction lowers risk but is **not** a guarantee of anonymisation. OCR errors and unusual formats can let PII through, which is why leakage is measured rather than assumed.
- Do not add real patient documents to `data/synthetic/`.
- See the threat model in [ARCH_SPEC.md](ARCH_SPEC.md#6-privacy-and-threat-model-skeleton) for more detail.
