"""
Project: Medical Information Simplifier (Final-Year Software Engineering Project)
Module: MVP Pipeline (OCR -> Local PII Redaction -> Local Retrieval -> Secure API Inference)
Author: Sara Larkem
Description: Implements a privacy-first data pipeline that extracts text from complex
             medical documents, sanitizes PII locally via Microsoft Presidio, looks up
             trusted definitions for the medical terms in a local knowledge base, and
             queries an open-weight LLM via a secure, zero-retention API endpoint.

Usage:
    python mvp_pipeline.py
"""

import os
import sys
import time

import pytesseract
import requests
from dotenv import load_dotenv
from PIL import Image
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine

from evaluate_pipeline import configure_tesseract
from rag.kb import load_kb
from rag.prompt import build_messages, check_citations, format_sources
from rag.retriever import retrieve
from rag.terms import extract_terms

# --- SYSTEM CONFIGURATION ---
IMAGE_PATH = "sample_report.png"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3.8-27b"
REQUEST_TIMEOUT = 60  # seconds

# Ground "Medical terms" definitions in a local knowledge base (MedlinePlus + curated
# glossary). Build it once with:  python -m rag.build_kb
USE_RAG = True
RAG_MAX_PASSAGES = 10

# Entities redacted locally before any text leaves the device
REDACT_ENTITIES = ["PERSON", "DATE_TIME", "PHONE_NUMBER", "PPS_NUMBER", "EIRCODE", "LOCATION"]

# Strict anti-hallucination guardrails and formatting template
SYSTEM_PROMPT = """You are a strictly non-diagnostic information-accessibility tool. Translate complex medical text into plain language at an 8th-grade reading level. Do not diagnose, prescribe treatment, or suggest alternative medication.

CRITICAL CONSTRAINT: Pay strict attention to explicit section headers (e.g., 'Immunization Record') and do not invent, assume, or substitute medical procedures, tests, or actions that are not explicitly documented in the input text.

Execute analysis step-by-step:
1. Extract difficult medical terminology.
2. Formulate simple definitions for these extracted terms.
3. Summarize the overall meaning.

Format your final response EXACTLY using these sections:
- What does this say?
- Medical terms
- What the results mean
- Important information
- Questions you may want to ask your doctor"""


# --- CUSTOM IRISH DATA RECOGNIZERS ---
def build_analyzer():
    """Presidio analyzer with custom recognizers for Irish identifiers."""
    # PPS Number: 7 numbers followed by 1 or 2 letters (e.g., 1234567A or 1234567AB)
    pps_pattern = Pattern(name="pps_regex", regex=r"\b\d{7}[A-Za-z]{1,2}\b", score=0.9)
    pps_recognizer = PatternRecognizer(supported_entity="PPS_NUMBER", patterns=[pps_pattern])

    # Eircode: 3 characters, optional space, 4 characters (e.g., A65 F4E2)
    eircode_pattern = Pattern(name="eircode_regex", regex=r"\b[A-Za-z]\d{2}\s?[A-Za-z0-9]{4}\b", score=0.9)
    eircode_recognizer = PatternRecognizer(supported_entity="EIRCODE", patterns=[eircode_pattern])

    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(pps_recognizer)
    analyzer.registry.add_recognizer(eircode_recognizer)
    return analyzer


# --- STEP 1: IMAGE PRE-PROCESSING & OCR EXTRACTION ---
def extract_text(image_path):
    """OCR the image with Tesseract after light pre-processing."""
    img = Image.open(image_path)

    # Engineering Rationale: Upscale resolution by 2x and convert to 8-bit grayscale
    # to optimize character edge detection for dense medical tables and forms.
    img = img.resize((img.width * 2, img.height * 2))
    img = img.convert("L")

    # Execute Tesseract OCR with fully automatic page segmentation (--psm 3)
    return pytesseract.image_to_string(img, config=r"--oem 3 --psm 3")


# --- STEP 2: CLIENT-SIDE LOCAL PII REDACTION ---
def redact(text, analyzer, anonymizer):
    """Scan text locally for sensitive entities and replace them with placeholders."""
    analysis_results = analyzer.analyze(text=text, entities=REDACT_ENTITIES, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=analysis_results).text


# --- STEP 2b/2c: LOCAL TERM EXTRACTION & RETRIEVAL ---
def load_knowledge_base():
    """Load the local KB, or return None (pipeline then runs without RAG)."""
    kb = load_kb()
    if len(kb) == 0:
        print("Warning: knowledge base not found; continuing WITHOUT trusted definitions. "
              "Build it with:  python -m rag.build_kb")
        return None
    return kb


def find_definitions(safe_text, kb):
    """Find medical terms and their trusted definitions.

    Runs entirely on this machine and only ever receives the REDACTED text.
    """
    terms = extract_terms(safe_text, kb)
    return retrieve(terms, kb, max_items=RAG_MAX_PASSAGES, text=safe_text)


def print_passages(passages):
    if not passages:
        print("No terms from the knowledge base were found in this text.")
    for i, p in enumerate(passages, 1):
        flags = (" [AMBIGUOUS]" if p.ambiguous else "") + ("" if p.verified else " [unverified]")
        print(f"[S{i}] {p.term} -> {p.title} ({p.origin}, {p.match_type}){flags}")
        print(f"     {p.text}")
        print(f"     {p.source} | {p.url}")


# --- STEP 3: SECURE CLOUD TRANSMISSION & INFERENCE ---
def call_llm(messages, api_key):
    """Send one chat request. Returns the answer, Groq's token usage and the latency.

    Raises RuntimeError on an API error, so batch evaluation can carry on.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "messages": messages}
    start = time.perf_counter()
    response = requests.post(API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    seconds = time.perf_counter() - start
    if response.status_code != 200:
        raise RuntimeError(f"API Error [{response.status_code}]: {response.text}")
    body = response.json()
    return {
        "content": body["choices"][0]["message"]["content"],
        "usage": body.get("usage", {}),
        "seconds": round(seconds, 2),
    }


def simplify(safe_text, api_key, passages=None):
    """Send redacted text (plus any trusted definitions) to the LLM; return the explanation."""
    return call_llm(build_messages(SYSTEM_PROMPT, safe_text, passages or []), api_key)["content"]


def main():
    # Load environment variables securely via python-dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: GROQ_API_KEY is not set. Add it to your .env file.")
    configure_tesseract()

    try:
        extracted_text = extract_text(IMAGE_PATH)
    except FileNotFoundError:
        sys.exit(f"Error: Target image '{IMAGE_PATH}' not found in working directory.")
    print("--- ORIGINAL EXTRACTED TEXT ---")
    print(extracted_text)

    print("\n--- EXECUTING LOCAL PII REDACTION (DATA MINIMIZATION) ---")
    safe_text = redact(extracted_text, build_analyzer(), AnonymizerEngine())
    print(safe_text)

    passages = []
    kb = load_knowledge_base() if USE_RAG else None
    if kb is not None:
        print("\n--- LOCAL TERM EXTRACTION & RETRIEVAL (redacted text only, no network) ---")
        passages = find_definitions(safe_text, kb)
        print_passages(passages)

    print("\nExecuting secure API request (TLS 1.3 / Zero-Retention RAM)...")
    try:
        simplified_text = simplify(safe_text, api_key, passages)
    except RuntimeError as err:
        sys.exit(str(err))
    print("\n--- PLAIN LANGUAGE EXPLANATION ---")
    print(simplified_text)

    if passages:
        print("\n" + format_sources(passages))
        c = check_citations(simplified_text, len(passages))
        fmt = lambda nums: ", ".join(f"[S{n}]" for n in nums) or "none"
        print(f"\nCitations: {len(c['cited'])}/{len(passages)} sources cited, "
              f"invalid: {fmt(c['invalid'])}, not cited: {fmt(c['uncited'])}")


if __name__ == "__main__":
    main()
