"""
Project: Medical Information Simplifier (Final-Year Software Engineering Project)
Module: MVP Pipeline (OCR -> Local PII Redaction -> Secure API Inference)
Author: Sara Larkem
Description: Implements a privacy-first data pipeline that extracts text from complex 
             medical documents, sanitizes PII locally via Microsoft Presidio, and 
             queries an open-weight LLM via a secure, zero-retention API endpoint.
"""

import os
import pytesseract
from PIL import Image
import requests
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from dotenv import load_dotenv

# Load environment variables securely via python-dotenv
load_dotenv()

# --- SYSTEM CONFIGURATION ---
# Path to local Tesseract OCR binary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize local PII redaction engines (Microsoft Presidio)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

image_path = "sample_report.png"

# --- STEP 1: IMAGE PRE-PROCESSING & OCR EXTRACTION ---
try:
    img = Image.open(image_path)
    
    # Engineering Rationale: Upscale resolution by 2x and convert to 8-bit grayscale 
    # to optimize character edge detection for dense medical tables and forms.
    img = img.resize((img.width * 2, img.height * 2))
    img = img.convert('L')
    
    # Execute Tesseract OCR with fully automatic page segmentation (--psm 3)
    extracted_text = pytesseract.image_to_string(img, config=r'--oem 3 --psm 3')
    
    print("--- ORIGINAL EXTRACTED TEXT ---")
    print(extracted_text)
except FileNotFoundError:
    print(f"Error: Target image '{image_path}' not found in working directory.")
    exit()

# --- STEP 2: CLIENT-SIDE LOCAL PII REDACTION ---
print("\n--- EXECUTING LOCAL PII REDACTION (DATA MINIMIZATION) ---")
# Scan text locally for sensitive entities before any network transmission
analysis_results = analyzer.analyze(
    text=extracted_text, 
    entities=["PERSON", "DATE_TIME", "PHONE_NUMBER"], 
    language='en'
)

# Anonymize detected entities with structural placeholders
anonymized_result = anonymizer.anonymize(text=extracted_text, analyzer_results=analysis_results)
safe_text = anonymized_result.text
print(safe_text)

# --- STEP 3: SECURE CLOUD TRANSMISSION & INFERENCE ---
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Payload structured with strict anti-hallucination guardrails and formatting templates
payload = {
    "model": "qwen/qwen3.8-27b",
    "messages": [
        {
            "role": "system", 
            "content": """You are a strictly non-diagnostic information-accessibility tool. Translate complex medical text into plain language at an 8th-grade reading level. Do not diagnose, prescribe treatment, or suggest alternative medication. 

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
        },
        {
            "role": "user", 
            "content": f"Simplify this medical text: {safe_text}"
        }
    ]
}

print("\nExecuting secure API request (TLS 1.3 / Zero-Retention RAM)...")
response = requests.post(API_URL, json=payload, headers=headers)

if response.status_code == 200:
    simplified_text = response.json()["choices"][0]["message"]["content"]
    print("\n--- PLAIN LANGUAGE EXPLANATION ---")
    print(simplified_text)
else:
    print(f"API Error [{response.status_code}]: {response.text}")