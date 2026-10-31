import os
import pytesseract
from PIL import Image
import requests
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()

# --- SETUP ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize Presidio for local PII redaction
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Pointing to your specific PNG file
image_path = "sample_report.png"

try:
    img = Image.open(image_path)
    
    # PRE-PROCESSING: Double the resolution and convert to grayscale for clearer OCR
    img = img.resize((img.width * 2, img.height * 2))
    img = img.convert('L')
    
    # Default fully automatic page segmentation
    custom_config = r'--oem 3 --psm 3'
    extracted_text = pytesseract.image_to_string(img, config=custom_config)
    
    print("--- ORIGINAL MEDICAL TEXT ---")
    print(extracted_text)
except FileNotFoundError:
    print(f"Error: Please place '{image_path}' in the VS Code folder.")
    exit()

# --- STEP 2: LOCAL REDACTION (PRE-TRANSMISSION) ---
print("\n--- REDACTING SENSITIVE INFO ---")
# Scan for names, dates, and phone numbers locally
results = analyzer.analyze(text=extracted_text, entities=["PERSON", "DATE_TIME", "PHONE_NUMBER"], language='en')

# Replace found PII with generic placeholders
anonymized_result = anonymizer.anonymize(text=extracted_text, analyzer_results=results)
safe_text = anonymized_result.text
print(safe_text)

# --- STEP 3: SECURE CLOUD PROCESSING ---
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "qwen/qwen3.8-27b",
    "messages": [
        {
            "role": "system", 
            "content": """You are a strictly non-diagnostic information-accessibility tool. Translate complex medical text into plain language at an 8th-grade reading level. Do not diagnose, prescribe treatment, or suggest alternative medication. Pay strict attention to section headers (e.g., 'Immunization Record') and do not invent or assume medical procedures, tests, or actions that are not explicitly stated in the text.

Analyze the text step-by-step:
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

print("\nSending secure text to model for simplification...\n")
response = requests.post(API_URL, json=data, headers=headers)

if response.status_code == 200:
    simplified_text = response.json()["choices"][0]["message"]["content"]
    print("--- PLAIN LANGUAGE EXPLANATION ---")
    print(simplified_text)
else:
    print(f"API Error: {response.status_code}")
    print(response.text)