# Import the Tesseract OCR library to extract text from images
import pytesseract
# Import the Image module from the Pillow (PIL) library to open image files
from PIL import Image
# Import the requests library to send HTTP requests to the Groq API over the internet
import requests

# --- SETUP ---
# Point Python to the exact location where the Tesseract OCR engine is installed on your Windows machine.
# Without this, pytesseract won't know how to execute the underlying OCR software.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Your personal Groq API key used to authenticate your request. 
# SECURITY WARNING: Never upload this key to a public GitHub repository.
API_KEY = "YOUR_API_KEY_HERE"
# The specific Groq endpoint URL designed to accept chat completion requests.
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- STEP 1: READ IMAGE & EXTRACT TEXT (OCR) ---
# Define the filename of the medical document image you want to process.
image_path = "sample_report.png"

try:
    # Open the image file from your local folder into memory.
    img = Image.open(image_path)
    
    # Run the Tesseract OCR engine on the image to extract all readable text into a string.
    # This keeps the text in temporary memory (satisfying the zero-retention requirement).
    extracted_text = pytesseract.image_to_string(img)
    
    # Print the raw text to the terminal so you can verify the OCR worked correctly.
    print("--- ORIGINAL MEDICAL TEXT ---")
    print(extracted_text)
    print("\n" + "="*50 + "\n")
    
except FileNotFoundError:
    # If the image isn't found in the VS Code folder, gracefully exit the script instead of crashing.
    print(f"Error: Please place a medical image named '{image_path}' in the VS Code folder.")
    exit()

# --- STEP 2: SECURE CLOUD PROCESSING (IN-MEMORY) ---
# Set up the HTTP headers. This acts as your secure ID badge, passing your API key to Groq.
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Construct the payload (the data) to send to the AI model.
data = {
    # Specify the open-weights model to use (Qwen 27B).
    "model": "qwen/qwen3.8-27b",
    "messages": [
        {
            # The 'system' role sets the strict rules for the AI. 
            # This enforces your 8th-grade reading level requirement and prevents medical diagnoses.
            "role": "system", 
            "content": "You are an information-accessibility tool. Translate the following complex medical text into clear, plain language at an 8th-grade reading level. Do not diagnose, prescribe treatment, or suggest alternative medication."
        },
        {
            # The 'user' role is where you pass the actual data you want processed.
            # We inject the OCR text we extracted in Step 1 here.
            "role": "user", 
            "content": f"Simplify this medical text: {extracted_text}"
        }
    ]
}

# --- STEP 3: GET & DISPLAY SIMPLIFIED OUTPUT ---
print("Sending to model for simplification...\n")

# Send the data securely to the Groq servers using a POST request.
response = requests.post(API_URL, json=data, headers=headers)

# Check if the server responded with a '200 OK' status (meaning success).
if response.status_code == 200:
    # Navigate through the returned JSON data to isolate just the AI's text response.
    simplified_text = response.json()["choices"][0]["message"]["content"]
    
    # Print the final plain-language translation to the terminal.
    print("--- PLAIN LANGUAGE EXPLANATION ---")
    print(simplified_text)
else:
    # If the request fails (e.g., a 401 or 404 error), print the error code and the server's explanation.
    print(f"API Error: {response.status_code}")
    print(response.text)