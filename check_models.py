"""List the models available to your Groq API key.

Reads GROQ_API_KEY from the environment / .env file (never paste the key into this file).

Usage:
    python check_models.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

MODELS_URL = "https://api.groq.com/openai/v1/models"


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: GROQ_API_KEY is not set. Add it to your .env file.")

    print("Fetching available models...\n")
    response = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    if response.status_code != 200:
        sys.exit(f"Error: {response.status_code}\n{response.text}")

    for model in response.json().get("data", []):
        print(model["id"])


if __name__ == "__main__":
    main()
