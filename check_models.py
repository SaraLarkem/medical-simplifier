import requests

# Paste your actual Groq API key here
API_KEY = "YOUR_API_KEY_HERE"
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

print("Fetching available models...\n")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = response.json().get("data", [])
    for model in models:
        print(model["id"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)