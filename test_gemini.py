#!/usr/bin/env python3
"""Quick test of Gemini API"""
import requests
import json

GEMINI_API_KEY = "AIzaSyDRJJ2Ho8M1nitZeSj_82G6l5qvRKtL3u0"
GEMINI_MODEL = "gemini-1.5-flash"

def test_gemini():
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    
    data = {
        "contents": [{
            "parts": [{
                "text": "Say 'Hello, World!' in JSON format with a field 'message'."
            }]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 100,
        }
    }
    
    print("Testing Gemini API...")
    response = requests.post(url, headers=headers, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"\nExtracted text: {text}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    test_gemini()
