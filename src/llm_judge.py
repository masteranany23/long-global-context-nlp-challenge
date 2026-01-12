# src/llm_judge.py
import os
import json
import time
import threading
import pathway as pw
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================
# CONFIGURATION
# ============================

# Load multiple Gemini API keys for load distribution
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1"),
    os.environ.get("GEMINI_API_KEY_2")
]

# Filter out None values and validate
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key and key != "your_first_gemini_api_key_here" and key != "your_second_gemini_api_key_here"]

if not GEMINI_API_KEYS:
    raise ValueError("Set GEMINI_API_KEY_1 and GEMINI_API_KEY_2 in your .env file")

print(f"✓ Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")

# Use correct Gemini model name for v1beta API
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro" if available

# Max tokens per request for free-tier (safe margin)
MAX_TOKENS = 2000

# Thread-safe request throttling
# With 2 keys @ 15 RPM each = 30 RPM = 2 seconds per request safe rate
_request_lock = threading.Lock()
_last_request_time = 0
_min_interval = 2.0  # seconds between requests
_current_key_index = 0

# ============================
# GOOGLE GEMINI CALL FUNCTION
# ============================
Uses thread-safe throttling to prevent rate limit errors.
    Returns the raw text response.
    """
    import requests
    global _current_key_index, _last_request_time
    
    # Thread-safe request throttling
    with _request_lock:
        # Enforce minimum interval between ANY requests
        elapsed = time.time() - _last_request_time
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        
        # Select API key with round-robin rotation
        api_key = GEMINI_API_KEYS[_current_key_index % len(GEMINI_API_KEYS)]
        _current_key_index += 1
        
        _last_request_time = time.time()
    global _current_key_index
    
    # Select API key with round-robin rotation
    api_key = GEMINI_API_KEYS[_current_key_index % len(GEMINI_API_KEYS)]
    _current_key_index += 1

    # Use v1beta API with gemini-2.5-flash model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": MAX_TOKENS,
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return text.strip()
        
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

# ============================
# PROMPT ENGINEERING
# ============================

def build_prompt(claim: str, character: str, evidence_chunks: List[str]) -> str:
    """
    Build a strong reasoning prompt for consistency check.
    """
    prompt = f"""
You are a literary analyst. Determine if the following backstory claim
is consistent with the provided evidence chunks from a novel.

Backstory Claim: "{claim}"
Character: "{character}"

Evidence Chunks:
"""
    for i, chunk in enumerate(evidence_chunks, 1):
        prompt += f"{i}. {chunk}\n"

    prompt += """

Instructions:
- Only answer "consistent" or "contradict".
- Optionally, provide a short rationale (1-2 sentences).
- Focus on logical and causal consistency with the novel's events.
- Ignore minor language differences or style; focus on narrative constraints.
- Do not hallucinate new events; only reason from the evidence provided.

Answer in JSON format like:
{
    "prediction": 1,   # 1=consistent, 0=contradict
    "rationale": "Short reasoning here."
}
"""
    return prompt

# ============================
# MAIN LLM JUDGE FUNCTION
# ============================

@pw.udf
def llm_judge(claim: str, character: str, evidence_chunks: List[str]) -> dict:
    """
    Pathway UDF: returns prediction + rationale for a single claim.
    """
    if not evidence_chunks:
        # fallback: no evidence, mark contradict by default
        return {"prediction": 0, "rationale": "No evidence available."}

    # Limit number of evidence chunks to stay under token budget
    MAX_CHUNKS = 5
    evidence_chunks = evidence_chunks[:MAX_CHUNKS]

    prompt = build_prompt(claim, character, evidence_chunks)

    try:
        response_text = call_gemini(prompt)
        # Try to parse JSON response
        # Sometimes the response might have markdown code blocks
        if "```json" in response_text:
            # Extract JSON from markdown code block
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            # Generic code block
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        result = json.loads(response_text)
        if "prediction" in result and result["prediction"] in [0, 1]:
            return result
        else:
            # fallback parsing
            pred = 1 if "consistent" in response_text.lower() else 0
            return {"prediction": pred, "rationale": response_text[:200]}
    except Exception as e:
        return {"prediction": 0, "rationale": f"LLM error: {str(e)}"}

# ============================
# HELPER TO PROCESS TABLE
# ============================

def judge_claims_table(claims_table: pw.Table) -> pw.Table:
    """
    Input Pathway table with columns: story_id, claim, book_name, character, evidence_chunks
    Output table with: story_id, claim, prediction, rationale
    """
    judged_table = claims_table.select(
        story_id=pw.this.story_id,
        claim=pw.this.claim,
        character=pw.this.character,
        evidence_chunks=pw.this.evidence_chunks,
        result=llm_judge(pw.this.claim, pw.this.character, pw.this.evidence_chunks)
    ).with_columns(
        prediction=pw.this.result["prediction"],
        rationale=pw.this.result["rationale"]
    )

    return judged_table.select(
        story_id=pw.this.story_id,
        claim=pw.this.claim,
        prediction=pw.this.prediction,
        rationale=pw.this.rationale
    )
