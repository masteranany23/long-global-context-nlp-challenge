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
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3")
]

# Filter out None values and validate
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key and key != "your_first_gemini_api_key_here" and key != "your_second_gemini_api_key_here" and key != "your_third_gemini_api_key_here"]

if not GEMINI_API_KEYS:
    raise ValueError("Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, and GEMINI_API_KEY_3 in your .env file")

print(f"✓ Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")

# Use correct Gemini model name for v1beta API
GEMINI_MODEL = "gemini-2.5-flash"

# Max tokens per request - increased to avoid truncation
MAX_TOKENS = 1536  # Enough for full prediction + detailed rationale

# Thread-safe request throttling
# With 2 keys @ 15 RPM each = 30 RPM = 2 seconds per request safe rate
_request_lock = threading.Lock()
_last_request_time = 0
_min_interval = 2.0  # seconds between requests
_current_key_index = 0

# ============================
# GOOGLE GEMINI CALL FUNCTION
# ============================

def call_gemini(prompt: str) -> str:
    """
    Call Google Gemini API to generate text.
    Rotates between multiple API keys to distribute load.
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
  "prediction": "consistent",  # or "contradict"
  "rationale": "Brief 1-2 sentence explanation."
}
"""
    return prompt

# ============================
# PATHWAY UDF FOR JUDGING
# ============================

def judge_claim_with_evidence(
    claim: str,
    character: str,
    evidence_chunks: List[str]
) -> dict:
    """
    UDF that calls Gemini to judge claim + evidence.
    Returns dict with { "prediction", "rationale" }.
    """
    try:
        if not evidence_chunks:
            return {
                "prediction": "0",
                "rationale": "No evidence found"
            }

        prompt_text = build_prompt(claim, character, evidence_chunks)
        response_text = call_gemini(prompt_text)

        # Parse JSON from LLM response - strip markdown code blocks if present
        try:
            # Clean up the response text
            cleaned_text = response_text.strip()
            
            # Remove markdown code fences if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]  # Remove ```
            
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]  # Remove trailing ```
            
            cleaned_text = cleaned_text.strip()
            
            # Try to parse the JSON
            try:
                parsed = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # If JSON is incomplete/truncated, try to fix it
                # Look for the prediction field even in incomplete JSON
                import re
                
                # Try to extract prediction using regex
                pred_match = re.search(r'"prediction"\s*:\s*"(\w+)"', cleaned_text)
                rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)', cleaned_text)
                
                if pred_match:
                    prediction_raw = pred_match.group(1).lower()
                    prediction = "1" if "consistent" in prediction_raw else "0"
                    
                    # Get partial rationale if available
                    if rat_match:
                        rationale = rat_match.group(1) + "... (truncated)"
                    else:
                        rationale = "LLM response was incomplete"
                    
                    return {
                        "prediction": prediction,
                        "rationale": rationale
                    }
                else:
                    # Complete failure
                    raise json.JSONDecodeError("Cannot extract prediction", cleaned_text, 0)
            
            # Successfully parsed JSON
            prediction_raw = parsed.get("prediction", "unknown").lower()
            # Map "consistent" -> 1, "contradict" -> 0
            if "consistent" in prediction_raw:
                prediction = "1"
            else:
                prediction = "0"

            rationale = parsed.get("rationale", "No explanation provided")
            
        except (json.JSONDecodeError, Exception) as e:
            prediction = "0"
            rationale = f"Failed to parse LLM response: {response_text[:200]}"

        return {
            "prediction": prediction,
            "rationale": rationale
        }

    except Exception as e:
        return {
            "prediction": "0",
            "rationale": f"LLM error: {str(e)}"
        }


# ============================
# TABLE-LEVEL PROCESSING
# ============================

def judge_claims_table(evidence_table: pw.Table) -> pw.Table:
    """
    Process the evidence table and judge each claim using LLM.
    
    Args:
        evidence_table: Table with columns original_id, claim, character, evidence_chunks
        
    Returns:
        Table with columns original_id, prediction, rationale
    """
    # Apply UDF to each row
    judged = evidence_table.select(
        original_id=pw.this.original_id,
        result=pw.apply(
            judge_claim_with_evidence,
            pw.this.claim,
            pw.this.character,
            pw.this.evidence_chunks
        )
    )
    
    # Extract prediction and rationale from JSON result
    final = judged.select(
        original_id=pw.this.original_id,
        prediction=pw.this.result["prediction"],
        rationale=pw.this.result["rationale"]
    )
    
    return final
