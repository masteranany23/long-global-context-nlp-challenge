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
    os.environ.get("GEMINI_API_KEY_3"),
    os.environ.get("GEMINI_API_KEY_4")
]

# Filter out None values and validate
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key and key != "your_first_gemini_api_key_here" and key != "your_second_gemini_api_key_here" and key != "your_third_gemini_api_key_here" and key != "your_fourth_gemini_api_key_here"]

if not GEMINI_API_KEYS:
    raise ValueError("Set at least one GEMINI_API_KEY in your .env file")

print(f"✓ Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")

# Use correct Gemini model name for v1beta API
GEMINI_MODEL = "gemini-2.5-flash"

# Max tokens per request - increased to avoid truncation
MAX_TOKENS = 1536  # Enough for full prediction + detailed rationale

# Thread-safe request throttling
# With 4 keys @ 15 RPM each = 60 RPM = 1 second per request safe rate
_request_lock = threading.RLock()  # Reentrant lock to allow nested acquisitions
_last_request_time = 0
_min_interval = 1.0  # seconds between requests - faster with 4 keys
_current_key_index = 0
_processed_count = 0  # Track how many claims processed
_error_count = 0  # Track how many failed

# Retry configuration
MAX_RETRIES = 0  # No retries - fail immediately to save time when quota exhausted
RETRY_DELAY = 1.0  # Short delay if needed

# ============================
# GOOGLE GEMINI CALL FUNCTION
# ============================

def call_gemini(prompt: str, retry_count: int = 0) -> str:
    """
    Call Google Gemini API to generate text.
    Rotates between multiple API keys to distribute load.
    Uses thread-safe throttling to prevent rate limit errors.
    Returns the raw text response or raises RuntimeError after MAX_RETRIES.
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
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP {e.response.status_code}"
        
        # Don't raise - return error marker that UDF will detect
        if e.response.status_code == 429:
            return "__API_ERROR__:Rate limit exceeded (429) - daily quota reached"
        
        # Only retry on server errors
        if retry_count < MAX_RETRIES and e.response.status_code in [500, 502, 503, 504]:
            print(f"⚠️  {error_msg}. Retry {retry_count + 1}/{MAX_RETRIES} in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            return call_gemini(prompt, retry_count + 1)
        
        # Don't raise - return error marker
        return f"__API_ERROR__:Gemini API error: {error_msg}"
    
    except requests.exceptions.Timeout:
        if retry_count < MAX_RETRIES:
            print(f"⚠️  Timeout. Retry {retry_count + 1}/{MAX_RETRIES} in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            return call_gemini(prompt, retry_count + 1)
        return "__API_ERROR__:Gemini API timeout after retries"
    
    except Exception as e:
        # Don't raise - return error marker
        return f"__API_ERROR__:Gemini API error: {str(e)}"

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
    NEVER raises exceptions - always returns a valid dict.
    """
    try:
        if not evidence_chunks:
            return {
                "prediction": "0",
                "rationale": "No evidence found"
            }

        prompt_text = build_prompt(claim, character, evidence_chunks)
        
        # Call Gemini - returns error marker string if it fails
        response_text = call_gemini(prompt_text)
        
        # Check if response is an error marker
        if response_text.startswith("__API_ERROR__:"):
            error_msg = response_text.replace("__API_ERROR__:", "")
            return {
                "prediction": "0",
                "rationale": f"API Error: {error_msg}"
            }

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
        # Catch-all for any unexpected errors during JSON parsing
        with _request_lock:
            _error_count += 1
        print(f"❌ [{_processed_count}] JSON parse error: {str(e)[:80]}")
        return {
            "prediction": "0",
            "rationale": f"JSON parse error: {str(e)[:100]}"
        }


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
