# Pipeline Optimization Summary

## Current Implementation: 1 LLM Call per Claim ✅

The pipeline is already optimized to make **ONE** LLM call per claim (not per chunk):

```python
# In llm_judge.py
@pw.udf
def llm_judge(claim: str, character: str, evidence_chunks: List[str]) -> dict:
    """
    Single call with ALL evidence chunks for this claim
    """
    # evidence_chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    
    prompt = build_prompt(claim, character, evidence_chunks)  # All chunks in one prompt
    response = call_gemini(prompt)  # ONE API call
    return parse_response(response)
```

## Rate Limiting Strategy

### Gemini Free Tier Limits:
- **15 requests per minute** (RPM)
- **1 million tokens per day** (TPD)  
- **1,500 requests per day** (RPD)

### Current Settings:
- **Base delay**: 5 seconds between requests (conservative, 12 RPM)
- **Exponential backoff**: 2x multiplier on retries
- **Max retries**: 5 attempts
- **Automatic retry** on 429 (rate limit) and 403 (quota) errors

### For 60 claims:
- **Minimum time**: 60 claims × 5 seconds = **5 minutes**
- **With retries**: Could take 10-15 minutes depending on rate limits

## Further Optimizations (if needed):

### Option 1: Switch to Paid Tier
Gemini Pro API paid tier allows:
- **1000 RPM** (vs 15 for free)
- Much higher daily quotas

### Option 2: Use Cheaper/Faster Model
- **gemini-flash**: Faster, cheaper, same quality for simple tasks
- Already using gemini-2.5-flash (good choice!)

### Option 3: Batch API (if available)
- Process multiple claims in parallel
- Would require Gemini Batch API support

### Option 4: Alternative LLMs
- **OpenAI GPT-3.5-turbo**: Higher rate limits, cheap
- **Anthropic Claude**: Good rate limits
- **Local models**: llama.cpp, Ollama (no rate limits!)

## Current Performance:
✅ **5x improvement** from calling LLM once per claim (not per chunk)
✅ **Automatic retry** with exponential backoff
✅ **Global rate limiting** prevents hitting API limits
✅ **Error handling** for timeouts, quota exceeded, etc.

The pipeline is well-optimized for free tier usage!
