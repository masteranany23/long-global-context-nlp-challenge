# Knowledge-Driven Story Hallucination Detection

RAG-based pipeline for detecting inconsistencies in backstory claims using sentence transformers and LLM judging.

---

## 🚀 Quick Start for Judges

**To run the evaluation pipeline:**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your 3 Gemini API keys
   ```

3. **Run the pipeline:**
   ```bash
   python src/predict.py
   ```

4. **Check results:**
   ```bash
   cat results.csv
   ```

The pipeline will process 60 test claims and generate `results.csv` with predictions (~2 minutes runtime).

---

## Features

- **Local Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) for semantic search
- **Semantic Retrieval**: Top-k most relevant chunks via cosine similarity
- **LLM Judging**: Gemini API for claim consistency evaluation
- **Rate Limiting**: Multi-key rotation to handle API quotas
- **Pathway Framework**: Streaming data processing pipeline

## Requirements

- Python 3.10+
- 3 Gemini API keys (free tier: 15 RPM each = 45 RPM total)
  - Get free keys at: https://aistudio.google.com/app/apikey

---

## Detailed Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd KDSH
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your 3 Gemini API keys
   ```
   Get free API keys at: https://aistudio.google.com/app/apikey

5. **Run the pipeline**
   
   **This generates `results.csv` with predictions for all test claims.**

---   python src/predict.py
   ```

## Project Structure

```
KDSH/
├── data/
│   ├── test.csv              # Test claims with character backstories
│   ├── train.csv             # Training data
│   └── novels/               # Source novels as text files
├── src/
│   ├── predict.py            # Main pipeline orchestration
│   ├── chunking.py           # Document chunking logic
│   ├── index_build.py        # Embedding index management
│   ├── retrieval.py          # Semantic similarity retrieval
│   ├── llm_judge.py          # Gemini API integration
│   └── aggregation.py        # Result aggregation
├── indexes/                  # Generated embeddings (auto-created)
├── results.csv               # Output predictions
├── .env.example              # API key template
└── requirements.txt          # Python dependencies
```

## How It Works

1. **Chunking**: Splits novels into 180-300 token chunks with overlap
2. **Embedding**: Converts chunks to 384-dim vectors using sentence-transformers
3. **Retrieval**: For each claim, retrieves top-5 most similar chunks via cosine similarity
4. **Judging**: Gemini evaluates claim consistency against evidence
5. **Output**: Generates results.csv with predictions (0=contradict, 1=consistent)

## API Rate Limits

- **Free tier**: 15 requests/minute per key
- **With 3 keys**: 45 requests/minute (2-second intervals)
- **60 test claims**: ~2 minutes total runtime

**File:** `results.csv`

After running `python src/predict.py`, the output will be:

```csv
story_id,prediction,rationale
95,0,"The evidence does not mention Noirtier handing a dossier..."
136,1,"Evidence confirms Faria's scholarly work and isolation..."
```

- `story_id`: Original ID from test.csv
- `prediction`: 0 (contradict) or 1 (consistent)
- `rationale`: LLM's reasoning for the judgment

---```

## Notes

- First run builds embeddings (saved to `indexes/`)
- Subsequent runs reuse existing embeddings
- Delete `indexes/` to rebuild from scratch
- Rate limiting prevents 429 errors from Gemini API

## License

MIT
