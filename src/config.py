# src/config.py
"""Configuration for the RAG pipeline"""

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyDRJJ2Ho8M1nitZeSj_82G6l5qvRKtL3u0"
GEMINI_MODEL = "gemini-pro"
MAX_TOKENS = 2000

# Rate Limiting Configuration
INITIAL_RATE_LIMIT = 2.0  # seconds between requests
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0

# Free tier limits: 15 requests per minute = 1 request every 4 seconds to be safe
MIN_REQUEST_INTERVAL = 4.0  # seconds

# Retrieval Configuration
MAX_EVIDENCE_CHUNKS = 5  # chunks per claim
TOP_K_RETRIEVAL = 5

# Paths
DATA_DIR = "./data"
INDEX_DIR = "./indexes"
NOVEL_DIR = "./data/novels"
RESULTS_FILE = "results.csv"
