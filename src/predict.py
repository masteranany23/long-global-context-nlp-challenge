# src/predict.py
import os
import pathway as pw
from pathlib import Path

# Import your modules
from chunking import build_chunks
from index_build import build_index, load_indexes
from retrieval import retrieve_for_backstory
from llm_judge import judge_claims_table
from aggregation import aggregate_claims

# ============================
# CONFIG
# ============================

DATA_DIR = "./data"
INDEX_DIR = "./indexes"
RESULTS_FILE = "results.csv"

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")  # optional if needed for embedding
GEMINI_KEY = "AIzaSyDRJJ2Ho8M1nitZeSj_82G6l5qvRKtL3u0"  # os.environ.get("GEMINI_API_KEY")  # required for llm_judge

MAX_EVIDENCE_CHUNKS = 5  # limit per claim for LLM prompt

# ============================
# LOAD TEST DATA
# ============================

test_file = Path(DATA_DIR) / "test.csv"
if not test_file.exists():
    raise FileNotFoundError(f"{test_file} not found. Make sure test.csv is in ./data/")

schema = pw.schema_from_csv(str(test_file))
test_df = pw.io.csv.read(str(test_file), schema=schema, mode="static")


# ============================
# STEP 1: Chunk novels
# ============================


novel_dir = "./data/novels"

# Load all novels (each file becomes a row)
novels_table = pw.io.fs.read(
    path=novel_dir,
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"
)

# UDF to extract book_name from metadata
@pw.udf
def extract_book_name(metadata: pw.Json) -> str:
    # metadata is a Pathway Json object, need to convert to dict
    meta_dict = metadata.as_dict()
    file_path = meta_dict.get("path", meta_dict.get("file_path", ""))
    # Normalize to title case for consistent matching
    return Path(file_path).stem.title()

# Normalize columns for chunking
novels_table = novels_table.with_columns(
    book_name=extract_book_name(pw.this._metadata),
    full_text=pw.this.data
)

# Create chunks
children_table = build_chunks(novels_table)

# ============================
# STEP 2: Build / Load Index
# ============================

# Build or load index per book
if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
    print("Building indexes...")
    indexes = build_index(children_table, openai_key=OPENAI_KEY, index_dir=INDEX_DIR)
else:
    print("Loading existing embedder and rebuilding index...")
    loaded = load_indexes(INDEX_DIR)
    # Still need to rebuild the table with the loaded embedder
    embedder = loaded["embedder"]
    embedded_table = children_table.with_columns(
        vector=embedder(pw.this.text)
    )
    indexes = {"embedder": embedder, "table": embedded_table}

# ============================
# STEP 3: Prepare claims from test.csv
# ============================

# Test CSV should have: id, book_name, char, caption, content
# Normalize book_name for case-insensitive matching
@pw.udf
def normalize_book_name(name: str) -> str:
    return name.title()

# Store original ID as a regular data column (not as Pathway's internal ID)
claims_table = test_df.select(
    original_id=pw.this.id,  # Keep the original integer ID as a data column
    book_name=normalize_book_name(pw.this.book_name),
    claim=pw.this.content,
    character=pw.this.char
)

# ============================
# STEP 4: Retrieve evidence chunks for each claim
# ============================

evidence_table = retrieve_for_backstory(
    claims_table=claims_table,
    indexes=indexes,
    top_k=MAX_EVIDENCE_CHUNKS
)

print("Retrieved evidence for claims")

# ============================
# STEP 5: LLM Judge per claim + chunk
# ============================

judged_table = judge_claims_table(evidence_table)
print("Judged claims with LLM")

# ============================
# STEP 6: Aggregate chunk-level predictions per claim
# ============================

aggregated_table = aggregate_claims(judged_table)
print("Aggregated predictions for claims")

# ============================
# STEP 7: Prepare final results.csv
# ============================

# Map original_id back to story_id for output
final_results = aggregated_table.select(
    story_id=pw.this.original_id,
    prediction=pw.this.prediction,
    rationale=pw.this.rationale
)

# Save to CSV - set up the output connector
pw.io.csv.write(final_results, RESULTS_FILE)

# Run the computation
pw.run()

print(f"Results saved to {RESULTS_FILE}")
