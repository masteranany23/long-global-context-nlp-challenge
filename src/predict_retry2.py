# src/predict_retry2.py
# Second retry pipeline for remaining 3 failed rows - outputs to results_retry2.csv

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
RESULTS_FILE = "results_retry2.csv"  # Output to separate file

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = "AIzaSyDRJJ2Ho8M1nitZeSj_82G6l5qvRKtL3u0"

MAX_EVIDENCE_CHUNKS = 5

# ============================
# LOAD TEST DATA
# ============================

test_file = Path(DATA_DIR) / "test_retry2.csv"  # Use second retry subset
if not test_file.exists():
    raise FileNotFoundError(f"{test_file} not found.")

# Define schema without using 'id' as primary key
class TestSchema(pw.Schema):
    id: int
    book_name: str
    char: str
    caption: str
    content: str

test_df = pw.io.csv.read(
    str(test_file),
    schema=TestSchema,
    mode="static"
)

# ============================
# STEP 1: Chunk novels
# ============================

novel_dir = "./data/novels"

novels_table = pw.io.fs.read(
    path=novel_dir,
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"
)

@pw.udf
def extract_book_name(metadata: pw.Json) -> str:
    meta_dict = metadata.as_dict()
    file_path = meta_dict.get("path", meta_dict.get("file_path", ""))
    return Path(file_path).stem.title()

novels_table = novels_table.with_columns(
    book_name=extract_book_name(pw.this._metadata),
    full_text=pw.this.data
)

children_table = build_chunks(novels_table)

# ============================
# STEP 2: Build / Load Index
# ============================

if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
    print("Building indexes...")
    indexes = build_index(children_table, openai_key=OPENAI_KEY, index_dir=INDEX_DIR)
else:
    print("Loading existing embedder and rebuilding index...")
    loaded = load_indexes(INDEX_DIR)
    embedder = loaded["embedder"]
    embedded_table = children_table.with_columns(
        vector=embedder(pw.this.text)
    )
    indexes = {"embedder": embedder, "table": embedded_table}

# ============================
# STEP 3: Prepare claims
# ============================

@pw.udf
def normalize_book_name(name: str) -> str:
    return name.title()

claims_table = test_df.select(
    original_id=pw.this.id,
    book_name=normalize_book_name(pw.this.book_name),
    claim=pw.this.content,
    character=pw.this.char
)

# ============================
# STEP 4: Retrieve evidence
# ============================

evidence_table = retrieve_for_backstory(
    claims_table=claims_table,
    indexes=indexes,
    top_k=MAX_EVIDENCE_CHUNKS
)

print("Retrieved evidence for claims")

# ============================
# STEP 5: LLM Judge
# ============================

judged_table = judge_claims_table(evidence_table)
print("Judged claims with LLM")

# ============================
# STEP 6: Aggregate
# ============================

aggregated_table = aggregate_claims(judged_table)
print("Aggregated predictions for claims")

# ============================
# STEP 7: Prepare final results
# ============================

final_results = aggregated_table.select(
    story_id=pw.this.original_id,
    prediction=pw.this.prediction,
    rationale=pw.this.rationale
)

pw.io.csv.write(final_results, RESULTS_FILE)

# Run the computation
print("\nRunning second retry pipeline for 3 remaining failed rows...")

try:
    pw.run()
    
    # Fix story IDs
    import csv
    
    test_ids = []
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(int(row['id']))
    
    results_rows = []
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_rows.append(row)
    
    if len(results_rows) != len(test_ids):
        print(f"⚠ Warning: Row count mismatch. Expected {len(test_ids)}, got {len(results_rows)}")
    
    for i, row in enumerate(results_rows):
        if i < len(test_ids):
            row['story_id'] = str(test_ids[i])
    
    with open(RESULTS_FILE, 'w', newline='') as f:
        if results_rows:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)
    
    print(f"\n✓ Second retry pipeline completed!")
    print(f"✓ Results saved to {RESULTS_FILE} ({len(test_ids)} predictions)")

except Exception as e:
    print(f"\n✗ Error during computation: {e}")
    raise
