# src/predict_small.py - Test with small dataset
import os
import pathway as pw
from pathlib import Path

from chunking import build_chunks
from index_build import build_index, load_indexes
from retrieval import retrieve_for_backstory
from llm_judge import judge_claims_table
from aggregation import aggregate_claims

DATA_DIR = "./data"
INDEX_DIR = "./indexes"
RESULTS_FILE = "results_small.csv"
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# Load SMALL test data
test_file = Path(DATA_DIR) / "test_small.csv"
schema = pw.schema_from_csv(str(test_file))
test_df = pw.io.csv.read(str(test_file), schema=schema, mode="static")

# Load novels
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

# Create chunks
children_table = build_chunks(novels_table)

# Load/build index
if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
    print("Building indexes...")
    indexes = build_index(children_table, openai_key=OPENAI_KEY, index_dir=INDEX_DIR)
else:
    print("Loading existing embedder...")
    loaded = load_indexes(INDEX_DIR)
    embedder = loaded["embedder"]
    embedded_table = children_table.with_columns(
        vector=embedder(pw.this.text)
    )
    indexes = {"embedder": embedder, "table": embedded_table}

# Prepare claims
@pw.udf
def normalize_book_name(name: str) -> str:
    return name.title()

claims_table = test_df.select(
    story_id=pw.this.id,
    book_name=normalize_book_name(pw.this.book_name),
    claim=pw.this.content,
    character=pw.this.char
)

print("Processing claims...")

# Retrieve evidence
evidence_table = retrieve_for_backstory(
    claims_table=claims_table,
    indexes=indexes,
    top_k=3  # Reduced for testing
)
print("Retrieved evidence")

# Debug: check evidence table
pw.io.csv.write(
    evidence_table.select(pw.this.story_id, pw.this.claim),
    "debug_evidence_output.csv"
)

# Judge with LLM
judged_table = judge_claims_table(evidence_table)
print("Judged claims")

# Debug: check judged table
pw.io.csv.write(judged_table, "debug_judged.csv")

# Aggregate
aggregated_table = aggregate_claims(judged_table)
print("Aggregated")

# Debug: check what's in aggregated table
pw.io.csv.write(aggregated_table, "debug_aggregated.csv")

# Final results
final_results = aggregated_table.select(
    story_id=pw.this.story_id,
    prediction=pw.this.prediction,
    rationale=pw.this.rationale
)

pw.io.csv.write(final_results, RESULTS_FILE)
pw.run()

print(f"Results saved to {RESULTS_FILE}")
