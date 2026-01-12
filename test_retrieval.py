# Test retrieval step
import os
import pathway as pw
from pathlib import Path

from chunking import build_chunks
from index_build import build_index, load_indexes
from retrieval import retrieve_for_backstory

DATA_DIR = "./data"
INDEX_DIR = "./indexes"

# Load test data (just first 5 claims for speed)
test_file = Path(DATA_DIR) / "test.csv"
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
    return Path(file_path).stem

novels_table = novels_table.with_columns(
    book_name=extract_book_name(pw.this._metadata),
    full_text=pw.this.data
)

# Create chunks
children_table = build_chunks(novels_table)

# Load index
print("Loading embedder...")
loaded = load_indexes(INDEX_DIR)
embedder = loaded["embedder"]
embedded_table = children_table.with_columns(
    vector=embedder(pw.this.text)
)
indexes = {"embedder": embedder, "table": embedded_table}

# Prepare claims
claims_table = test_df.select(
    story_id=pw.this.id,
    book_name=pw.this.book_name,
    claim=pw.this.content,
    character=pw.this.char
)

print("Testing retrieval...")
evidence_table = retrieve_for_backstory(
    claims_table=claims_table,
    indexes=indexes,
    top_k=3  # Just 3 for debug
)

# Write evidence
pw.io.csv.write(
    evidence_table.select(
        story_id=pw.this.story_id,
        book_name=pw.this.book_name,
        num_chunks=pw.apply(len, pw.this.evidence_chunks)
    ),
    "debug_evidence.csv"
)

print("Running...")
pw.run()
print("Check debug_evidence.csv")
