# Test without LLM - just retrieval
import os
import pathway as pw
from pathlib import Path

from chunking import build_chunks
from index_build import load_indexes

DATA_DIR = "./data"
INDEX_DIR = "./indexes"

# Load small test data
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

# Load index
loaded = load_indexes(INDEX_DIR)
embedder = loaded["embedder"]
embedded_table = children_table.with_columns(
    vector=embedder(pw.this.text)
)

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

# Just write claims and chunks to verify data exists
pw.io.csv.write(
    claims_table.select(pw.this.story_id, pw.this.book_name),
    "test_claims_only.csv"
)

pw.io.csv.write(
    embedded_table.select(pw.this.book_name, text_len=pw.apply(len, pw.this.text)),
    "test_chunks_only.csv"
)

print("Running basic test...")
pw.run()
print("Done - check test_claims_only.csv and test_chunks_only.csv")
