# src/predict_debug.py - Debug version with intermediate outputs
import os
import pathway as pw
from pathlib import Path

from chunking import build_chunks
from index_build import build_index, load_indexes

DATA_DIR = "./data"
INDEX_DIR = "./indexes"

# Load test data
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

# Extract book_name
@pw.udf
def extract_book_name(metadata: pw.Json) -> str:
    meta_dict = metadata.as_dict()
    file_path = meta_dict.get("path", meta_dict.get("file_path", ""))
    return Path(file_path).stem

novels_table = novels_table.with_columns(
    book_name=extract_book_name(pw.this._metadata),
    full_text=pw.this.data
)

# Debug: Write novels
pw.io.csv.write(novels_table.select(pw.this.book_name), "debug_novels_list.csv")

# Create chunks
children_table = build_chunks(novels_table)

# Debug: Write chunks
pw.io.csv.write(
    children_table.select(
        book_name=pw.this.book_name,
        text_preview=pw.apply(lambda t: t[:100], pw.this.text)
    ),
    "debug_chunks.csv"
)

# Build/load index
if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
    print("Building indexes...")
    indexes = build_index(children_table, index_dir=INDEX_DIR)
else:
    print("Loading existing embedder and rebuilding index...")
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

# Debug: Write claims
pw.io.csv.write(
    claims_table.select(pw.this.story_id, pw.this.book_name, pw.this.character),
    "debug_claims.csv"
)

print("Debug outputs configured. Running...")
pw.run()
print("Check debug_novels_list.csv, debug_chunks.csv, and debug_claims.csv")
