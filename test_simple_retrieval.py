# Simplest possible retrieval test
import os
import pathway as pw
from pathlib import Path
from typing import List

from chunking import build_chunks
from index_build import load_indexes

DATA_DIR = "./data"
INDEX_DIR = "./indexes"

# Load small test
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

children_table = build_chunks(novels_table)

loaded = load_indexes(INDEX_DIR)
embedder = loaded["embedder"]
embedded_table = children_table.with_columns(
    vector=embedder(pw.this.text)
)

@pw.udf
def normalize_book_name(name: str) -> str:
    return name.title()

claims_table = test_df.select(
    story_id=pw.this.id,
    book_name=normalize_book_name(pw.this.book_name),
    claim=pw.this.content,
    character=pw.this.char
)

# Simple retrieval - just join and group, no vectors
joined = claims_table.join(
    embedded_table,
    pw.left.book_name == pw.right.book_name
).select(
    story_id=pw.left.story_id,
    claim=pw.left.claim,
    chunk_text=pw.right.text
)

# Group by story_id and collect first 3 chunks
evidence = joined.groupby(pw.this.story_id).reduce(
    story_id=pw.this.story_id,
    claim=pw.reducers.any(pw.this.claim),
    chunks=pw.reducers.sorted_tuple(pw.this.chunk_text)
)

# Extract first 3 chunks
@pw.udf
def get_first_n(chunks_tuple: tuple, n: int) -> list:
    return list(chunks_tuple[:n])

evidence_final = evidence.with_columns(
    evidence_chunks=get_first_n(pw.this.chunks, 3)
).select(
    story_id=pw.this.story_id,
    claim=pw.this.claim,
    num_chunks=pw.apply(len, pw.this.evidence_chunks)
)

pw.io.csv.write(evidence_final, "test_simple_retrieval.csv")

print("Running simple retrieval...")
pw.run()
print("Done - check test_simple_retrieval.csv")
