#!/usr/bin/env python3
"""
Simplified predict.py for debugging
"""
import os
import pathway as pw
from pathlib import Path

# Import modules
from src.chunking import build_chunks
from src.index_build import build_index

# Config
DATA_DIR = "./data"
INDEX_DIR = "./indexes"

# Load novels
novel_dir = "./data/novels"

novels_table = pw.io.fs.read(
    path=novel_dir,
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"
)

print("Loaded novels table")

# Extract book name from metadata
@pw.udf
def extract_book_name(metadata: pw.Json) -> str:
    meta_dict = metadata.as_dict()
    file_path = meta_dict.get("path", meta_dict.get("file_path", "unknown"))
    return Path(file_path).stem

novels_table = novels_table.with_columns(
    book_name=extract_book_name(pw.this._metadata),
    full_text=pw.this.data
)

print("Added book_name column")

# Create chunks
chunks_table = build_chunks(novels_table)
print("Created chunks")

# Debug: write chunks to CSV to see what we have
pw.io.csv.write(chunks_table.select(
    book_name=pw.this.book_name,
    chapter_id=pw.this.chapter_id,
    text_preview=pw.this.text[:100]  # first 100 chars
), "debug_chunks.csv")

print("About to run...")
pw.run()
print("Chunks written to debug_chunks.csv")
