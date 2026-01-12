# Test with mock LLM (no API calls)
import os
import pathway as pw
from pathlib import Path
from typing import List

from chunking import build_chunks
from index_build import load_indexes
from retrieval import retrieve_for_backstory

DATA_DIR = "./data"
INDEX_DIR = "./indexes"

# Mock LLM judge
@pw.udf
def mock_llm_judge(claim: str, character: str, evidence_chunks: List[str]) -> dict:
    """Mock LLM that always returns consistent"""
    return {"prediction": 1, "rationale": "Mock: Always consistent"}

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
indexes = {"embedder": embedder, "table": embedded_table}

@pw.udf
def normalize_book_name(name: str) -> str:
    return name.title()

claims_table = test_df.select(
    story_id=pw.this.id,
    book_name=normalize_book_name(pw.this.book_name),
    claim=pw.this.content,
    character=pw.this.char
)

print("Retrieving evidence...")
evidence_table = retrieve_for_backstory(
    claims_table=claims_table,
    indexes=indexes,
    top_k=3
)

print("Mock judging...")
judged_table = evidence_table.select(
    story_id=pw.this.story_id,
    claim=pw.this.claim,
    character=pw.this.character,
    evidence_chunks=pw.this.evidence_chunks,
    result=mock_llm_judge(pw.this.claim, pw.this.character, pw.this.evidence_chunks)
).with_columns(
    prediction=pw.this.result["prediction"],
    rationale=pw.this.result["rationale"]
).select(
    story_id=pw.this.story_id,
    claim=pw.this.claim,
    prediction=pw.this.prediction,
    rationale=pw.this.rationale
)

pw.io.csv.write(judged_table, "test_mock_results.csv")

print("Running...")
pw.run()
print("Done - check test_mock_results.csv")
