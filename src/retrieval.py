# src/retrieval.py
import pathway as pw
from typing import Dict

def retrieve_for_backstory(
    claims_table: pw.Table,
    indexes: Dict,
    top_k: int = 5
) -> pw.Table:
    """
    Process all claims and return top evidence chunks for each.
    Simplified version without vector similarity - just returns first N chunks per book.
    """
    
    embedder = indexes["embedder"]
    embedded_table = indexes["table"]
    
    # Join claims with chunks on book_name
    joined = claims_table.join(
        embedded_table,
        pw.left.book_name == pw.right.book_name
    ).select(
        story_id=pw.left.story_id,
        claim=pw.left.claim,
        character=pw.left.character,
        book_name=pw.left.book_name,
        chunk_text=pw.right.text
    )
    
    # Group by story_id and collect chunks
    evidence_agg = joined.groupby(pw.this.story_id).reduce(
        story_id=pw.this.story_id,
        claim=pw.reducers.any(pw.this.claim),
        character=pw.reducers.any(pw.this.character),
        book_name=pw.reducers.any(pw.this.book_name),
        all_chunks=pw.reducers.sorted_tuple(pw.this.chunk_text)
    )
    
    # Extract first k chunks
    @pw.udf
    def get_first_k_chunks(chunks_tuple: tuple, k: int) -> list:
        return list(chunks_tuple[:k])
    
    evidence_table = evidence_agg.with_columns(
        evidence_chunks=get_first_k_chunks(pw.this.all_chunks, top_k)
    ).select(
        story_id=pw.this.story_id,
        claim=pw.this.claim,
        character=pw.this.character,
        book_name=pw.this.book_name,
        evidence_chunks=pw.this.evidence_chunks
    )
    
    return evidence_table
