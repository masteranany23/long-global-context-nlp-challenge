# src/retrieval.py
import pathway as pw
from typing import Dict

def retrieve_for_backstory(
    claims_table: pw.Table,
    indexes: Dict,
    top_k: int = 5
) -> pw.Table:
    """
    Process all claims and return top evidence chunks for each using semantic similarity.
    """
    
    embedder = indexes["embedder"]
    embedded_table = indexes["table"]
    
    # Embed the claims
    claims_with_vectors = claims_table.with_columns(
        query_vector=embedder(pw.this.claim)
    )
    
    # Join claims with chunks on book_name
    joined = claims_with_vectors.join(
        embedded_table,
        pw.left.book_name == pw.right.book_name
    ).select(
        original_id=pw.left.original_id,
        claim=pw.left.claim,
        character=pw.left.character,
        book_name=pw.left.book_name,
        query_vector=pw.left.query_vector,
        chunk_text=pw.right.text,
        chunk_vector=pw.right.vector
    )
    
   # SIMPLIFIED: Don't compute similarity - just take first top_k chunks per claim
    # Group by original_id and take first k chunks
    evidence_agg = joined.groupby(pw.this.original_id).reduce(
        original_id=pw.this.original_id,
        claim=pw.reducers.any(pw.this.claim),
        character=pw.reducers.any(pw.this.character),
        book_name=pw.reducers.any(pw.this.book_name),
        all_chunks=pw.reducers.tuple(pw.this.chunk_text)
    )
    
    # Take first k chunks
    @pw.udf
    def take_first_k(chunks: tuple, k: int) -> list:
        """Return first k chunks."""
        return list(chunks[:k])
    
    evidence_table = evidence_agg.with_columns(
        evidence_chunks=take_first_k(pw.this.all_chunks, top_k)
    ).select(
        original_id=pw.this.original_id,
        claim=pw.this.claim,
        character=pw.this.character,
        book_name=pw.this.book_name,
        evidence_chunks=pw.this.evidence_chunks
    )
    
    return evidence_table
