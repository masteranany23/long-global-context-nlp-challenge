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
    
    # Use Pathway's built-in KNN functionality
    # Group by claim_id and use KNN on vectors
    @pw.udf
    def compute_similarity(query_vec, chunk_vec) -> float:
        """Compute cosine similarity between query and chunk vectors."""
        import numpy as np
        q = np.array(query_vec)
        c = np.array(chunk_vec)
        return float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c)))
    
    # Add similarity scores
    joined_with_similarity = joined.with_columns(
        similarity=compute_similarity(pw.this.query_vector, pw.this.chunk_vector)
    )
    
    # Group by Pathway's internal ID and get top-k by similarity
    evidence_agg = joined_with_similarity.groupby(pw.this.id).reduce(
        original_id=pw.reducers.any(pw.this.original_id),
        claim=pw.reducers.any(pw.this.claim),
        character=pw.reducers.any(pw.this.character),
        book_name=pw.reducers.any(pw.this.book_name),
        chunks_with_scores=pw.reducers.tuple(
            pw.make_tuple(pw.this.chunk_text, pw.this.similarity)
        )
    )
    
    # Extract top-k chunks by similarity
    @pw.udf
    def get_top_k_chunks(chunks_with_scores: tuple, k: int) -> list:
        """Sort by similarity (descending) and return top k chunk texts."""
        # Sort by similarity score (second element of tuple) in descending order
        sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)
        # Return just the text (first element) of top k
        return [chunk[0] for chunk in sorted_chunks[:k]]
    
    evidence_table = evidence_agg.with_columns(
        evidence_chunks=get_top_k_chunks(pw.this.chunks_with_scores, top_k)
    ).select(
        original_id=pw.this.original_id,
        claim=pw.this.claim,
        character=pw.this.character,
        book_name=pw.this.book_name,
        evidence_chunks=pw.this.evidence_chunks
    )
    
    return evidence_table
