# src/aggregation.py
import pathway as pw

@pw.udf
def aggregate_predictions(predictions: list, rationales: list):
    """
    Aggregate multiple chunk-level LLM judgments into a single claim-level decision.
    - predictions: list of 0/1 from different chunks
    - rationales: list of strings
    Returns dict with:
      - final_prediction: 0/1
      - final_rationale: short concatenated rationale
    """
    if not predictions:
        return {"final_prediction": 0, "final_rationale": "No evidence provided."}

    # Simple conservative strategy: any contradiction → contradict
    final_pred = 0 if 0 in predictions else 1

    # Combine rationales (take first 2 non-empty)
    non_empty_rats = [r for r in rationales if r.strip()]
    final_rat = " | ".join(non_empty_rats[:2]) if non_empty_rats else ""

    return {"final_prediction": final_pred, "final_rationale": final_rat}


def aggregate_claims(evidence_predictions_table: pw.Table) -> pw.Table:
    """
    Takes a table with columns:
      - story_id
      - claim
      - prediction
      - rationale
    Since we have one row per claim already, just ensure story_id is preserved.
    """
    # Group by story_id to ensure unique results
    agg_table = evidence_predictions_table.groupby(pw.this.story_id).reduce(
        story_id=pw.this.story_id,
        claim=pw.reducers.any(pw.this.claim),  # take any (should be same)
        prediction=pw.reducers.any(pw.this.prediction),  # take any non-zero
        rationale=pw.reducers.sorted_tuple(pw.this.rationale)  # collect all rationales
    )
    
    # Extract first rationale from tuple
    @pw.udf
    def get_first_rationale(rationales_tuple: tuple) -> str:
        if rationales_tuple:
            return str(rationales_tuple[0])
        return ""
    
    agg_table = agg_table.with_columns(
        rationale=get_first_rationale(pw.this.rationale)
    )

    return agg_table.select(
        story_id=pw.this.story_id,
        prediction=pw.this.prediction,
        rationale=pw.this.rationale
    )
