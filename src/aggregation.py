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
      - original_id
      - prediction
      - rationale
    Since we already have one row per claim, just pass through.
    """
    return evidence_predictions_table.select(
        original_id=pw.this.original_id,
        prediction=pw.this.prediction,
        rationale=pw.this.rationale
    )
