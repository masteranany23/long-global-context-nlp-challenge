# src/ingest.py
import pathway as pw

def ingest_novels(novel_dir: str):
    """
    Reads all .txt novels and returns a table with:
    - book_name
    - full_text
    """

    novels = pw.io.fs.read(
        novel_dir,
        format="text",
        with_metadata=True
    ).select(
        book_name=pw.this._metadata["path"],
        full_text=pw.this.data
    )

    return novels
