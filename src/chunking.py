# src/chunking.py
import re
import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter


CHAPTER_PATTERN = re.compile(
    r'(?im)^(?:\s*)(chapter\s+([IVXLCDM]+|\d+)|([IVXLCDM]{1,6})|(\d{1,3}))\s*$'
)


def split_into_chapters(text: str, min_chapter_chars: int = 800):
    """
    Robust chapter splitter for novels.

    Detects:
      CHAPTER I
      CHAPTER 1
      I
      1

    Avoids:
      years, page numbers, random numerals

    Returns list of (chapter_id, chapter_text)
    """

    matches = list(CHAPTER_PATTERN.finditer(text))
    chapters = []

    if not matches:
        return [("FULL_BOOK", text.strip())]

    for i, match in enumerate(matches):
        start = match.end()   # skip heading
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        body = text[start:end].strip()

        # filter garbage chapters
        if len(body) < min_chapter_chars:
            continue

        raw_id = match.group(0).strip().upper()

        # normalize
        if raw_id.startswith("CHAPTER"):
            chap_id = raw_id.replace("CHAPTER", "").strip()
        else:
            chap_id = raw_id

        chapters.append((chap_id, body))

    if not chapters:
        return [("FULL_BOOK", text.strip())]

    return chapters


@pw.udf
def chapter_udf(text: str):
    return split_into_chapters(text)

def build_chunks(novels_table):

    chapters = novels_table.select(
        book_name=pw.this.book_name,
        chapters=chapter_udf(pw.this.full_text)
    ).flatten(pw.this.chapters)

    chapters = chapters.select(
        book_name=pw.this.book_name,
        chapter_id=pw.this.chapters[0],
        chapter_text=pw.this.chapters[1]
    )

    # Add a simple numeric index for chapter order
    # In Pathway, we can use the id hash as a simple index
    @pw.udf
    def simple_hash_to_int(chapter_id: str) -> int:
        return hash(chapter_id) % 10000
    
    chapters = chapters.with_columns(
        chapter_index=simple_hash_to_int(pw.this.chapter_id)
    )

    splitter = TokenCountSplitter(
        min_tokens=180,
        max_tokens=300
    )

    chunks = chapters.select(
        book_name=pw.this.book_name,
        chapter_id=pw.this.chapter_id,
        chapter_index=pw.this.chapter_index,
        text=splitter(pw.this.chapter_text)
    ).flatten(pw.this.text)

    # Add chunk local index using hash of row id
    @pw.udf
    def chunk_hash_to_int(text: str) -> int:
        return hash(text[:50]) % 1000  # hash first 50 chars
    
    chunks = chunks.with_columns(
        chunk_local_index=chunk_hash_to_int(pw.this.text)
    )

    # Absolute narrative time
    chunks = chunks.with_columns(
        narrative_time = pw.this.chapter_index * 10_000 + pw.this.chunk_local_index
    )

    return chunks
