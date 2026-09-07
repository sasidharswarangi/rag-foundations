"""
Step 2: Split document text into overlapping chunks.

This is a character-based sliding window with overlap — the simplest
chunking strategy that works. It's not "sentence aware" or "semantic";
it just cuts at chunk_size characters and backs up by `overlap` chars
so context isn't lost at the boundary. That's genuinely most of what
LangChain's RecursiveCharacterTextSplitter is doing under the hood,
minus the separator-priority list.

Bigger chunk_size = more context per chunk but less precise retrieval.
Smaller = more precise retrieval but each chunk has less context for
the LLM to reason over. 500-1000 chars with 100-200 overlap is a
reasonable starting point for prose; tune by inspecting your results.
"""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # step forward, but re-include the overlap

    return chunks


def chunk_documents(docs: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """
    docs: [{"source": ..., "text": ...}, ...] from loaders.load_documents

    Returns a flat list of chunk records:
    [{"source": ..., "chunk_id": 0, "text": "..."}, ...]

    Flattening now (rather than keeping a nested per-doc structure) is
    what lets everything downstream — embeddings, the index, retrieval
    results — just be a plain parallel list indexed by an integer.
    That's the whole "abstraction" a vector DB gives you: an id -> record
    lookup. We're building it by hand with list indices.
    """
    chunks = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})
    return chunks
