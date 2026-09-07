"""
Step 3: Turn text chunks into vectors.

Uses sentence-transformers locally (all-MiniLM-L6-v2: small, fast on
CPU, 384-dim, no API key, no per-call cost). Good enough to learn on.

If you'd rather use OpenAI's or Voyage's embedding API instead, swap
the body of embed_texts() — everything downstream (the index, the
manual retrieval function) only cares that it gets back a numpy array
of shape (n_texts, dim), so nothing else needs to change. That
swappability IS the abstraction boundary you actually want to
understand: embeddings in, vectors out.
"""

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(
        texts,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=True,  # unit-norm vectors -> dot product == cosine similarity
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
