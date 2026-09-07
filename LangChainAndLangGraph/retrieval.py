"""
Step 5: Retrieval — written by hand, on purpose.

This is the part LangChain/LlamaIndex hide behind `vectorstore.similarity_search()`.
There's no magic in there: it's (1) embed the query, (2) compute a
similarity score between the query vector and every stored vector,
(3) sort, (4) take the top k. That's it. Read this file top to bottom
before you ever call a framework's retriever again.
"""

import json
from pathlib import Path

import numpy as np

from embeddings import embed_query


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    query_vec: shape (dim,)
    matrix:    shape (n, dim)  -- one row per stored chunk
    returns:   shape (n,)      -- similarity of query to every chunk

    cosine_similarity(a, b) = (a . b) / (||a|| * ||b||)

    We compute it explicitly (not relying on embed_texts' normalize=True)
    so this function is correct even if you swap in an embedding model
    that doesn't return unit vectors.
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm  # dot product of every row with the query


def load_store(data_dir: str = "./data"):
    data_dir = Path(data_dir)
    vectors = np.load(data_dir / "vectors.npy")
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    if len(metadata) != vectors.shape[0]:
        raise ValueError("vectors.npy and metadata.json are out of sync — re-run ingest.py")
    return vectors, metadata


def retrieve(query: str, vectors: np.ndarray, metadata: list[dict], top_k: int = 4) -> list[dict]:
    """
    The whole algorithm:
      1. embed the query into the same vector space as the chunks
      2. score every chunk against it
      3. argsort descending, take the first top_k
      4. attach the score to each result for transparency/debugging
    """
    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, vectors)

    top_indices = np.argsort(-scores)[:top_k]  # negate for descending order

    results = []
    for idx in top_indices:
        record = dict(metadata[idx])
        record["score"] = float(scores[idx])
        results.append(record)
    return results


if __name__ == "__main__":
    # quick manual smoke test
    import sys

    vectors, metadata = load_store()
    query = " ".join(sys.argv[1:]) or "What is this about?"
    for r in retrieve(query, vectors, metadata, top_k=4):
        print(f"[{r['score']:.3f}] {r['source']} (chunk {r['chunk_id']})")
        print(f"  {r['text'][:150]}...\n")
