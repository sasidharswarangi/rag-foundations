"""
Step 4: Wire load -> chunk -> embed -> store together.

Storage is deliberately two-layered so you can see both sides:

  1. A FAISS IndexFlatIP, saved to disk — this is the "real" vector
     store, in the format the ecosystem actually uses. We build it so
     you have it, but we never call index.search() in this project.
  2. The raw (vectors.npy + metadata.json) pair — what retrieval.py
     actually reads, so the cosine-similarity function you write by
     hand has plain numpy arrays to work with, not a black-box index.

Run: python ingest.py --docs ./docs --out ./data
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from loaders import load_documents
from chunking import chunk_documents
from embeddings import embed_texts


def build_index(docs_dir: str, out_dir: str, chunk_size: int, overlap: int):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from {docs_dir} ...")
    docs = load_documents(docs_dir)
    print(f"  loaded {len(docs)} document(s)")

    print("Chunking ...")
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    print(f"  produced {len(chunks)} chunk(s)")
    if len(chunks) == 0:
        raise ValueError("No chunks produced — check your documents aren't empty.")

    print("Embedding (first run downloads the model, ~90MB) ...")
    vectors = embed_texts([c["text"] for c in chunks])
    print(f"  embedded -> shape {vectors.shape}")

    # --- Layer 1: FAISS index, saved so you have a real vector-store artifact ---
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product; vectors are unit-normalized -> cosine
    index.add(vectors)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    # --- Layer 2: raw arrays, for the manual retrieval function to read ---
    np.save(out_dir / "vectors.npy", vectors)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nSaved to {out_dir}/: index.faiss, vectors.npy, metadata.json")
    print(f"({len(chunks)} chunks from {len(docs)} documents, {dim}-dim vectors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store.")
    parser.add_argument("--docs", default="./docs", help="Folder of PDFs/markdown to ingest")
    parser.add_argument("--out", default="./data", help="Where to write the index + vectors")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    build_index(args.docs, args.out, args.chunk_size, args.overlap)
