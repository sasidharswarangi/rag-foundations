# RAG Foundations

A hands-on learning repo for understanding Retrieval-Augmented Generation
(RAG) from first principles — embeddings, vector search, chunking, and
retrieval — before relying on frameworks like LangChain or LlamaIndex to
do it for you.

## Contents

### [`BasicsAndFundamentals/`](./BasicsAndFundamentals)

A Jupyter notebook (`rag_foundations.ipynb`) covering the core building
blocks of a retrieval pipeline:

- What an embedding actually is (toy 2D vectors, then real ones)
- Cosine similarity vs. dot product, and when they differ
- Real embeddings with `sentence-transformers`
- Vector indexes: Flat, IVF, HNSW — what each trades off
- Chunking strategies compared side-by-side
- A tiny end-to-end retrieval example

Start here if you're new to RAG concepts and want the theory made concrete.

### [`RAGFromScratch/`](./RAGFromScratch)

A complete, working RAG pipeline with **no framework in the middle** —
every step (loading, chunking, embedding, indexing, retrieval, generation)
is a plain, readable Python file, so you can see exactly what a framework
like LangChain or LlamaIndex would otherwise hide from you. Includes a CLI
to ingest your own documents and ask questions grounded in them.

Move here once the concepts from `BasicsAndFundamentals/` feel familiar
and you want to build the real thing by hand.

## Branches

- **`master`** — stable branch. Merged, working content only.
- **`dev`** — integration branch. Feature branches are merged here first,
  then periodically merged into `master`.
- **Feature branches** (created off `dev`, merged back via PR, then
  deleted): e.g. `basics` (added the `BasicsAndFundamentals` notebook),
  `RAGFromScratch` (added the framework-free RAG pipeline). New work
  follows the same pattern — branch off `dev`, PR back into `dev`.

## Suggested order

1. Read/run through `BasicsAndFundamentals/rag_foundations.ipynb`.
2. Build and run the pipeline in `RAGFromScratch/` against your own
   documents.
3. Only after both, try the same task in LangChain or LlamaIndex — you'll
   recognize every internal step instead of treating it as a black box.
