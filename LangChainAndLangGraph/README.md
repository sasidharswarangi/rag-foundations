# Minimal RAG over your own documents

A working RAG pipeline with no framework in the middle. Every step is a
plain, readable Python file so you can see exactly what a framework like
LangChain or LlamaIndex would otherwise hide from you.

## What is RAG?

RAG (Retrieval-Augmented Generation) solves one specific problem: an LLM
only "knows" what was in its training data, so it can't answer questions
about your private documents, or anything that changed after training. RAG
fixes that by **looking things up before answering**, instead of answering
purely from memory.

Concretely, it's two steps glued together — which maps directly onto the
project you just built:

**1. Retrieval** — given a question, find the most relevant passages from
your own documents.
- Your question gets turned into a vector (embedding)
- That vector is compared against pre-computed vectors for every chunk of
  your documents
- The top-k most similar chunks are pulled out (this is your
  `retrieval.py` — cosine similarity + argsort)

**2. Augmented Generation** — hand those chunks to the LLM as context,
alongside the original question, and ask it to answer *using that context*
rather than its own trained knowledge.
- The prompt looks like: "Here are some relevant excerpts: [chunks]. Using
  only this, answer: [question]"
- The LLM synthesizes an answer grounded in what was retrieved (your
  `generate.py`)

**Why this beats just fine-tuning or dumping everything in the context
window:**
- No retraining needed when your documents change — just re-run ingestion
- Scales past what fits in a context window — you might have 10,000
  chunks but only feed the model the 4 that matter
- Answers are traceable — you can point to *which* chunk supported the
  answer (which is why your CLI prints the retrieved chunks with scores
  before generating)
- Reduces hallucination, because the model is reasoning over text you
  handed it, not guessing from memory — though it doesn't eliminate it;
  the model can still misread or ignore the context

**What it does *not* do:**
- It doesn't make the model "understand" your documents in some deep
  persistent sense — every query re-retrieves from scratch, there's no
  memory between questions unless you build that separately
- It's only as good as retrieval — if the wrong chunks get pulled (bad
  chunking, bad embeddings, ambiguous query), the LLM will confidently
  answer from irrelevant context
- It's not magic search — it's semantic similarity, so it can miss exact
  keyword matches (e.g., a specific product code or number) that a plain
  keyword search would catch instantly. That's why production RAG systems
  often combine it with keyword/BM25 search (hybrid retrieval) — noted as
  a next step below.

## Pipeline

```
docs/*.pdf, *.md          loaders.py
        |                      |
        v                      v
   raw text  ---->  chunking.py (sliding window + overlap)
        |
        v
   text chunks ---->  embeddings.py (sentence-transformers, local)
        |
        v
   vectors ---->  ingest.py  ---->  data/index.faiss   (real vector store)
        |                    \----> data/vectors.npy   (for manual retrieval)
        |                     \---> data/metadata.json
        v
   [query] ---->  retrieval.py (embed query, cosine similarity BY HAND, top-k)
        |
        v
   retrieved chunks + question ---->  generate.py (Claude or GPT API call)
        |
        v
   answer
```

## Setup

```bash
pip install -r requirements.txt
```

Set an API key for whichever LLM you'll use for generation:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

(You can also put these in a `.env` file — `python-dotenv` loads it automatically.)

## Run it

1. Drop 5-10 PDFs or markdown files into `docs/`.

2. Ingest them:
   ```bash
   python cli.py ingest --docs ./docs
   ```
   This loads, chunks, embeds, and writes everything to `./data/`.

3. Ask questions:
   ```bash
   python cli.py ask "What does the report say about Q3 churn?"
   ```
   You'll see the retrieved chunks (with cosine scores) printed first,
   then the LLM's answer — so you can always check whether a bad
   answer came from bad retrieval or the LLM ignoring good context.

## What to actually read, in order

1. **`chunking.py`** — the sliding-window chunker. This is most of what
   `RecursiveCharacterTextSplitter` does.
2. **`embeddings.py`** — text -> vector. Swap the model here (e.g. an
   API-based embedder) and nothing downstream changes, because the
   contract is just "list of strings in, numpy array out."
3. **`retrieval.py`** — the important one. `cosine_similarity()` and
   `retrieve()` are the entire algorithm behind `vectorstore.similarity_search()`:
   embed the query, score every stored vector against it, sort, slice.
   No index, no framework — just a numpy array and `argsort`.
4. **`ingest.py`** — glues loading/chunking/embedding together and
   writes both a real FAISS index (for when you want to swap in FAISS's
   optimized search later) and the raw arrays retrieval.py reads.
5. **`generate.py`** — builds the "answer only from this context"
   prompt and calls the LLM.

## Where this deliberately stays manual vs. where a framework would help

- **Chunking, retrieval, prompt construction**: manual on purpose —
  this is the whole point of the exercise.
- **FAISS's own optimized search (IVF, HNSW, etc.)**: not used. At the
  scale of 5-10 documents, brute-force cosine similarity over a numpy
  array *is* what FAISS does internally for small indexes — you're not
  missing anything by skipping the optimized index types yet. Once
  you're indexing 100k+ chunks, swap `retrieve()` for `index.search()`
  on the FAISS index that's already being saved in `ingest.py`.
- **Chroma**: not used here in favor of raw FAISS + numpy, since the
  goal was seeing the vectors directly rather than going through
  another client library. Swapping in Chroma later only touches
  `ingest.py` and `retrieval.py`.

## Next steps once this feels obvious

- Swap `retrieve()`'s brute-force loop for `index.search()` on the
  FAISS index and confirm you get the same top-k results — that's the
  moment "what LangChain hides" stops being mysterious.
- Try re-ranking: retrieve top-20 with cosine similarity, then re-score
  with a cross-encoder before taking the top-4.
- Try hybrid search: combine this cosine score with basic keyword/BM25
  matching.
- Only after all that, try the same task in LangChain or LlamaIndex —
  you'll recognize every internal step.
