"""
Step 7: The CLI. Two commands: `ingest` (run once, or whenever docs change)
and `ask` (run per question).

Usage:
  python cli.py ingest --docs ./docs
  python cli.py ask "What does the Q3 report say about churn?"
  python cli.py ask "..." --provider openai --top-k 6
"""

import argparse

from ingest import build_index
from retrieval import load_store, retrieve
from generate import generate_answer


def cmd_ingest(args):
    build_index(args.docs, args.out, args.chunk_size, args.overlap)


def cmd_ask(args):
    vectors, metadata = load_store(args.data)
    chunks = retrieve(args.question, vectors, metadata, top_k=args.top_k)

    print("--- retrieved chunks ---")
    for c in chunks:
        print(f"[{c['score']:.3f}] {c['source']} (chunk {c['chunk_id']})")
    print("------------------------\n")

    answer = generate_answer(args.question, chunks, provider=args.provider)
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal RAG over your own documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load, chunk, embed, and store your documents.")
    p_ingest.add_argument("--docs", default="./docs")
    p_ingest.add_argument("--out", default="./data")
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--overlap", type=int, default=150)
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question about your ingested documents.")
    p_ask.add_argument("question")
    p_ask.add_argument("--data", default="./data")
    p_ask.add_argument("--top-k", type=int, default=4)
    p_ask.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)
