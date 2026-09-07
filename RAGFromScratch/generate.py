"""
Step 6: Feed retrieved chunks + the question into an LLM and get an answer.

This is the "G" in RAG and it's the simplest part: build a prompt that
puts the retrieved context first, the question second, and an
instruction to only answer from that context (to reduce hallucination
and make it obvious when retrieval failed vs. when the LLM ignored it).

Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment (or a .env
file — python-dotenv is in requirements.txt). Defaults to Anthropic.
"""

import os

from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """You are answering questions using ONLY the context below, \
which was retrieved from the user's own documents. If the context doesn't \
contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['source']}, chunk {c['chunk_id']}]\n{c['text']}" for c in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, chunks: list[dict], provider: str = "anthropic") -> str:
    prompt = build_prompt(question, chunks)

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider '{provider}' — use 'anthropic' or 'openai'")
