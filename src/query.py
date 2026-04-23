import sqlite3
import sqlite_vec
import numpy as np
import requests
import sys
from sentence_transformers import SentenceTransformer

DB_PATH = "rag.db"
MODEL = "llama3.2"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, k=3):
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    q_emb = embed_model.encode(query).astype(np.float32).tobytes()

    results = conn.execute("""
        SELECT c.id, c.doc_id, c.text, v.distance
        FROM vec_chunks v
        JOIN chunks c ON v.id = c.id
        WHERE v.embedding MATCH ?
          AND k = ?
    """, (q_emb, k)).fetchall()

    conn.close()
    return results


def ask_ollama(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    res.raise_for_status()
    return res.json()["response"]


def build_prompt(question, contexts):
    context_blocks = []
    for i, (chunk_id, doc_id, text, distance) in enumerate(contexts, start=1):
        context_blocks.append(
            f"[Context {i}] chunk_id={chunk_id} doc_id={doc_id} distance={distance:.4f}\n{text}"
        )

    context_text = "\n\n".join(context_blocks)

    return f"""
You are a careful assistant answering questions using ONLY the provided context.

Rules:
- Use only the context below.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Keep the answer concise and grounded.

Context:
{context_text}

Question:
{question}

Answer:
""".strip()


def print_results(question, contexts, answer):
    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n" + "=" * 80)
    print("RETRIEVED CONTEXT")
    print("=" * 80)

    for i, (chunk_id, doc_id, text, distance) in enumerate(contexts, start=1):
        print(f"\n--- Retrieved Chunk {i} ---")
        print(f"chunk_id : {chunk_id}")
        print(f"doc_id   : {doc_id}")
        print(f"distance : {distance:.4f}")
        print("text     :")
        print(text[:800] + ("..." if len(text) > 800 else ""))

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(answer)


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/query.py "your question here" [k]')
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    contexts = retrieve(question, k=k)
    prompt = build_prompt(question, contexts)

    try:
        answer = ask_ollama(prompt)
    except Exception as e:
        answer = f"LLM unavailable. Showing retrieved context only.\nError: {e}"

    print_results(question, contexts, answer)


if __name__ == "__main__":
    main()