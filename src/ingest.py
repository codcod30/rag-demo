import os
import sqlite3
import sqlite_vec
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DB_PATH = "rag.db"

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        docs.append({"id": filename, "text": text})

    return docs


def chunk_text(text, size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size]
        chunks.append(chunk)
        start += size - overlap

    return chunks


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT,
        text TEXT
    )
    """)

    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
        id TEXT PRIMARY KEY,
        embedding FLOAT[384]
    )
    """)

    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM vec_chunks")

    return conn


def main():
    docs = load_documents()

    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for i, c in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['id']}_{i}",
                "doc_id": doc["id"],
                "text": c
            })

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts)

    conn = init_db()

    for i, chunk in enumerate(all_chunks):
        emb = embeddings[i].astype(np.float32).tobytes()

        conn.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)",
            (chunk["id"], chunk["doc_id"], chunk["text"])
        )

        conn.execute(
            "INSERT OR REPLACE INTO vec_chunks VALUES (?, ?)",
            (chunk["id"], emb)
        )

    conn.commit()
    print("Indexing complete!")


if __name__ == "__main__":
    main()