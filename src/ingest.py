import os
import sqlite3
import sqlite_vec # SQLite extension for vector similarity search
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to the folder containing .txt documents to be ingested
DATA_DIR = "data"

# Path to the SQLite database file where chunks and embeddings will be stored
DB_PATH = "rag.db"

# Load the sentence-transformers embedding model
# all-MiniLM-L6-v2 is a lightweight model that produces 384-dimensional embeddings
# and provides a good balance between speed and semantic quality
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():
    """
    Load all .txt files from the data directory.
    Each file is treated as one source document.
    Returns a list of dicts with 'id' (filename) and 'text' (file contents).
    """

    docs = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Store the filename as the document ID for traceability during retrieval
        docs.append({"id": filename, "text": text})

    return docs


def chunk_text(text, size=300, overlap=50):
    """
    Split a document into overlapping character-based chunks.
 
    Args:
        text: The full document text to split.
        size: The number of characters per chunk (default 300).
        overlap: The number of characters shared between consecutive chunks (default 50).
 
    Returns:
        A list of text chunk strings.
 
    The overlap ensures that relevant context near chunk boundaries is not lost —
    if a key idea spans two chunks, the overlap means it appears in both,
    reducing the chance of missing it during retrieval.
    """

    chunks = []
    start = 0

    while start < len(text):
        # Extract a chunk of 'size' characters starting from 'start'
        chunk = text[start:start + size]
        chunks.append(chunk)

        # Advance the start position by (size - overlap) so the next chunk
        # shares 'overlap' characters with the current one
        start += size - overlap

    return chunks


def init_db():
    """
    Initialise the SQLite database with the required tables.
 
    Creates two tables:
    - chunks: stores the raw text and metadata for each chunk
    - vec_chunks: a virtual table using sqlite-vec for vector similarity search
 
    Clears existing data on each run so the index is always rebuilt fresh.
    Returns an open database connection.
    """
    
    conn = sqlite3.connect(DB_PATH)

    # Enable the sqlite-vec extension to support vector operations
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Table to store chunk text and metadata
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT,
        text TEXT
    )
    """)

    # Virtual table for vector similarity search using sqlite-vec
    # FLOAT[384] specifies that each embedding has 384 dimensions,
    # matching the output size of all-MiniLM-L6-v2
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
        id TEXT PRIMARY KEY,
        embedding FLOAT[384]
    )
    """)

    # Clear existing data so re-running ingest always produces a fresh index
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM vec_chunks")

    return conn


def main():
    """
    Main ingestion pipeline:
    1. Load all documents from the data directory
    2. Split each document into overlapping chunks
    3. Generate embeddings for all chunks in a single batch
    4. Store chunks and embeddings in the SQLite database
    """

    # Step 1: Load all documents from the data directory
    docs = load_documents()

     # Step 2: Chunk all documents and collect into a flat list
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for i, c in enumerate(chunks):
            all_chunks.append({
                # Unique ID combining the document filename and chunk index
                "id": f"{doc['id']}_{i}",
                "doc_id": doc["id"],
                "text": c
            })

    # Step 3: Generate embeddings for all chunks in a single batch
    # Batching is more efficient than encoding one chunk at a time
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts)

    # Step 4: Initialise the database and insert chunks with their embeddings
    conn = init_db()

    for i, chunk in enumerate(all_chunks):
        # Convert the embedding to float32 and serialise as raw bytes
        # sqlite-vec expects embeddings in binary format for storage
        emb = embeddings[i].astype(np.float32).tobytes()

        # Insert the chunk text and metadata into the chunks table
        conn.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)",
            (chunk["id"], chunk["doc_id"], chunk["text"])
        )

        # Insert the serialised embedding into the vec_chunks virtual table
        conn.execute(
            "INSERT OR REPLACE INTO vec_chunks VALUES (?, ?)",
            (chunk["id"], emb)
        )

    # Commit all inserts to persist them to disk
    conn.commit()
    print("Indexing complete!")


if __name__ == "__main__":
    main()