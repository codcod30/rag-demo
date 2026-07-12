import sqlite3
import sqlite_vec # SQLite extension for vector similarity search
import numpy as np
import requests # Used to call the Ollama API for local LLM inference
import sys
from sentence_transformers import SentenceTransformer

# Path to the SQLite database created by the ingest.py script
DB_PATH = "rag.db"

# The Ollama model used for answer generation
MODEL = "llama3.2"

# Load the same embedding model used in ingest.py to ensure consistency in vector representations
# Using the same model ensures query embedding and document embeddings are 
# in the same vector space, which is crucial for accurate similarity search.
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=3):
    """
    Retrieve the top-k most semantically similar chunks for a given query.
 
    Args:
        query: The user's question as a plain text string.
        k: The number of chunks to retrieve (default 3).
 
    Returns:
        A list of tuples: (chunk_id, doc_id, text, distance)
        Distance is the vector similarity score — lower means more similar.
    """
    
    conn = sqlite3.connect(DB_PATH)

    # Enable the SQLite extension for vector similarity search
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Embed the query using the same model used for document embeddings
    # Convert to float32 and then to bytes for sqlite_vec compatibility
    q_emb = embed_model.encode(query).astype(np.float32).tobytes()

    # Perform vector similarity search using sqlite-vec's MATCH operator
    # Joins vec_chunks (vector index) with chunks (text metadata) on id
    # Returns the top-k closest chunks by embedding distance
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
    """
    Send a prompt to the locally running Ollama instance and return the response.
 
    Args:
        prompt: The full prompt string including context and question.
 
    Returns:
        The model's response as a string.
 
    Raises:
        requests.HTTPError if the Ollama API returns an error status.
    """

    # Call the Ollama REST API running locally on port 11434
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False # Disable streaming to receive the full response at once
        },
        timeout=120 # Allow up to 120 seconds for the model to generate a response
    )

    # Raise an exception if the API returned a non-200 status code
    res.raise_for_status()

    # Extract and return the generated text from the response JSON
    return res.json()["response"]

def build_prompt(question, contexts):
    """
    Construct a grounded prompt that instructs the LLM to answer
    using only the retrieved context chunks.
 
    Args:
        question: The user's original question.
        contexts: List of retrieved chunks as (chunk_id, doc_id, text, distance) tuples.
 
    Returns:
        A formatted prompt string ready to be sent to the LLM.
 
    The prompt explicitly instructs the model not to use outside knowledge,
    ensuring answers are grounded in the retrieved documents.
    """

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
    """
    Print the full pipeline output in a structured, readable format.
 
    Displays the question, each retrieved chunk with metadata,
    and the final generated answer. The verbose output makes it easy
    to inspect retrieval quality and diagnose failures.
 
    Args:
        question: The user's original question.
        contexts: List of retrieved chunks as (chunk_id, doc_id, text, distance) tuples.
        answer: The LLM-generated answer string.
    """

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
    """
    Main query pipeline:
    1. Parse the query and optional k value from command-line arguments
    2. Retrieve the top-k most relevant chunks from the vector store
    3. Build a grounded prompt using the retrieved context
    4. Generate an answer via Ollama (with graceful fallback if unavailable)
    5. Print the full results including retrieved context and final answer
    """

    # Validate that a query was provided as a command-line argument
    if len(sys.argv) < 2:
        print('Usage: python src/query.py "your question here" [k]')
        sys.exit(1)

    # Parse the query string from the first argument
    question = sys.argv[1]

    # Parse the optional k value from the second argument, defaulting to 3
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # Step 1: Retrieve the top-k most relevant chunks for the query
    contexts = retrieve(question, k=k)

    # Step 2: Build a grounded prompt from the retrieved context
    prompt = build_prompt(question, contexts)

    # Step 3: Generate an answer using Ollama
    # Falls back gracefully if Ollama is unavailable,
    # still printing the retrieved context for inspection
    try:
        answer = ask_ollama(prompt)
    except Exception as e:
        answer = f"LLM unavailable. Showing retrieved context only.\nError: {e}"

    # Step 4: Print the full results
    print_results(question, contexts, answer)


if __name__ == "__main__":
    main()