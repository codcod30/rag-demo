# Local RAG Demo

This project is a small Retrieval-Augmented Generation (RAG) prototype built to demonstrate an end-to-end question answering pipeline over a local document collection.

It ingests documents from a `data/` folder, chunks them, generates embeddings, stores them in a lightweight local vector store (`sqlite-vec`), retrieves the top-k most relevant passages for a user query, and uses a local LLM via Ollama to produce an answer grounded in those passages.

The goal of this demo is not production readiness. The goal is to make the retrieval behavior easy to inspect, debug, and explain.

---

## Features

- Ingests 10–30 local text documents  
- Splits documents into overlapping chunks  
- Generates embeddings with `sentence-transformers`  
- Stores chunks and embeddings in SQLite using `sqlite-vec`  
- Retrieves the top-k relevant passages for a query  
- Uses Ollama for fully local answer generation  
- Prints both retrieved context and final answer for easy inspection  

---

## Project Structure

```text
rag-demo/
  src/
    ingest.py
    query.py
  requirements.txt
  README.md
```

## Setup

### 1. Clone repository
```bash
git clone <repo-url>
cd rag-demo
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull a model

Install Ollama, then pull the model used by this demo:

```bash
ollama pull llama3.2
```

You can verify the model is available with:

```bash
ollama list
```
### 5. Create data folder

```bash
mkdir data
```

### 6. Add documents

Place 10–30 `.txt` documents in the `data/` folder.

### 7. Build the index

```bash
python src/ingest.py
```

This loads the documents, chunks them, generates embeddings, and stores everything in `rag.db`.

### 8. Ask a question

```bash
python src/query.py "your question here" 3
```

The second argument is the number of retrieved passages (`k`). If omitted, the default is `3`.

## How It Works

### 1. Document ingestion

`src/ingest.py` reads every file in `data/` and treats each file as one source document.

### 2. Chunking

Each document is split into overlapping character-based chunks. The overlap helps preserve context across chunk boundaries and reduces the chance of splitting a relevant idea in half.

### 3. Embeddings

Each chunk is embedded with `sentence-transformers` using the `all-MiniLM-L6-v2` model. The same embedding model is used at query time so the query vector lives in the same semantic space as the document vectors.

### 4. Vector storage

Embeddings and chunk metadata are stored locally in SQLite using `sqlite-vec`. This keeps the system simple, portable, and easy to inspect.

### 5. Retrieval

When the user asks a question, the question is embedded and compared against the stored chunk embeddings. The system retrieves the top-k most similar chunks and prints them with their document IDs and distances.

### 6. Answer generation

The retrieved chunks are passed into a prompt for Ollama. The prompt instructs the model to answer only from the retrieved context. If the context does not contain the answer, the model is instructed to say so.

### 7. Output

The CLI prints:

* the question
* the retrieved context chunks
* their source document IDs
* their similarity distances
* the final answer

This makes it easy to inspect retrieval quality and failure cases.

## Design Choices

### Why a CLI?

A CLI is the fastest way to show a complete end-to-end RAG pipeline. It is easy to run, easy to debug, and easy to explain.

### Why SQLite and sqlite-vec?

`sqlite-vec` is a lightweight local vector store that keeps the demo self-contained. It avoids the complexity of running a separate vector database service and makes the system easy to set up on a fresh machine.

### Why `all-MiniLM-L6-v2`?

This embedding model is small, fast, and good enough for a prototype. It is easy to run locally and provides a solid baseline for semantic retrieval.

### Why Ollama?

Ollama allows the entire generation step to run locally without API credits. That keeps the demo reproducible and cost-free.

### Why overlapping chunks?

Overlapping chunks reduce the chance of losing meaning at chunk boundaries. This is a simple baseline that works well enough for a small prototype.

## Tradeoffs

* **Simple chunking instead of structure-aware chunking**: the implementation is easy to understand, but it may split concepts across boundaries if the documents are complex.
* **No reranker**: retrieval depends only on embedding similarity, so the top-k results may not always be the best possible passages.
* **Local LLM instead of a hosted model**: this avoids API costs, but output quality depends on the installed Ollama model and the local machine.
* **Character-based chunking**: this is straightforward, but token-based chunking or section-aware chunking would be more precise.

## Limitations

* Works best on clean text documents
* Not optimized for large-scale corpora
* No evaluation harness or automatic retrieval metrics
* No reranking or hybrid search
* No document-level citations beyond the printed chunk metadata
* Answer quality depends heavily on retrieval quality
* Not production-ready for concurrency, access control, or deployment

## Debuggability

The system is intentionally verbose. It prints retrieved chunks, chunk IDs, document IDs, and vector distances before showing the final answer. This makes it easy to see:

* whether the right documents were retrieved
* whether the answer is grounded in the retrieved passages
* where retrieval failed when the answer is incorrect

## Requirements Satisfied

| Requirements                                     | Implementation                                                         |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| Build a prototype over 10–30 documents           | Local `.txt` files placed in `data/`                                   |
| Ingest and chunk the documents                   | `src/ingest.py` loads and chunks each document                         |
| Generate embeddings                              | `sentence-transformers` with `all-MiniLM-L6-v2`                        |
| Store embeddings in a lightweight vector store   | SQLite + `sqlite-vec`                                                  |
| Retrieve top-k relevant passages                 | `src/query.py` retrieves the nearest chunks                            |
| Produce an answer grounded in retrieved passages | Ollama prompt uses only retrieved context                              |
| Show final answer and retrieved context          | CLI prints both clearly                                                |
| Simple local option for vector store             | `sqlite-vec` is used locally with SQLite                               |
| Deliverable can be CLI / notebook / web app      | Implemented as a CLI                                                   |
| Include setup instructions and explanation       | This README includes setup, design choices, tradeoffs, and limitations |

## Example Usage

Build the index:

```bash
python src/ingest.py
```

Ask a question:

```bash
python src/query.py "Summarize the key points in these documents" 3
```

Try different `k` values to inspect retrieval quality:

```bash
python src/query.py "What is this document about?" 1
python src/query.py "What is this document about?" 5
```

## Notes

If Ollama is unavailable, the script still prints the retrieved context. This makes it possible to inspect retrieval behavior even when generation is not available.
