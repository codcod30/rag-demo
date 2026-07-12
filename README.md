# Local RAG Demo

This project is a small Retrieval-Augmented Generation (RAG) prototype built to demonstrate an end-to-end question answering pipeline over a local document collection.

It ingests documents from a `data/` folder, chunks them, generates embeddings, stores them in a lightweight local vector store (`sqlite-vec`), retrieves the top-k most relevant passages for a user query, and uses a local LLM via Ollama to produce an answer grounded in those passages.

Built to deepen my understanding of RAG systems from the ground up, every component from chunking to retrieval to generation is implemented explicitly and kept inspectable, making it easy to understand where the pipeline succeeds and where it fails.

---

## System Overview
 
```
Documents (data/)
      │
      ▼
  src/ingest.py
  ┌─────────────────────────────────────┐
  │ 1. Load .txt documents              │
  │ 2. Split into overlapping chunks    │
  │ 3. Generate embeddings              │
  │    (sentence-transformers)          │
  │ 4. Store in SQLite + sqlite-vec     │
  └─────────────────────────────────────┘
              │
              ▼
           rag.db
              │
              ▼
  src/query.py
  ┌─────────────────────────────────────┐
  │ 1. Embed user query                 │
  │ 2. Retrieve top-k similar chunks    │
  │ 3. Build grounded prompt            │
  │ 4. Generate answer via Ollama       │
  │ 5. Print context + answer           │
  └─────────────────────────────────────┘
```

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
  data/                # place .txt documents here
    ...
  src/
    ingest.py          # document ingestion, chunking, embedding, storage
    query.py           # query embedding, retrieval, prompt building, generation
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
### 5. Sample documents are included

Sample `.txt` documents are already included in the `data/` folder. Remove them if necessary.

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
* Struggles with broad or summarization-style queries when `k` is small, since retrieval returns only a few local chunks rather than full document context

## Observability

The system is intentionally verbose. It prints retrieved chunks, chunk IDs, document IDs, and vector distances before showing the final answer. This makes it easy to see:

* whether the right documents were retrieved
* whether the answer is grounded in the retrieved passages
* where retrieval failed when the answer is incorrect

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
