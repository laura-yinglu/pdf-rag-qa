# PDF RAG QA System

## 🎯 Goal
Build a lightweight Retrieval-Augmented Generation system that can answer questions based on uploaded PDFs.

## 📌 How It Works

This system takes in one or more PDFs, breaks the content into chunks, embeds them via OpenAI or HuggingFace embedding models, stores them in a FAISS vector DB, and allows the user to input a query that retrieves the most relevant chunks.


## 🧱 Tech Stack
- Python, LangChain, FAISS
- OpenAI Embedding (or SentenceTransformers)
- FastAPI (optional)

## 📦 Features
- [ ] PDF chunking
- [ ] Embedding + vector storage
- [ ] Query interface
- [ ] Streamlit/Gradio demo (optional)

## 🚧 Status: In Progress...

## 📁 Project Structure

This is the high-level structure of the PDF RAG QA system:
```
pdf-rag-qa/
├── README.md
├── requirements.txt
├── data/ # original pdf files
├── src/
│ ├── chunking.py
│ ├── embedding.py
│ ├── vector_store.py
│ └── query_api.py
└── notebooks/ # for debug usage: colab/temp testing script
```

## 🧠 Source Code Overview

| File | Description |
|------|-------------|
| `chunking.py` | Splits raw PDF content into overlapping text chunks |
| `embedding.py` | Converts chunks into vector embeddings using OpenAI or HuggingFace models |
| `vector_store.py` | Stores and retrieves vectors via FAISS |
| `query_api.py` | Provides a simple query interface (e.g. FastAPI or CLI-based) |


## 🔁 Workflow

```mermaid
graph TD
    A[PDF] --> B[Chunking]
    B --> C[Embedding]
    C --> D[FAISS Vector DB]
    D --> E[Query Input]
    E --> F[Top-K Relevant Chunks]
    F --> G[Return Raw Chunks]



