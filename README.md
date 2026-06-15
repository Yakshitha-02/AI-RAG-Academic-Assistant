# AI-RAG-Academic-Assistant

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions based on their content.

## Features

- PDF document upload and processing
- Automatic document chunking
- Semantic search using vector embeddings
- Context-aware answer generation
- Interactive Streamlit interface
- Retrieval using FAISS
- LLM-powered response generation

## Tech Stack

- Python
- Streamlit
- FAISS
- SentenceTransformers
- OpenRouter API
- NLP

## Workflow

1. Upload PDF documents
2. Extract and chunk text
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant context
6. Generate answers using LLMs

## Project Structure

```text
data/
embeddings/
app.py
requirements.txt
