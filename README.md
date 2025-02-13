# Pdf Question Answering System using LLAMA 3.3 and GROQ Cloud
## Overview
This is a Python-based PDF Question Answering (QA) system that allows users to upload multiple PDF files and ask questions about their contents. The application uses advanced natural language processing techniques to extract and retrieve relevant information from the uploaded documents.
## Features
- PDF file upload
- Document text extraction and chunking
- Vector-based semantic search
- Question answering using state-of-the-art language model
- Gradio-based web interface for easy interaction

## Technologies Used
- Gradio: Web interface creation
- LangChain: Document processing and QA workflow
- HuggingFace: Embedding generation
- Groq: Large Language Model (LLM) for question answering
- Fiass: Vector store for document embeddings

## How It Works
- Content in the URL are loaded and split into smaller chunks
- Document chunks are converted to vector embeddings
- When a query is submitted, semantic similarity search finds relevant document chunks
- The language model generates an answer based on the retrieved chunks

## Dependencies
- gradio
- python-dotenv
- langchain
- transformers
- chromadb
- groq

## Acknowledgments
- LangChain
- HuggingFace
- Groq
- Gradio
