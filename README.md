# PDF Question Answering System using LLAMA 3.3
## Overview
This is a Python-based PDF Question Answering (QA) system that allows users to upload multiple PDF files and ask questions about their contents. The application uses advanced natural language processing techniques to extract and retrieve relevant information from the uploaded documents.
This system helps you ask questions about the content of documents you upload, like PDFs, Excel files and web pages.
### Retrieval:
First the documents are loaded and split into smaller chunks for better processing. These chunks are then converted into vector embeddings using a pre-trained model (sentence-transformers/all-MiniLM-L6-v2). These embeddings are stored in a vector database (FAISS). When you ask a question, the system looks for the most relevant document pieces by comparing the question to the stored embeddings.
### Generation:
After finding the relevant pieces, the system uses a language model to generate an answer by reading those pieces. It combines the information from the documents with its own knowledge to give you the best possible answer.

## Features
- PDF file upload
- Document text extraction and chunking
- Vector-based semantic search
- Question answering using state-of-the-art language model
- Streamlit-based web interface for easy interaction

## Technologies Used
- Streamlit: Web interface creation
- LangChain: Document processing and QA workflow
- HuggingFace: Embedding generation
- Groq: Large Language Model (LLM) for question answering
- Fiass: Vector store for document embeddings

## How It Works
PDF documents are loaded and split into smaller chunks
Document chunks are converted to vector embeddings
When a query is submitted, semantic similarity search finds relevant document chunks
The language model generates an answer based on the retrieved chunks

## Dependencies
- gradio
- python-dotenv
- langchain
- transformers
- fiass
- groq
- streamlit

## Acknowledgments
- LangChain
- HuggingFace
- Groq
- Gradio
