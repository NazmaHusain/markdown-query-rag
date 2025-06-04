# markdown-query-rag
Ask questions about your Markdown files using AI. This project uses LangChain, Chroma, HuggingFace embeddings and TinyLLaMA to build a local RAG pipeline that retrieves relevant chunks and generates context-aware answers, all without using the cloud.

# 📚 Querying Markdown Files with LangChain and Chroma

This is a fun little project where I build a system that can **read `.md` files**, understand what’s inside, and **answer questions** about them.

I use some powerful tools like:

- **LangChain** to connect all the pieces
- **Chroma** to store and search through text (in vector form)
- **HuggingFace models** to turn text into numbers (embeddings)
- **Ollama** with TinyLLaMA (or any small LLM) to generate answers

# What it does?

Here’s what the project does, step-by-step:

1. Reads all the `.md` files from a folder (`data/books`)
2. Breaks the text into smaller pieces (called “chunks”)
3. Converts each chunk into a vector (numbers that capture the meaning)
4. Stores those vectors in a Chroma database
5. When you ask a question, it:
   - Finds the chunks that are most similar to your question
   - Sends them (plus your question) to a language model
   - Shows you a smart, context-based answer!

# Folder Structure

markdown-rag-langchain-chroma/
│
├── data/ # This is where the .md files go
│ └── books/
│
├── chroma/ # This gets created when you run the code
│
├── rag.ipynb # The notebook with all the code
│
└── README.md # This file you're reading!

# required libraries

langchain
chromadb
sentence-transformers
huggingface-hub
ollama
