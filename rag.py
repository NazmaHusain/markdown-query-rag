from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os
import shutil

# Load Markdown documents
DATAPATH = "data/books"
loader = DirectoryLoader(DATAPATH, glob="*.md")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")

# Optional: preview one chunk
new_doc = chunks[2]
print(new_doc.page_content)
print(new_doc.metadata)

# Convert documents to vectors and store in Chroma
CHROMA_PATH = "chroma"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

# Query the vector store
query_text = "What characters did Alice meet in her dream in Alice's Adventures in Wonderland?"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

results = db.similarity_search_with_relevance_scores(query_text, k=3)

if len(results) == 0 or results[0][1] < 0.7:
    print("Unable to find matching results")
else:
    print(f"Highest similarity score: {results[0][1]}")

# Build prompt
context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
print(prompt)

# Generate response using local LLM
llm = Ollama(model="tinyllama:1.1b-chat-v1-q4_0")
response = llm.predict(prompt)
print(response)
f
