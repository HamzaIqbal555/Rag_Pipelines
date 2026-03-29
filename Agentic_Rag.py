import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load PDFs from a folder


def load_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs


docs = load_docs("/Users/hamzaiqbal/Documents/Practice Area/Rag_Mastery/data")
print("PDF Pages Loaded:", len(docs))


# Split PDFs into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
chunks = text_splitter.split_documents(docs)
print("Chunks Created:", len(chunks))

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save texts into Chroma vector DB
texts = [c.page_content for c in chunks]
db = Chroma(
    collection_name="rag_store",
    embedding_function=embedding_model
)
db.add_texts(texts)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})


# Local LLM
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150
)


# Agent brain
def agent_controller(query):
    q = query.lower()
    if any(word in q for word in ["pdf", "document", "data", "summarize", "information", "find"]):
        return "search"
    return "direct"


# RAG
def rag_answer(query):
    action = agent_controller(query)

    if action == "search":
        print(f"🕵️ Agent decided to SEARCH document for: '{query}'")
        results = retriever.invoke(query)
        context = "\n".join([r.page_content for r in results])
        final_prompt = f"Use this context:\n{context}\n\nAnswer:\n{query}"
    else:
        print(f"🤖 Agent decided to answer DIRECTLY: '{query}'")
        final_prompt = query

    response = llm(final_prompt)[0]["generated_text"]
    return response


# Test 1: A document-specific question
query = "Summarize the doc in 5 bullet points"
print(rag_answer(query))

print("-" * 20)

# Test 2: A general knowledge question
print(rag_answer("What is an Ideal Resume Format? Explain in 50 words."))
