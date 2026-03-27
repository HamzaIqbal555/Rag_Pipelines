import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline

# Load our document
with open("my_knowledge.txt") as f:
    knowledge_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 20,
    length_function = len
)

# 2. Create the chunks
chunks = text_splitter.split_text(knowledge_text)

# print(f"We have {len(chunks)} chunks:")
# for i, chunk in enumerate(chunks):
#     print(f"--- Chunk {i+1} ---\n{chunk}\n")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks)

# print(embeddings.shape)

vec_emb_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(vec_emb_dim)

index.add(np.array(embeddings).astype('float32'))

# print(f'Faiss Index created for {index.ntotal} vectors')

generator = pipeline('text2text-generation', 'google/flan-t5-small')


def answere_question(query):
    #embed the query
    embedded_query = embedder.encode([query]).astype('float32')

    # search vector database for the most similar chunks or vector
    k=2
    distances, indices = index.search(embedded_query, 2)

    # get the original text chunks from the original chunks list
    retreived_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retreived_chunks)

    prompt_template = f"""
    Answer the following question using *only* the provided context.
    If the answer is not in the context, say "I don't have that information."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    answer = generator(prompt_template, max_length = 100)
    print(f'Context: {context}')
    return answer[0]['generated_text']


query = "What is the company's dental plan?"
response = answere_question(query)
print(f'Query: {query}')
print(f'Answer: {response}')