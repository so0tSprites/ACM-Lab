from dotenv import load_dotenv
import os
import requests
import numpy as np
import json

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = "openai/text-embedding-3-small"
#RERANK_MODEL = "cohere/rerank-v3.5"
CHAT_MODEL = "google/gemma-4-26b-a4b-it"
BASE_URL = "https://openrouter.ai/api/v1"

def get_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

def embed(texts):
    """Generate embeddings for a list of texts."""
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers=get_headers(),
        json={"model": EMBEDDING_MODEL, "input": texts},
    )
    return [item["embedding"] for item in response.json()["data"]]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query_embedding, doc_embeddings, top_n=10):
    """Find the top N most similar documents by cosine similarity."""
    scored = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(doc_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def rerank(query, documents, top_n=3):
    """Rerank documents using a cross-encoder model."""
    response = requests.post(
        f"{BASE_URL}/rerank",
        headers=get_headers(),
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
    )
    return response.json()["results"]

def generate(query, context_docs):
    """Generate an answer grounded in the provided context."""
    context = "\n\n".join(
        f"[{i+1}] {doc}"
        for i, doc in enumerate(context_docs)
    )
    response = requests.post(
        f"{BASE_URL}/responses",
        headers=get_headers(),
        json={
            "model": CHAT_MODEL,
            "instructions": (
                "Answer the user's question using only the provided context. "
                "Cite sources with [n]. If the context is insufficient, say so."
            ),
            "input": f"Context:\n{context}\n\nQuestion: {query}",
        },
    )
    data = response.json()
    message = next(item for item in reversed(data["output"]) if item["type"] == "message")
    return next(part["text"] for part in message["content"] if part["type"] == "output_text")

# --- Pipeline ---

# 1. Index: chunk your knowledge base and embed
chunks = [
    "OpenRouter is a unified API gateway for LLMs. It aggregates models from multiple providers into a single OpenAI-compatible interface.",
    "RAG stands for Retrieval-Augmented Generation. It retrieves relevant documents and uses them as context for LLM generation.",
    "Embeddings convert text into high-dimensional vectors where semantically similar texts are closer together.",
    "Reranking uses a cross-encoder to compare each document directly against the query, producing more accurate relevance scores than embedding similarity alone.",
    "Vector databases like Pinecone, Weaviate, and Qdrant are optimized for storing and querying embedding vectors at scale.",
    "Prompt caching reduces costs by reusing computations for repeated prompt prefixes across requests.",
    "OpenRouter supports provider routing to control which upstream providers serve your requests based on cost, latency, or privacy preferences.",
]
doc_embeddings = embed(chunks)

# 2. Retrieve: embed the query and find similar chunks
query = "How does RAG improve LLM responses?"
query_embedding = embed([query])[0]
top_matches = retrieve(query_embedding, doc_embeddings, top_n=5)
retrieved_texts = [chunks[i] for i, _ in top_matches]

# 3. Rerank: re-score with a cross-encoder for better precision
'''
reranked = rerank(query, retrieved_texts, top_n=3)
context_texts = [r["document"]["text"] for r in reranked]
'''

# 4. Generate: produce a grounded answer
answer = generate(query, retrieved_texts)
print(f"Q: {query}\nA: {answer}")