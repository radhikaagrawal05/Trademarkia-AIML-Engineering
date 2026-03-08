from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import chromadb
from cache import SemanticCache
import uvicorn

app = FastAPI(title="Trademarkia Semantic Search")

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    print("Loading models...")
    app.state.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    app.state.topic_model = BERTopic.load("topic_model")
    app.state.chroma_client = chromadb.PersistentClient(path="chroma_db")
    app.state.collection = app.state.chroma_client.get_collection("newsgroups")
    app.state.cache = SemanticCache()
    print("Ready!")

@app.post("/query")
async def query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Empty query")

    query_emb = app.state.embedder.encode(query, normalize_embeddings=True)
    topics, probs = app.state.topic_model.transform(
        documents=[query],
        embeddings=query_emb.reshape(1, -1)
    )
    dominant_topic = int(topics[0])

    # Semantic cache lookup
    cache_result = app.state.cache.lookup(query, query_emb, probs[0])
    if cache_result:
        return cache_result

    # Cache miss → real retrieval
    results = app.state.collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=10,
        include=["documents", "distances", "metadatas"]
    )

    formatted_results = []
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        formatted_results.append({
            "text": doc[:1000] + "..." if len(doc) > 1000 else doc,
            "score": round(1 - dist, 4),
            "original_category": meta["original_category"],
            "cluster": meta["dominant_topic"]
        })

    response = {
        "query": query,
        "cache_hit": False,
        "result": formatted_results,
        "dominant_cluster": dominant_topic
    }

    # Store in cache
    app.state.cache.store(query, query_emb, formatted_results, dominant_topic, probs[0])

    return response

@app.get("/cache/stats")
async def cache_stats():
    return app.state.cache.stats()

@app.delete("/cache")
async def clear_cache():
    app.state.cache.clear()
    return {"status": "cache cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
