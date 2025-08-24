# app.py
import json
import pinecone
import google.generativeai as genai
from fastapi import FastAPI, Request
from pydantic import BaseModel
import os

# --- 1. Environment Variables ---
PINECONE_API_KEY = "KEY"
PINECONE_INDEX = "papers-index"
GEMINI_API_KEY = "KEY"

# --- 2. Initialize Clients ---
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
genai.configure(api_key=GEMINI_API_KEY)

# --- 3. FastAPI App ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search_papers(req: QueryRequest):
    try:
        query_text = req.query.strip()
        if not query_text:
            return {"error": "Query text is required"}

        # Create embedding using Gemini
        query_emb = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text
        )["embedding"]

        # Query Pinecone
        results = index.query(
            vector=query_emb,
            top_k=10,
            include_metadata=True
        )

        # Format response
        formatted = []
        for match in results["matches"]:
            md = match.get("metadata", {})
            formatted.append({
                "title": md.get("title", ""),
                "abstract": md.get("abstract", ""),
                "authors": md.get("authors", ""),
                "source_url": md.get("source_url", ""),
                "link": md.get("link", ""),
                "score": match.get("score", 0)
            })
        for r in results["matches"]:
            print(r["score"], r["metadata"]["title"], "-", r["metadata"]["authors"])

        return {"results": formatted}

    except Exception as e:
        return {"error": str(e)}
