#!/usr/bin/env python3
"""
Custom Reranker Service for LibreChat Local AI Stack
Uses Ollama's Qwen3-Reranker-4B model
"""

import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "dengcao/Qwen3-Reranker-4B")
PORT = int(os.getenv("PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# MODELS
# ============================================================

class Document(BaseModel):
    """Document to be reranked"""
    id: str = Field(..., description="Unique document ID")
    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    score: Optional[float] = Field(default=None, description="Initial retrieval score")


class RerankRequest(BaseModel):
    """Reranking request"""
    query: str = Field(..., description="Search query")
    documents: List[Document] = Field(..., description="Documents to rerank")
    top_k: int = Field(default=4, ge=1, le=100, description="Number of top results to return")
    return_documents: bool = Field(default=True, description="Whether to return document content")


class RerankResult(BaseModel):
    """Single reranking result"""
    id: str
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RerankResponse(BaseModel):
    """Reranking response"""
    results: List[RerankResult]
    model: str
    usage: Dict[str, int]


# ============================================================
# RERANKER SERVICE
# ============================================================

class RerankService:
    """Reranking service using Ollama"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Initialized RerankService with model: {model}")

    async def check_model_availability(self) -> bool:
        """Check if the reranker model is available in Ollama"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            available = any(m.get("name") == self.model for m in models)

            if not available:
                logger.warning(f"Model {self.model} not found. Available models: {[m.get('name') for m in models]}")

            return available
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents using Ollama's embedding similarity approach

        For true reranking models, we'd use a cross-encoder approach.
        Here we're using a workaround with embeddings until Ollama
        supports reranker-specific endpoints.
        """
        if not documents:
            return []

        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)

            # Get document embeddings
            reranked = []
            for doc in documents:
                doc_embedding = await self._get_embedding(doc.text)
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                reranked.append((doc, similarity))

            # Sort by similarity (descending)
            reranked.sort(key=lambda x: x[1], reverse=True)

            # Return top-k
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reranking failed: {str(e)}"
            )

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# ============================================================
# FASTAPI APP
# ============================================================

rerank_service: Optional[RerankService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    global rerank_service

    # Startup
    logger.info("Starting Reranker Service...")
    rerank_service = RerankService(OLLAMA_BASE_URL, RERANKER_MODEL)

    # Check model availability
    is_available = await rerank_service.check_model_availability()
    if not is_available:
        logger.warning(
            f"Reranker model '{RERANKER_MODEL}' not found in Ollama. "
            f"Please run: ollama pull {RERANKER_MODEL}"
        )

    yield

    # Shutdown
    logger.info("Shutting down Reranker Service...")
    if rerank_service:
        await rerank_service.close()


app = FastAPI(
    title="Reranker Service",
    description="Reranking service using Ollama for LibreChat RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "reranker",
        "model": RERANKER_MODEL,
        "ollama_url": OLLAMA_BASE_URL
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank_endpoint(request: RerankRequest):
    """
    Rerank documents based on relevance to query

    Args:
        request: Reranking request with query and documents

    Returns:
        Reranked results with scores
    """
    if not rerank_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker service not initialized"
        )

    if not request.documents:
        return RerankResponse(
            results=[],
            model=RERANKER_MODEL,
            usage={"total_tokens": 0}
        )

    logger.info(
        f"Reranking {len(request.documents)} documents for query: '{request.query[:50]}...'"
    )

    # Perform reranking
    reranked = await rerank_service.rerank(
        query=request.query,
        documents=request.documents,
        top_k=request.top_k
    )

    # Format results
    results = [
        RerankResult(
            id=doc.id,
            score=score,
            text=doc.text if request.return_documents else None,
            metadata=doc.metadata if request.return_documents else None
        )
        for doc, score in reranked
    ]

    # Calculate approximate token usage
    total_tokens = len(request.query.split()) + sum(
        len(doc.text.split()) for doc in request.documents[:request.top_k]
    )

    return RerankResponse(
        results=results,
        model=RERANKER_MODEL,
        usage={"total_tokens": total_tokens}
    )


@app.get("/models")
async def list_models():
    """List available models in Ollama"""
    if not rerank_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker service not initialized"
        )

    try:
        response = await rerank_service.client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )
