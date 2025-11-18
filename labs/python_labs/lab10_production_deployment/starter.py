"""Lab 10: Production Deployment - Starter Code"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str
    tokens: int


app = FastAPI(title="vLLM Production API")


@app.on_event("startup")
async def startup_event():
    """Initialize vLLM engine on startup."""
    # TODO 1: Initialize AsyncLLMEngine
    pass


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text completion."""
    # TODO 2: Implement generation endpoint
    pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # TODO 3: Implement health check
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    # TODO 4: Return metrics
    return {"requests": 0, "latency_ms": 0}


def main():
    """Run the server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
