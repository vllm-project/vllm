"""
Example 10: FastAPI Integration

Simple REST API for vLLM inference using FastAPI.

Usage:
    python 10_fastapi_integration.py
    # Then visit http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn


class GenerateRequest(BaseModel):
    """Request model."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    """Response model."""
    text: str


# Initialize FastAPI app
app = FastAPI(title="vLLM API")

# Global LLM instance
llm = None


@app.on_event("startup")
async def startup():
    """Initialize model on startup."""
    global llm
    print("Loading model...")
    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
    print("Model loaded!")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text endpoint."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    output = llm.generate([request.prompt], sampling_params)[0]
    return GenerateResponse(text=output.outputs[0].text)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy" if llm else "loading"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
