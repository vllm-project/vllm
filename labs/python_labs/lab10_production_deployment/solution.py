"""Lab 10: Production Deployment - Complete Solution"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str
    tokens: int


app = FastAPI(title="vLLM Production API", version="1.0.0")

# Global engine instance
engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize vLLM engine on startup."""
    global engine
    logger.info("Initializing vLLM engine...")

    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        trust_remote_code=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logger.info("vLLM engine initialized successfully")


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text completion."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        request_id = f"req-{id(request)}"
        final_output = None

        async for output in engine.generate(request.prompt, sampling_params, request_id):
            final_output = output

        if final_output and final_output.outputs:
            generated_text = final_output.outputs[0].text
            token_count = len(final_output.outputs[0].token_ids)

            return GenerateResponse(text=generated_text, tokens=token_count)

        raise HTTPException(status_code=500, detail="Generation failed")

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if engine is None:
        return {"status": "unhealthy", "reason": "Engine not initialized"}
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    # Placeholder for metrics
    return {
        "requests_total": 0,
        "avg_latency_ms": 0.0,
        "engine_status": "running" if engine else "not_initialized"
    }


def main():
    """Run the server."""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
