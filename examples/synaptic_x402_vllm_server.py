"""
Example: Serving DeepSeek-R1 / Llama-3 on vLLM with SynapticChain HTTP 402 Pay-Per-Inference.
"""

from fastapi import FastAPI
from synaptic_vllm_x402 import VLLMX402Middleware, VLLMX402Config

app = FastAPI(title="vLLM x402 Model Server")

# Attach SynapticChain Layer-1 Micropayment Middleware in 1 line
app.add_middleware(
    VLLMX402Middleware,
    config=VLLMX402Config(
        fee_recipient="syn1dejphz2hjetjqva9fg39c7hg8gpr7muapqyvq7",
        cost_per_request="0.0008",
        currency="sUSD"
    )
)

@app.post("/v1/chat/completions")
async def chat_completions(req: dict):
    # Simulated vLLM model inference response
    return {
        "id": "chatcmpl-synaptic-882",
        "object": "chat.completion",
        "model": "deepseek-ai/DeepSeek-R1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! I am DeepSeek-R1 running on a self-hosted vLLM instance monetized via SynapticChain Layer-1 HTTP 402 micro-settlements."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": { "prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45 }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 vLLM x402 Inference Server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
