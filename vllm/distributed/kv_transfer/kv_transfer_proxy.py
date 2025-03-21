import asyncio
import uuid
import json
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from multiprocessing import Manager

from collections import defaultdict

# Initialize the FastAPI app
app = FastAPI()

# Prefill and Decode vLLM workers
PREFILL_BASE_URL = "http://localhost:7080/v1"
DECODE_BASE_URL = "http://localhost:7090/v1"

# Global variables
tp_size = int(os.getenv("TP_SIZE", 1))  # Read TP_SIZE from env, default to 1

# Persistent HTTP clients
app.state.prefill_client = None
app.state.decode_client = None

# Shared request store using multiprocessing.Manager()
manager = Manager()
# request_store = manager.dict()  # Stores request_id → request_data

cond = asyncio.Condition()  # Using internal lock to protect shared state
kv_cache_counter = defaultdict(int)

@app.on_event("startup")
async def startup_event():
    """Initialize persistent HTTPX clients."""
    app.state.prefill_client = httpx.AsyncClient(timeout=None, base_url=PREFILL_BASE_URL)
    app.state.decode_client = httpx.AsyncClient(timeout=None, base_url=DECODE_BASE_URL)


@app.on_event("shutdown")
async def shutdown_event():
    """Close the persistent HTTPX clients."""
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


async def send_request_to_vllm(client: httpx.AsyncClient, req_data: dict):
    """Send a request to a vLLM process using a persistent client."""
    response = await client.post("/chat/completions", json=req_data)
    response.raise_for_status()
    return response


async def stream_vllm_response(client: httpx.AsyncClient, req_data: dict):
    """Stream the response from the decode vLLM process."""
    async with client.stream("POST", "/chat/completions", json=req_data) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/chat/completions")
async def proxy_request(request: Request):
    """Send the request to prefill worker but wait for KV cache readiness before sending to decode worker."""
    req_data = await request.json()

    # Generate a unique request_id
    request_id = str(uuid.uuid4())
    print(f"~~~ Generated request ID: {request_id}")
    kv_cache_counter[request_id] = 0

    original_max_tokens = req_data["max_tokens"] if "max_tokens" in req_data else None

    try:
        # Send request to prefill worker
        req_data["max_tokens"] = 1
        req_data["request_id"] = request_id
        await send_request_to_vllm(app.state.prefill_client, req_data)

        async with cond:
            while kv_cache_counter.get(request_id) < tp_size:
                print(f"________ {kv_cache_counter.get(request_id)} {tp_size}")
                await asyncio.wait_for(cond.wait(), timeout=60)

        # Send request to decode worker and stream response
        if original_max_tokens is not None:
            req_data["max_tokens"] = original_max_tokens
        else:
            del req_data["max_tokens"]
        return StreamingResponse(stream_vllm_response(app.state.decode_client, req_data), media_type="application/json")

    except asyncio.TimeoutError:
        # request_store.pop(request_id, None)
        # kv_cache_counter.pop(request_id, None)
        raise HTTPException(status_code=504, detail="Timeout: KV cache not ready after 60 seconds")

    except Exception as e:
        print(f"Error processing request: {e}")
        # request_store.pop(request_id, None)
        # kv_cache_counter.pop(request_id, None)
        raise HTTPException(status_code=500, detail="Failed to process request")


@app.post("/v1/kv_cache_ready")
async def kv_cache_ready(request: Request):
    """Handle notification that KV cache is ready and trigger decode request."""

    req_data = await request.json()
    print(f"~~~~~~ recved {req_data}")
    request_id = req_data.get("request_id")
    request_tp_size = req_data.get("world_size")
    assert request_tp_size == tp_size

    if not request_id:
        return JSONResponse(status_code=400, content={"error": "Missing request_id"})
    
    request_id = request_id.removeprefix("chatcmpl-")
    print(f"--- Received KV cache ready for request ID: {request_id}")

    # Print out the contents of both dictionaries
    # print("===== request_store Contents =====")
    # print(json.dumps(request_store, indent=4, default=str))

    print("===== kv_cache_counter Contents =====")
    print(json.dumps(kv_cache_counter, indent=4, default=str))

    # if request_id not in request_store:
    #     return JSONResponse(status_code=404, content={"error": "Request not found"})

    # Decrement the KV cache counter instead of setting to True
    async with cond:
        kv_cache_counter[request_id] += 1
        cond.notify()

    return JSONResponse({"message": "KV cache updated, checking decode trigger", "request_id": request_id})
