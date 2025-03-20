import asyncio
import uuid
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from multiprocessing import Manager

# Initialize the FastAPI app
app = FastAPI()

# Prefill and Decode vLLM workers
PREFILL_BASE_URL = "http://localhost:7080/v1"
DECODE_BASE_URL = "http://localhost:7090/v1"

# Persistent HTTP clients
app.state.prefill_client = None
app.state.decode_client = None

# Shared request store using multiprocessing.Manager()
manager = Manager()
request_store = manager.dict()  # Stores request_id → request_data
kv_cache_ready_flags = manager.dict()  # Stores request_id → readiness flag


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
    print(f"~~~genertated request id {request_id}" )

    # Store request in shared memory
    request_store[request_id] = req_data
    kv_cache_ready_flags[request_id] = False  # KV cache is initially not ready

    try:
        # Send request to prefill worker
        req_data["max_tokens"] = 1
        req_data["request_id"] = request_id
        await send_request_to_vllm(app.state.prefill_client, req_data)

        # Wait for KV cache to be ready, but time out after 60 seconds
        async def wait_for_kv_cache():
            for _ in range(600):  # 600 iterations * 0.1s = 60 seconds max wait
                if kv_cache_ready_flags.get(request_id, False):
                    return True
                await asyncio.sleep(0.1)
            return False  # If it times out

        success = await asyncio.wait_for(wait_for_kv_cache(), timeout=60.0)

        if not success:
            raise HTTPException(status_code=504, detail="Timeout: KV cache not ready after 60 seconds")

        # Retrieve the original request
        req_data = request_store.pop(request_id, None)
        kv_cache_ready_flags.pop(request_id, None)  # Cleanup

        if req_data is None:
            raise HTTPException(status_code=500, detail="Request lost in memory")

        # Send request to decode worker and stream response
        return StreamingResponse(stream_vllm_response(app.state.decode_client, req_data), media_type="application/json")

    except asyncio.TimeoutError:
        request_store.pop(request_id, None)
        kv_cache_ready_flags.pop(request_id, None)
        raise HTTPException(status_code=504, detail="Timeout: KV cache not ready after 60 seconds")

    except Exception as e:
        print(f"Error processing request: {e}")
        request_store.pop(request_id, None)
        kv_cache_ready_flags.pop(request_id, None)
        raise HTTPException(status_code=500, detail="Failed to process request")


@app.post("/v1/kv_cache_ready")
async def kv_cache_ready(request: Request):
    """Handle notification that KV cache is ready and trigger decode request."""
    req_data = await request.json()
    request_id = req_data.get("request_id")

    if not request_id:
        return JSONResponse(status_code=400, content={"error": "Missing request_id"})
    
    request_id = request_id.removeprefix("chatcmpl-")
    print(f"--- received kv_cach_ready {request_id} ")

    # Print out the contents of both dictionaries
    print("===== request_store Contents =====")
    print(json.dumps(request_store, indent=4, default=str))  # Convert to JSON-like format

    print("===== kv_cache_ready_flags Contents =====")
    print(json.dumps(kv_cache_ready_flags, indent=4, default=str))

    if request_id not in request_store:
        return JSONResponse(status_code=404, content={"error": "Request not found"})

    # Mark KV cache as ready in shared memory
    kv_cache_ready_flags[request_id] = True

    return JSONResponse({"message": "KV cache ready, starting decode", "request_id": request_id})
