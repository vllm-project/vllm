# api_proxy.py
import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
import aiohttp
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global configuration
ENCODE_SERVER_URL = "http://localhost:19534"
PREFILL_DECODE_SERVER_URL = "http://localhost:19535"
E_RANK = 0
PD_RANK = 1

# Create persistent clients
encode_session: Optional[aiohttp.ClientSession] = None
decode_session: Optional[aiohttp.ClientSession] = None

@app.on_event("startup")
async def startup_event():
    global encode_session, decode_session
    encode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), 
        timeout=aiohttp.ClientTimeout(total=100000))
    decode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), 
        timeout=aiohttp.ClientTimeout(total=100000))

@app.on_event("shutdown")
async def shutdown_event():
    global encode_session, decode_session
    if encode_session:
        await encode_session.close()
    if decode_session:
        await decode_session.close()

async def forward_streaming_request(
    request_data: dict,
    request_id: str
) -> AsyncIterator[str]:
    headers = {"x-request-id": request_id}
    task1 = asyncio.create_task(
        encode_session.post(
            f"{ENCODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            headers=headers
        )
    )
    await task1
    try:
        async with decode_session.post(
            f"{PREFILL_DECODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            headers=headers
        ) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_chunked(128):
                if chunk:
                    yield chunk.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        task1.cancel()
        raise

async def forward_non_streaming_request(
    request_data: dict,
    request_id: str
) -> dict:
    headers = {"x-request-id": request_id}
    
    # Start request to encode server
    task1 = asyncio.create_task(
        encode_session.post(
            f"{ENCODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            headers=headers
        )
    )
    await task1
    try:
        # Make request to decode server
        async with decode_session.post(
            f"{PREFILL_DECODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            headers=headers
        ) as response2:
            response2.raise_for_status()
            result = await response2.json()
        return result
    except Exception as e:
        logger.error(f"Error in non-streaming: {e}")
        task1.cancel()
        raise

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests."""
    try:
        request_data = await request.json()
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        request_id = f"{request_id}|{E_RANK}|{PD_RANK}"
        is_streaming = request_data.get("stream", False)
        if is_streaming:
            return StreamingResponse(
                forward_streaming_request(request_data, request_id),
                media_type="text/event-stream"
            )
        else:
            result = await forward_non_streaming_request(request_data, request_id)
            return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    try:
        async with decode_session.get(f"{PREFILL_DECODE_SERVER_URL}/v1/models") as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async def check_encode():
            try:
                async with encode_session.get(f"{ENCODE_SERVER_URL}/health") as response:
                    response.raise_for_status()
                    return True
            except Exception:
                return False
        
        async def check_decode():
            try:
                async with decode_session.get(f"{PREFILL_DECODE_SERVER_URL}/health") as response:
                    response.raise_for_status()
                    return True
            except Exception:
                return False
        
        encode_healthy, decode_healthy = await asyncio.gather(
            check_encode(), check_decode(), return_exceptions=True
        )
        
        health_status = {
            "proxy": "healthy",
            "encode_server": "healthy" if encode_healthy is True else "unhealthy",
            "prefill_decode_server": "healthy" if decode_healthy is True else "unhealthy"
        }
        
        if not (encode_healthy is True and decode_healthy is True):
            return JSONResponse(content=health_status, status_code=503)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={"proxy": "unhealthy", "error": str(e)},
            status_code=503
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Proxy for distributed vLLM servers")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")
    parser.add_argument("--encode-server-url", type=str, required=True,
                       help="URL of the encode server (e.g., http://localhost:8001)")
    parser.add_argument("--prefill-decode-server-url", type=str, required=True,
                       help="URL of the prefill/decode server (e.g., http://localhost:8002)")
    parser.add_argument("--e-rank", type=int, default=0, help="Encode server rank")
    parser.add_argument("--pd-rank", type=int, default=1, help="Prefill/decode server rank")
    
    args = parser.parse_args()
    
    ENCODE_SERVER_URL = args.encode_server_url
    PREFILL_DECODE_SERVER_URL = args.prefill_decode_server_url
    E_RANK = args.e_rank
    PD_RANK = args.pd_rank
    
    logger.info(f"Starting API proxy on {args.host}:{args.port} with 1 worker")
    logger.info(f"Encode server: {ENCODE_SERVER_URL} (rank {E_RANK})")
    logger.info(f"Prefill/Decode server: {PREFILL_DECODE_SERVER_URL} (rank {PD_RANK})")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop"
    )