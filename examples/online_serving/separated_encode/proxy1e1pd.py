# api_proxy.py
import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
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

POOL_LIMITS = httpx.Limits(
    max_keepalive_connections=64,
    max_connections=64,
)

# Create persistent clients
encode_client: Optional[httpx.AsyncClient] = None
decode_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def startup_event():
    global encode_client, decode_client
    encode_client = httpx.AsyncClient(limits=POOL_LIMITS, timeout=httpx.Timeout(100000.0))
    decode_client = httpx.AsyncClient(limits= POOL_LIMITS,timeout=httpx.Timeout(100000.0))

@app.on_event("shutdown")
async def shutdown_event():
    global encode_client, decode_client
    if encode_client:
        await encode_client.aclose()
    if decode_client:
        await decode_client.aclose()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    stop: Optional[list[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = 1
    user: Optional[str] = None

async def forward_streaming_request(
    request_data: dict,
    request_id: str
) -> AsyncIterator[str]:
    """Forward a streaming request to both servers and return the stream from server2."""
    headers = {"x-request-id": request_id}
    task1 = asyncio.create_task(
        encode_client.post(
            f"{ENCODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            headers=headers,
            timeout=httpx.Timeout(100000)
        )
    )
    await task1
    try:
        async with decode_client.stream(
            "POST",
            f"{PREFILL_DECODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            timeout=httpx.Timeout(100000.0),
            headers=headers
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                yield chunk
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        task1.cancel()
        raise

async def forward_non_streaming_request(
    request_data: dict,
    request_id: str
) -> dict:
    """Forward a non-streaming request to both servers and return response from server2."""
    headers = {"x-request-id": request_id}
    
    # Start request to server1
    task1 = asyncio.create_task(
        encode_client.post(
            f"{ENCODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            headers=headers
        )
    )
    
    try:
        # Make request to server2
        response2 = await decode_client.post(
            f"{PREFILL_DECODE_SERVER_URL}/v1/chat/completions",
            json=request_data,
            headers=headers
        )
        response2.raise_for_status()
        
        # Wait for server1 to complete
        try:
            response1 = await task1
            response1.raise_for_status()
        except Exception as e:
            logger.error(f"Error in server1 request: {e}")
        
        return response2.json()
        
    except Exception as e:
        logger.error(f"Error in non-streaming: {e}")
        task1.cancel()
        raise

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests."""
    try:
        # Parse request data
        request_data = await request.json()
        
        # Generate request ID if not provided
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add rank information to request ID
        request_id = f"{request_id}|{E_RANK}|{PD_RANK}"
        
        # Check if streaming is requested
        is_streaming = request_data.get("stream", False)
        
        if is_streaming:
            # Return streaming response
            return StreamingResponse(
                forward_streaming_request(request_data, request_id),
                media_type="text/event-stream"
            )
        else:
            # Return non-streaming response
            result = await forward_non_streaming_request(request_data, request_id)
            return JSONResponse(content=result)
                
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: Request):
    """Handle completion requests (non-chat)."""
    try:
        # Parse request data
        request_data = await request.json()
        
        # Generate request ID if not provided
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add rank information to request ID
        request_id = f"{request_id}|{E_RANK}|{PD_RANK}"
        
        # Check if streaming is requested
        is_streaming = request_data.get("stream", False)
        
        headers = {"x-request-id": request_id}
        
        if is_streaming:
            # Start request to server1
            task1 = asyncio.create_task(
                encode_client.post(
                    f"{ENCODE_SERVER_URL}/v1/completions",
                    json=request_data,
                    headers=headers
                )
            )
            # await task1
            
            # Stream from server2
            async def stream_completions():
                try:
                    async with decode_client.stream(
                        "POST",
                        f"{PREFILL_DECODE_SERVER_URL}/v1/completions",
                        json=request_data,
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_text():
                            yield chunk
                    
                    # Wait for server1
                    try:
                        response1 = await task1
                        response1.raise_for_status()
                    except Exception as e:
                        logger.error(f"Error in server1 request: {e}")
                except Exception as e:
                    task1.cancel()
                    raise
            
            return StreamingResponse(
                stream_completions(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming request
            task1 = asyncio.create_task(
                encode_client.post(
                    f"{ENCODE_SERVER_URL}/v1/completions",
                    json=request_data,
                    headers=headers
                )
            )
            
            try:
                response2 = await decode_client.post(
                    f"{PREFILL_DECODE_SERVER_URL}/v1/completions",
                    json=request_data,
                    headers=headers
                )
                response2.raise_for_status()
                
                try:
                    response1 = await task1
                    response1.raise_for_status()
                except Exception as e:
                    logger.error(f"Error in server1 request: {e}")
                
                return JSONResponse(content=response2.json())
            except Exception as e:
                task1.cancel()
                raise
                
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models."""
    try:
        response = await decode_client.get(f"{PREFILL_DECODE_SERVER_URL}/v1/models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check both servers
        tasks = [
            encode_client.get(f"{ENCODE_SERVER_URL}/health"),
            decode_client.get(f"{PREFILL_DECODE_SERVER_URL}/health")
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = {
            "proxy": "healthy",
            "encode_server": "healthy" if not isinstance(responses[0], Exception) else "unhealthy",
            "prefill_decode_server": "healthy" if not isinstance(responses[1], Exception) else "unhealthy"
        }
        
        if any(isinstance(r, Exception) for r in responses):
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
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    ENCODE_SERVER_URL = args.encode_server_url
    PREFILL_DECODE_SERVER_URL = args.prefill_decode_server_url
    E_RANK = args.e_rank
    PD_RANK = args.pd_rank
    
    logger.info(f"Starting API proxy on {args.host}:{args.port} with {args.workers} workers")
    logger.info(f"Encode server: {ENCODE_SERVER_URL} (rank {E_RANK})")
    logger.info(f"Prefill/Decode server: {PREFILL_DECODE_SERVER_URL} (rank {PD_RANK})")
    import os
    os.environ["ENCODE_SERVER_URL"] = args.encode_server_url
    os.environ["PREFILL_DECODE_SERVER_URL"] = args.prefill_decode_server_url
    os.environ["E_RANK"] = str(args.e_rank)
    os.environ["PD_RANK"] = str(args.pd_rank)
    if args.workers > 1:
        uvicorn.run(
            "proxy1e1pd:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info",
            access_log=False,  
            loop="uvloop" 
        )
    else:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=False,
            loop="uvloop"
        )