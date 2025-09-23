# api_proxy.py
import asyncio
import copy
import json
import time
import uuid
from typing import AsyncIterator, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
import aiohttp
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import argparse
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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


def has_mm_input(request_data: dict):
    if "messages" not in request_data:
        return False
    for message in request_data["messages"]:  
        if not isinstance(message.get("content"), list):  
            continue
        for content_item in message["content"]:  
            if content_item.get("type") in ["image_url", "audio_url", "input_audio"]:  
                return True 
    return False

async def forward_streaming_request(
    request_data: dict,
    request_id: str,
    e_server_url: str,
    pd_server_url: str,
) -> AsyncIterator[str]:
    

    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):
        encoder_request_data = copy.deepcopy(request_data)
        encoder_request_data["max_tokens"] = 1
        if "max_completion_tokens" in encoder_request_data:
            encoder_request_data["max_completion_tokens"] = 1
        task1 = asyncio.create_task(
            encode_session.post(
                f"{e_server_url}/v1/chat/completions",
                json=encoder_request_data,
                headers=headers
            )
        )
        try:
            response = await task1
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail={"error": "Request failed", "message": error_text}
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal server error", "message": str(e)}
            )

    # import time
    # time.sleep(10)
    try:
        async with decode_session.post(
            f"{pd_server_url}/v1/chat/completions",
            json=request_data,
            headers=headers
        ) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_chunked(128):
                if chunk:
                    yield chunk.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        raise

async def forward_non_streaming_request(
    request_data: dict,
    request_id: str,
    e_server_url: str,
    pd_server_url: str,
) -> dict:
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):
        encoder_request_data = copy.deepcopy(request_data)
        encoder_request_data["max_tokens"] = 1
        if "max_completion_tokens" in encoder_request_data:
            encoder_request_data["max_completion_tokens"] = 1
        # Start request to encode server
        task1 = asyncio.create_task(
            encode_session.post(
                f"{e_server_url}/v1/chat/completions",
                json=encoder_request_data,
                headers=headers
            )
        )

        try:
            response = await task1
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail={"error": "Request failed", "message": error_text}
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal server error", "message": str(e)}
            )

    try:
        # Make request to decode server
        async with decode_session.post(
            f"{pd_server_url}/v1/chat/completions",
            json=request_data,
            headers=headers
        ) as response2:
            response2.raise_for_status()
            result = await response2.json()
        return result
    except Exception as e:
        logger.error(f"Error in non-streaming: {e}")
        raise

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests."""
    try:
        e_instance = random.randint(0, len(app.state.e_urls) - 1)
        pd_instance = random.randint(0, len(app.state.pd_urls) - 1)
        e_server_url = app.state.e_urls[e_instance]
        pd_server_url = app.state.pd_urls[pd_instance]


        request_data = await request.json()
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        is_streaming = request_data.get("stream", False)
        if is_streaming:
            return StreamingResponse(
                forward_streaming_request(
                    request_data, request_id, e_server_url, pd_server_url),
                media_type="text/event-stream"
            )
        else:
            result = await forward_non_streaming_request(
                request_data, request_id, e_server_url, pd_server_url)
            return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    try:
        async with decode_session.get(f"{app.state.pd_urls[0]}/v1/models") as response:
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
                for e_url in app.state.e_urls:
                    async with encode_session.get(f"{e_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False
        
        async def check_decode():
            try:
                for pd_url in app.state.pd_urls:
                    async with encode_session.get(f"{pd_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False
        
        encode_healthy, decode_healthy = await asyncio.gather(
            check_encode(), check_decode(), return_exceptions=True
        )
        
        health_status = {
            "proxy": "healthy",
            "encode_servers": "healthy" if encode_healthy else "unhealthy",
            "prefill_decode_servers": "healthy" if decode_healthy else "unhealthy"
        }
        
        if encode_healthy is True and decode_healthy is True:
            return JSONResponse(content={"status": "OK"})

        return JSONResponse(content=health_status, status_code=503)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={"proxy": "unhealthy", "error": str(e)},
            status_code=503
        )

async def send_profile_cmd(request: Request, req_data, profiler_cmd):
    assert profiler_cmd in ["start", "stop"]
    # Send to all encoder and decoder, leaving iterator in same state.

    async def post(url, req_data, session):
        async with session.post(url, json=req_data) as response:
            return await response.json(content_type=None)

    tasks = []
    for e_url in app.state.e_urls:
        tasks.append(
            post(f"{e_url}/{profiler_cmd}_profile", req_data, encode_session)
        )
    
    for pd_url in app.state.pd_urls:
        tasks.append(
            post(f"{pd_url}/{profiler_cmd}_profile", req_data, decode_session)
        )

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            print(f"Response {i} was an exception: {r}", flush=True)
            raise r

    return responses


@app.post("/start_profile")
async def start_profile(request: Request):
    try:
        req_data = await request.json()
        return await send_profile_cmd(request, req_data, "start")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              " - start_profile endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


@app.post("/stop_profile")
async def stop_profile(request: Request):
    try:
        req_data = await request.json()
        return await send_profile_cmd(request, req_data, "stop")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              " - stop_profile endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Proxy for distributed vLLM servers")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")

    parser.add_argument("--encode-servers-urls", type=str, required=True,
                       help="URLs of the encode server in comma separated format"
                            "(e.g., \"http://localhost:8001,http://localhost:8002\")")
    
    parser.add_argument("--prefill-decode-servers-urls", type=str, required=True,
                       help="URLs of the prefill/decode servers in comma separated format"
                            "(e.g., \"http://localhost:8003,http://localhost:8004\")")

    
    args = parser.parse_args()
    app.state.e_urls = args.encode_servers_urls.split(",")
    app.state.pd_urls = args.prefill_decode_servers_urls.split(",")
    
    logger.info(f"Starting API proxy on {args.host}:{args.port} with 1 worker")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop"
    )