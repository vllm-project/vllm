# epd_proxy.py - E+P+D Separation Proxy
import asyncio
import copy
import json
import logging
import os
import time
import uuid
import random
from typing import AsyncIterator, Optional, Dict, Any, Union
from fastapi import FastAPI, Request, HTTPException
import aiohttp
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import argparse


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("proxy")

app = FastAPI()

encode_session: Optional[aiohttp.ClientSession] = None
prefill_session: Optional[aiohttp.ClientSession] = None
decode_session: Optional[aiohttp.ClientSession] = None


@app.on_event("startup")
async def startup_event():
    global encode_session, prefill_session, decode_session
    encode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=100000))
    prefill_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=100000))
    decode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=100000))


@app.on_event("shutdown")
async def shutdown_event():
    global encode_session, prefill_session, decode_session
    if encode_session:
        await encode_session.close()
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


def has_mm_input(request_data: dict):
    """Check if request has multimodal input (images, audio, etc.)"""
    if "messages" not in request_data:
        return False
    for message in request_data["messages"]:  
        if not isinstance(message.get("content"), list):  
            continue
        for content_item in message["content"]:  
            if content_item.get("type") in ["image_url", "audio_url", "input_audio"]:  
                return True
    return False


async def process_encoder_stage(
    request_data: dict,
    request_id: str,
    e_server_url: str
):
    """Process request through Encoder stage (if has MM input)"""
    if not has_mm_input(request_data):
        return
    
    logger.debug(f"Processing MM input through encoder for request_id: {request_id}/ url: {e_server_url}")
    encoder_request_data = request_data.copy()
    encoder_request_data["max_tokens"] = 1
    if "max_completion_tokens" in encoder_request_data:
        encoder_request_data["max_completion_tokens"] = 1
    
    headers = {"x-request-id": request_id}
    task1 = asyncio.create_task(
        encode_session.post(
            f"{e_server_url}/v1/chat/completions",
            json=encoder_request_data,
            headers=headers
        )
    )
    try:
        response = await task1  # seems fire-and-forget didnt gain any benefit
        if response.status != 200:
            error_text = await response.text()
            raise HTTPException(
                status_code=response.status,
                detail={"error": "Encoder request failed", "message": error_text}
            )
        logger.debug(f"Encoder processing completed successfully for request_id: {request_id}")
        return
    except Exception as e:
        logger.error(f"Encoder processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Encoder processing error", "message": str(e)}
        )


async def process_prefill_stage(
    request_data: dict,
    request_id: str,
    p_server_url: str
) -> dict:
    """Process request through Prefill stage and return kv_transfer_params"""
    logger.debug(f"Processing through prefill for request_id: {request_id}/ url: {p_server_url}")
    
    prefill_request = request_data.copy()
    prefill_request['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    prefill_request["stream"] = False
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1
    if "stream_options" in prefill_request:
        del prefill_request["stream_options"]
    
    headers = {"x-request-id": request_id}
    try:
        prefill_response = await prefill_session.post(
            f"{p_server_url}/v1/chat/completions",
            json=prefill_request,
            headers=headers
        )
        prefill_response.raise_for_status()

        if prefill_response.status != 200:
            error_text = await prefill_response.text()
            raise HTTPException(
                status_code=prefill_response.status,
                detail={"error": "Prefill request failed", "message": error_text}
            )
        logger.debug(f"Prefill processing completed successfully for request_id: {request_id}")
        
        return prefill_response

    except Exception as e:
        logger.error(f"Prefill processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Prefill processing error", "message": str(e)}
        )


async def forward_request(
    request_data: dict,
    request_id: str,
    e_server_url: str,
    p_server_url: str,
    d_server_url: str,
    is_streaming: bool = False
) -> Union[AsyncIterator[str], dict]:
    """Forward request through E->P->D pipeline"""
    
    # Step 1: Process through Encoder instance (if has MM input)
    await process_encoder_stage(
        request_data, request_id, e_server_url
    )
    
    # Step 2: Process through Prefill instance
    prefill_response = await process_prefill_stage(
        request_data, request_id, p_server_url
    )
    # for nixl connector to facilitate kv transfer...
    prefill_response_json = await prefill_response.json()
    kv_transfer_params = prefill_response_json.get('kv_transfer_params', {})
    if kv_transfer_params:
        request_data["kv_transfer_params"] = kv_transfer_params
        logger.debug(f"kv_transfer_params: {kv_transfer_params}")

    # Step 3: Process through Decode instance
    logger.debug(f"{'Streaming' if is_streaming else 'Getting'} response from decode for request_id: {request_id}/ url: {d_server_url}")
    
    headers = {"x-request-id": request_id}
    
    if is_streaming:
        # Streaming response
        async def generate_stream():
            try:
                async with decode_session.post(
                    f"{d_server_url}/v1/chat/completions",
                    json=request_data,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.content.iter_chunked(128):
                        if chunk:
                            yield chunk.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Error in decode streaming: {e}")
                raise
        
        return generate_stream()
    else:
        # Non-streaming response
        try:
            async with decode_session.post(
                f"{d_server_url}/v1/chat/completions",
                json=request_data,
                headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
            return result
        except Exception as e:
            logger.error(f"Error in decode response: {e}")
            raise


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests through E->P->D pipeline."""
    try:
        # Load balance across instances
        e_instance = random.randint(0, len(app.state.e_urls) - 1)
        p_instance = random.randint(0, len(app.state.p_urls) - 1)
        d_instance = random.randint(0, len(app.state.d_urls) - 1)
       
        e_server_url = app.state.e_urls[e_instance]
        p_server_url = app.state.p_urls[p_instance]
        d_server_url = app.state.d_urls[d_instance]

        request_data = await request.json()
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
       
        is_streaming = request_data.get("stream", False)
       
        logger.debug(f"Processing request {request_id} through E->P->D: {e_server_url} -> {p_server_url} -> {d_server_url}")
       
        if is_streaming:
            stream_generator = await forward_request(
                request_data, request_id, e_server_url, p_server_url, d_server_url, is_streaming=True
            )
            return StreamingResponse(stream_generator, media_type="text/event-stream")
        else:
            result = await forward_request(
                request_data, request_id, e_server_url, p_server_url, d_server_url, is_streaming=False
            )
            return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models from decode instances."""
    try:
        async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint for all instances."""
    try:
        async def check_encode():
            try:
                for e_url in app.state.e_urls:
                    async with encode_session.get(f"{e_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False
       
        async def check_prefill():
            try:
                for p_url in app.state.p_urls:
                    async with prefill_session.get(f"{p_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False
       
        async def check_decode():
            try:
                for d_url in app.state.d_urls:
                    async with decode_session.get(f"{d_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False
       
        encode_healthy, prefill_healthy, decode_healthy = await asyncio.gather(
            check_encode(), check_prefill(), check_decode(), return_exceptions=True
        )

        if not (encode_healthy is True and prefill_healthy is True and decode_healthy is True):
            status_code = 503
        else:
            status_code = 200
        
        return JSONResponse(
            {
                "proxy": "healthy",
                "encode_servers": "healthy" if encode_healthy is True else "unhealthy",
                "prefill_servers": "healthy" if prefill_healthy is True else "unhealthy",
                "decode_servers": "healthy" if decode_healthy is True else "unhealthy"
            },
            status_code=status_code,
        )
       
        return health_status
       
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={"proxy": "unhealthy", "error": str(e)},
            status_code=503
        )


@app.get("/status")
async def get_status():
    """Get status of all instances."""
    return {
        "encode_instance_count": len(app.state.e_urls),
        "prefill_instance_count": len(app.state.p_urls),
        "decode_instance_count": len(app.state.d_urls),
        "encode_instances": app.state.e_urls,
        "prefill_instances": app.state.p_urls,
        "decode_instances": app.state.d_urls
    }

## profiler methods ##

async def _post_if_available(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
) -> Optional[dict]:
    """
    POST `payload` to `url`.
    Returns
    -------
    • The decoded JSON body on success (2xx)  
    • None if the endpoint does not exist (404)  
    • Raises for anything else.
    """
    try:
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:           # profiling disabled on that server
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        # Pass 404 through the branch above, re-raise everything else
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise
    except Exception:
        # Network errors etc.: propagate
        raise

async def _profile_cmd(cmd: str, payload: dict, e_url: str, p_url: str, d_url: str):
    """
    Fire & forget to both clusters, tolerate 404.
    """
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(
        encode_session, f"{e_url}/{cmd}_profile", payload, headers
    )
    prefill_task = _post_if_available(
        prefill_session, f"{p_url}/{cmd}_profile", payload, headers
    )
    decode_task = _post_if_available(
        decode_session, f"{d_url}/{cmd}_profile", payload, headers
    )

    encode_res, prefill_res, decode_res = await asyncio.gather(encode_task, prefill_task, decode_task)

    # If *all* clusters said “I don’t have that route”, surface an error
    if encode_res is prefill_res is decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on all clusters",
        )

    return {
        "encode": encode_res,   # may be None
        "prefill": decode_res,   # may be None
        "decode": decode_res,   # may be None
    }

@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls)
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("start", body, e_url, p_url, d_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls)
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("stop", body, e_url, p_url, d_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E+PD Separation Proxy for distributed vLLM servers")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")

    parser.add_argument("--encode-servers-urls", type=str, required=True,
                       help="URLs of the encode servers in comma separated format"
                            "(e.g., \"http://localhost:8001,http://localhost:8002\")")
   
    parser.add_argument("--prefill-servers-urls", type=str, required=True,
                       help="URLs of the prefill servers in comma separated format"
                            "(e.g., \"http://localhost:8003,http://localhost:8004\")")
   
    parser.add_argument("--decode-servers-urls", type=str, required=True,
                       help="URLs of the decode servers in comma separated format"
                            "(e.g., \"http://localhost:8005,http://localhost:8006\")")
   
    args = parser.parse_args()
    app.state.e_urls = args.encode_servers_urls.split(",")
    app.state.p_urls = args.prefill_servers_urls.split(",")
    app.state.d_urls = args.decode_servers_urls.split(",")
   
    logger.info(f"Starting E+PD separation proxy on {args.host}:{args.port}")
    logger.info(f"Encode instances: {app.state.e_urls}")
    logger.info(f"Prefill instances: {app.state.p_urls}")
    logger.info(f"Decode instances: {app.state.d_urls}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop"
    )