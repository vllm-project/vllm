import argparse
import aiohttp
import asyncio
from aiohttp import web
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def handle_post(request):
    prefill_port = request.app['prefill_port']
    decode_port = request.app['decode_port']

    logger.debug(f"Received request to {request.path} with method {request.method}")

    # Read and parse the request payload
    try:
        payload = await request.json()
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
    except Exception as e:
        logger.error(f"Error parsing request payload: {str(e)}")
        return web.json_response({'error': str(e)}, status=400)
    
    # Modify max_tokens for prefill request
    payload_prefill = payload.copy()
    payload_prefill["max_tokens"] = 1
    logger.debug(f"Modified prefill payload: {json.dumps(payload_prefill, indent=2)}")

    async with aiohttp.ClientSession() as session:
        # Forward request to prefill port
        async with session.post(f"http://localhost:{prefill_port}/v1/completions", json=payload_prefill) as response_prefill:
            try:
                response_prefill_data = await response_prefill.json()
                logger.debug(f"Prefill response data: {json.dumps(response_prefill_data, indent=2)}")
            except aiohttp.ContentTypeError:
                response_prefill_data = await response_prefill.text()
                logger.debug(f"Prefill response text: {response_prefill_data}")
            
            if response_prefill.status != 200:
                logger.error(f"Prefill request failed with status {response_prefill.status}")
                return web.json_response(response_prefill_data, status=response_prefill.status)

        # Forward original request to decode port
        async with session.post(f"http://localhost:{decode_port}/v1/completions", json=payload) as response_decode:
            logger.debug(f"Forwarding request to decode port {decode_port}")
            if 'stream' in payload and payload['stream']:
                # If streaming, set up a streaming response
                response = web.StreamResponse(status=response_decode.status, reason=response_decode.reason, headers=response_decode.headers)
                await response.prepare(request)
                
                async for data, _ in response_decode.content.iter_chunks():
                    await response.write(data)
                    logger.debug(f"Streaming chunk: {data}")
                await response.write_eof()
                logger.debug("Finished streaming response")
                return response
            else:
                # Handle non-streaming response
                try:
                    response_decode_data = await response_decode.json()
                    logger.debug(f"Decode response data: {json.dumps(response_decode_data, indent=2)}")
                except aiohttp.ContentTypeError:
                    response_decode_data = await response_decode.text()
                    logger.debug(f"Decode response text: {response_decode_data}")
                return web.json_response(response_decode_data, status=response_decode.status)

async def init_app(prefill_port, decode_port):
    app = web.Application()
    app['prefill_port'] = prefill_port
    app['decode_port'] = decode_port
    app.router.add_post('/v1/completions', handle_post)
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Async Proxy server")
    parser.add_argument('prefill_port', type=int, help='Port to forward the first request to (with max_tokens=1)')
    parser.add_argument('decode_port', type=int, help='Port to forward the second request to')
    args = parser.parse_args()

    app = asyncio.run(init_app(args.prefill_port, args.decode_port))
    web.run_app(app, port=8000)