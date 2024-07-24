import argparse
import aiohttp
import asyncio
from aiohttp import web
import json

async def handle_post(request):
    prefill_port = request.app['prefill_port']
    decode_port = request.app['decode_port']

    # Read and parse the request payload
    try:
        payload = await request.json()
    except Exception as e:
        return web.json_response({'error': str(e)}, status=400)
    
    # Modify max_tokens for prefill request
    payload_prefill = payload.copy()
    payload_prefill["max_tokens"] = 1

    async with aiohttp.ClientSession() as session:
        # Forward request to prefill port
        async with session.post(f"http://localhost:{prefill_port}/v1/completions", json=payload_prefill) as response_prefill:
            if response_prefill.status != 200:
                return web.json_response(await response_prefill.json(), status=response_prefill.status)

        # Forward original request to decode port
        async with session.post(f"http://localhost:{decode_port}/v1/completions", json=payload) as response_decode:
            if 'stream' in payload and payload['stream']:
                # If streaming, set up a streaming response
                response = web.StreamResponse(status=response_decode.status, reason=response_decode.reason, headers=response_decode.headers)
                await response.prepare(request)
                
                async for data, _ in response_decode.content.iter_chunks():
                    await response.write(data)
                await response.write_eof()
                return response
            else:
                # Return non-streaming response as JSON
                return web.json_response(await response_decode.json(), status=response_decode.status)

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
