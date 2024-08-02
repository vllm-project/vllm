from quart import Quart, request, Response, jsonify, make_response
import aiohttp
import sys
import httpx
import traceback
import os

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session: 
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    yield chunk_bytes
    
@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        prefill_request['max_tokens'] = 1

        # finish prefill
        async for data in forward_request('http://localhost:8100/v1/completions', prefill_request):
            continue

        print(f"Request {prefill_request} prefill done. proceeding to decode.")
        
        # return decode
        generator = forward_request('http://localhost:8200/v1/completions', original_request_data)
        response = await make_response(generator)
        response.timeout = None

        return response
    
    except Exception as e:
        exc_info = sys.exc_info()
        print(e)
        print("".join(traceback.format_exception(*exc_info)))

if __name__ == '__main__':
    app.run(port=8000)
