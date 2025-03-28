# SPDX-License-Identifier: Apache-2.0

import os

import aiohttp
from quart import Quart, make_response, request
import uuid

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1

        prefill_host = "localhost"
        prefill_port = 20001
        decode_host = "localhost"
        decode_port = 20002
        request_id = f"___decode_host_({decode_host})___decode_port_({decode_port})_{random_uuid()}"

        # finish prefill
        async for _ in forward_request(f'http://{prefill_host}:{prefill_port}/v1/completions',
                                       prefill_request, request_id):
            continue

        # return decode
        generator = forward_request(f'http://{decode_host}:{decode_port}/v1/completions',
                                    original_request_data, request_id)
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10001)