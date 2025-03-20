# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import itertools
import aiohttp
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

# 解析命令行参数
parser = argparse.ArgumentParser(description="Disaggregated Prefill Proxy Server")
parser.add_argument("--p_ports", nargs="+", type=int, required=True, help="List of producer ports")
parser.add_argument("--d_ports", nargs="+", type=int, required=True, help="List of consumer ports")
parser.add_argument("--proxy_port", type=int, default=8000, help="Proxy server port")
args = parser.parse_args()

request_serial = 0
p_ports = args.p_ports
d_ports = args.d_ports
proxy_port = args.proxy_port

d_urls = []
max_burst = 32
for port in d_ports:
    for _ in range(max_burst):
        d_urls.append(f"localhost:{port}")

port_cycle = itertools.cycle(d_urls)

warm = False

async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
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
        async for _ in forward_request(f'http://localhost:{p_ports[0]}/v1/completions',
                                       prefill_request):
            continue
        url = next(port_cycle)
        generator = forward_request(f'http://{url}/v1/completions',
                                original_request_data)
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
    app.run(port=args.proxy_port)
