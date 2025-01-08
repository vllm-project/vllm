import asyncio
import json
import aiohttp

# test connect completions we assume prefill and decode are on the same node 
# 1. node:vllm serve facebook/opt-125m --port 7001 --zmq-server-port 7010 --chat-template ~/vllm/examples/template_chatglm2.jinja
# 2. vllm connect --prefill-addr nodeIp:7010 --decode-addr nodeIp:7010
# 3. python test_request.py

async def test_connect_completions(session):
    try:
        base_url = "http://localhost:8001/v1/connect/completions"
        body = {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 150,
        "frequency_penalty": 1.3,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.2,
        "model": "facebook/opt-125m",
        "prompt": "Can you introduce vllm?",
        "stream": True,
        "stream_options": {
                "include_usage": True
        }}
        print(f"Sending request to {base_url}, body {body}")   
        async with session.post(base_url, json= body) as response:
            
            print(response.status)
            print(response.headers)
            responseText = ""
            if response.status == 200:
                transfer_encoding = response.headers.get('Transfer-Encoding')
                if transfer_encoding == 'chunked':
                    async for chunk in response.content.iter_chunked(1024):
                        try:
                            decoded_chunk = chunk.decode('utf-8')
                            responseText += decoded_chunk
                        except UnicodeDecodeError:
                            print(f"Error decoding chunk: {chunk!r}")
                else:
                    # Print the headers and JSON response
                    print(f"Unexpected Transfer-Encoding: {transfer_encoding} {response.headers} {await response.json()}")  
            else:
                print(f"Request failed with status code {response.status}")
            print(f"baseurl {base_url} response data {extract_data(responseText)}")
    except aiohttp.ClientError as e:
        print(f"Error: {e}")

def extract_data(responseText):
    reply = ""
    for data in responseText.split("\n\n"):
        if data.startswith('data: '):
            content = data[6:]
            try:
                json_data = json.loads(content)
                choices = json_data["choices"]
                if len(choices) > 0:
                    content = choices[0]["text"]
                    reply += content
            except json.JSONDecodeError:
                print(f"Error: Invalid data format: {data}")
                return reply
        else:
            print(f"Error: Invalid data format: {data}")

    return reply

    
async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(1):
            tasks.append(test_connect_completions(session))
        await asyncio.gather(*tasks)


asyncio.run(main())

