import asyncio
import aiohttp


async def test_connect(session):
    try:
        print("Sending request")
        async with session.post("http://localhost:8001/v1/connect/completions", json={
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 150,
        "frequency_penalty": 1.3,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.2,
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [{
                "role": "assistant",
                "content": "what can i help you?"
        }, {
                "role": "user",
                "content": "tell me about us"
        }],
        "stream": True,
        "stream_options": {
                "include_usage": True
        }
}) as response:
            print(response.status)
            if response.status == 200:
                transfer_encoding = response.headers.get('Transfer-Encoding')
                if transfer_encoding == 'chunked':
                    async for chunk in response.content.iter_chunked(1024):
                        try:
                            decoded_chunk = chunk.decode('utf-8')
                            print(decoded_chunk)
                        except UnicodeDecodeError:
                            print(f"Error decoding chunk: {chunk!r}")
                else:
                    print(f"Unexpected Transfer-Encoding: {transfer_encoding}")
            else:
                print(f"Request failed with status code {response.status}")
    except aiohttp.ClientError as e:
        print(f"Error: {e}")

    
async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(2):
            tasks.append(test_connect(session))
        await asyncio.gather(*tasks)


asyncio.run(main())

