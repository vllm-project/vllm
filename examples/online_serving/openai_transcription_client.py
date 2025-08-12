# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import httpx
from openai import OpenAI

from vllm.assets.audio import AudioAsset

mary_had_lamb = AudioAsset('mary_had_lamb').get_local_path()
winning_call = AudioAsset('winning_call').get_local_path()

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def sync_openai():
    with open(str(mary_had_lamb), "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-small",
            language="en",
            response_format="json",
            temperature=0.0)
        print("transcription result:", transcription.text)


sync_openai()


# OpenAI Transcription API client does not support streaming.
async def stream_openai_response():
    data = {
        "language": "en",
        'stream': True,
        "model": "openai/whisper-large-v3",
    }
    url = openai_api_base + "/audio/transcriptions"
    print("transcription result:", end=' ')
    async with httpx.AsyncClient() as client:
        with open(str(winning_call), "rb") as f:
            async with client.stream('POST', url, files={'file': f},
                                     data=data) as response:
                async for line in response.aiter_lines():
                    # Each line is a JSON object prefixed with 'data: '
                    if line:
                        if line.startswith('data: '):
                            line = line[len('data: '):]
                        # Last chunk, stream ends
                        if line.strip() == '[DONE]':
                            break
                        # Parse the JSON response
                        chunk = json.loads(line)
                        # Extract and print the content
                        content = chunk['choices'][0].get('delta',
                                                          {}).get('content')
                        print(content, end='')


# Run the asynchronous function
asyncio.run(stream_openai_response())
