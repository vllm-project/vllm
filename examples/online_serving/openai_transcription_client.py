# SPDX-License-Identifier: Apache-2.0
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
with open(str(mary_had_lamb), "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        language="en",
        response_format="text",
        temperature=0.0)
    print("transcription result:", transcription)
