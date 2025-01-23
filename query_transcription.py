from openai import OpenAI
from openai.types.audio import TranscriptionCreateParams
from pathlib import Path
import io

mary_had_lamb = Path('/home/varun/.cache/vllm/assets/vllm_public_assets/mary_had_lamb.ogg')
winning_call = Path('/home/varun/.cache/vllm/assets/vllm_public_assets/winning_call.ogg')

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
                                                    prompt="<|startoftranscript|>",
                                                    response_format="text",
                                                    temperature=0.0) 
    print("transcription result:", transcription)
