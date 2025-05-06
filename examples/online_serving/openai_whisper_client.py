# SPDX-License-Identifier: Apache-2.0
"""
Example client for using OpenAI-compatible Whisper transcription via vLLM.
"""

import openai

# === User configuration ===
VLLM_SERVER_URL = (
    "http://your-vllm-server:port/v1/"  # Replace with your vLLM server URL
)
WHISPER_MODEL_NAME = (
    "your-whisper-model-name"  # Replace with the name of your Whisper model
)
AUDIO_FILE_PATH = (
    "path/to/your-audio-file.mp3"  # Replace with the path to your audio file
)

# Set the custom vLLM endpoint
openai.base_url = VLLM_SERVER_URL
openai.api_key = "no-key"  # Dummy key for compatibility

# Load and send the audio file for transcription
with open(AUDIO_FILE_PATH, "rb") as audio_file:
    transcript = openai.audio.transcriptions.create(
        model=WHISPER_MODEL_NAME,
        file=audio_file,
        response_format="text",  # Use "json" if you want full metadata
    )

print("Transcription result:")
print(transcript)
