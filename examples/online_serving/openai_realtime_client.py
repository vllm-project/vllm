# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to use the vLLM Realtime WebSocket API to perform
audio transcription by uploading an audio file.

Before running this script, you must start the vLLM server with a realtime-capable
model, for example:

    vllm serve mistralai/Voxtral-Mini-3B-Realtime-2602 --enforce-eager

Requirements:
- vllm with audio support
- websockets
- librosa
- numpy

The script:
1. Connects to the Realtime WebSocket endpoint
2. Converts an audio file to PCM16 @ 16kHz
3. Sends audio chunks to the server
4. Receives and prints transcription as it streams
"""

import argparse
import asyncio
import base64
import json

import librosa
import numpy as np
import websockets

from vllm.assets.audio import AudioAsset


def audio_to_pcm16_base64(audio_path: str) -> str:
    """
    Load an audio file and convert it to base64-encoded PCM16 @ 16kHz.
    """
    # Load audio and resample to 16kHz mono
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    # Convert to PCM16
    pcm16 = (audio * 32767).astype(np.int16)
    # Encode as base64
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")


async def realtime_transcribe(audio_path: str, host: str, port: int):
    """
    Connect to the Realtime API and transcribe an audio file.
    """
    uri = f"ws://{host}:{port}/v1/realtime"

    async with websockets.connect(uri) as ws:
        # Wait for session.created
        response = json.loads(await ws.recv())
        if response["type"] == "session.created":
            print(f"Session created: {response['id']}")
        else:
            print(f"Unexpected response: {response}")
            return

        # Signal ready to start
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Convert audio file to base64 PCM16
        print(f"Loading audio from: {audio_path}")
        audio_base64 = audio_to_pcm16_base64(audio_path)

        # Send audio in chunks (4KB of raw audio = ~8KB base64)
        chunk_size = 4096
        audio_bytes = base64.b64decode(audio_base64)
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size

        print(f"Sending {total_chunks} audio chunks...")
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

        # Signal all audio is sent
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        print("Audio sent. Waiting for transcription...\n")

        # Receive transcription
        print("Transcription: ", end="", flush=True)
        while True:
            response = json.loads(await ws.recv())
            if response["type"] == "transcription.delta":
                print(response["delta"], end="", flush=True)
            elif response["type"] == "transcription.done":
                print(f"\n\nFinal transcription: {response['text']}")
                if response.get("usage"):
                    print(f"Usage: {response['usage']}")
                break
            elif response["type"] == "error":
                print(f"\nError: {response['error']}")
                break


def main(args):
    if args.audio_path:
        audio_path = args.audio_path
    else:
        # Use default audio asset
        audio_path = str(AudioAsset("mary_had_lamb").get_local_path())
        print(f"No audio path provided, using default: {audio_path}")

    asyncio.run(realtime_transcribe(audio_path, args.host, args.port))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime WebSocket Transcription Client"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    args = parser.parse_args()
    main(args)
