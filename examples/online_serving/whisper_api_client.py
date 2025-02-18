# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
from subprocess import CalledProcessError, run

import aiohttp
import numpy as np

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
SAMPLE_RATE = 16000


def load_audio_from_file(file: str, sample_rate: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le",
        "-ac", "1", "-acodec", "pcm_s16le", "-ar",
        str(sample_rate), "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


async def iterate_response(response):
    output_text = ""
    if response.status == 200:
        async for chunk_bytes in response.content:
            chunk_bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue
            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
            if chunk != "[DONE]":
                output_text += json.loads(chunk)["text"]
    return output_text


async def transcribe_from_waveform(base_url: str, file_path: str):
    """Send waveform to API Server for transcription."""

    waveform = load_audio_from_file(file_path, SAMPLE_RATE)
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        url = f"{base_url}/generate_from_waveform"
        data = {
            "waveform_bytes": waveform.tobytes(),
            "sampling_rate": str(SAMPLE_RATE)
        }
        async with session.post(url, data=data) as response:
            output = await iterate_response(response)
            return output


async def transcribe_from_file(base_url: str, file_path: str):
    """Send file to API Server for transcription."""

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        url = f"{base_url}/generate_from_file"
        with open(file_path, 'rb') as f:
            async with session.post(url, data={'file': f}) as response:
                output = await iterate_response(response)
                print(output)


parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, default="1221-135766-0002.wav")
parser.add_argument("--send-waveform", action="store_true")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8000)

if __name__ == "__main__":
    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}"

    if args.send_waveform:
        asyncio.run(
            transcribe_from_waveform(base_url=api_url,
                                     file_path=args.filepath))
    else:
        asyncio.run(
            transcribe_from_file(base_url=api_url, file_path=args.filepath))
