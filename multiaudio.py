#!/usr/bin/env python3

import base64
import concurrent.futures
import io
import math
import struct
import sys
import wave

import requests


VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "google/gemma-4-E2B-it"


def make_wav_base64(duration_s: float, freq_hz: float = 440.0, sample_rate: int = 16000) -> str:
    """Generate a tiny mono PCM WAV and return base64 text."""
    n_samples = int(duration_s * sample_rate)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            sample = int(0.2 * 32767 * math.sin(2.0 * math.pi * freq_hz * t))
            wav.writeframes(struct.pack("<h", sample))

    return base64.b64encode(buf.getvalue()).decode("ascii")


def send_request(i: int, duration_s: float) -> dict:
    audio_b64 = make_wav_base64(duration_s, freq_hz=440.0 + i * 70)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Briefly describe the audio. One sentence only.",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{audio_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }

    r = requests.post(VLLM_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> None:
    # Different durations are important. The bug shows up when batched
    # audio feature tensors have different sequence lengths.
    durations = [1.0, 1.3, 1.7, 2.1, 2.6]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(durations)) as ex:
        futures = [
            ex.submit(send_request, i, duration)
            for i, duration in enumerate(durations)
        ]

        for i, fut in enumerate(futures):
            result = fut.result()  # Intentionally let exceptions break the script.
            text = result["choices"][0]["message"]["content"]
            print(f"[{i}] OK: {text!r}")

    print("PASS")


if __name__ == "__main__":
    main()