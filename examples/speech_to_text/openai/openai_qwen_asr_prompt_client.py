# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates the optional ``prompt`` parameter on
``v1/audio/transcriptions`` for Qwen3-ASR.

The prompt is mapped to the model's ``system`` turn and acts as a vocabulary
/ context hint for the transcription. This script transcribes the same audio
twice — once without a prompt and once with one — and prints a diff so the
effect on the output is visible.

Before running, start a vLLM server with a Qwen3-ASR model, e.g.::

    vllm serve Qwen/Qwen3-ASR-0.6B
"""

import argparse
import difflib
import sys

import requests

from vllm.assets.audio import AudioAsset


def transcribe(
    base_url: str,
    model: str,
    audio_path: str,
    prompt: str | None = None,
) -> str:
    url = f"{base_url}/v1/audio/transcriptions"
    data = {"model": model}
    if prompt:
        data["prompt"] = prompt

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path, f, "audio/ogg")}
        resp = requests.post(url, data=data, files=files, timeout=120)

    resp.raise_for_status()
    return resp.json()["text"]


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR `prompt` parameter demo")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="vLLM server base URL"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="Model name")
    args = parser.parse_args()

    audio_path = str(AudioAsset("mary_had_lamb").get_local_path())

    print("\n--- Transcription WITHOUT prompt ---")
    no_prompt = transcribe(args.base_url, args.model, audio_path)
    print(f"Result: {no_prompt}")

    prompt_text = "Listen for the words phonograph and fleece"
    print("\n--- Transcription WITH prompt ---")
    print(f"Prompt used: '{prompt_text}'")
    with_prompt = transcribe(args.base_url, args.model, audio_path, prompt=prompt_text)
    print(f"Result: {with_prompt}")

    print("\n" + "=" * 60)
    print("DIFF: no_prompt  →  with_prompt")
    print("=" * 60)

    words_a = no_prompt.split()
    words_b = with_prompt.split()
    sm = difflib.SequenceMatcher(None, words_a, words_b)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            print(f"  - {' '.join(words_a[i1:i2])}")
            print(f"  + {' '.join(words_b[j1:j2])}")
        elif op == "insert":
            print(f"  + {' '.join(words_b[j1:j2])}")
        elif op == "delete":
            print(f"  - {' '.join(words_a[i1:i2])}")

    print("\n--- Unified diff ---")
    for line in difflib.unified_diff(
        [no_prompt],
        [with_prompt],
        fromfile="no_prompt",
        tofile="with_prompt",
        lineterm="",
    ):
        print(line)

    print("=" * 60)

    if no_prompt == with_prompt:
        print("\nWARNING: Both outputs are identical — prompt had no effect!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
