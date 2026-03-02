# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test Qwen3-ASR prompt parameter support via v1/audio/transcriptions.

Reproduces the test from PR #35415:
  1. Transcribe mary_had_lamb.ogg WITHOUT a prompt
  2. Transcribe mary_had_lamb.ogg WITH a vocabulary-guiding prompt
  3. Show the diff — the prompt should guide the model toward correct vocabulary
"""

import argparse
import difflib
import sys

import requests

AUDIO_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
LOCAL_AUDIO = "/tmp/mary_had_lamb.ogg"


def download_audio():
    import os

    if os.path.exists(LOCAL_AUDIO):
        return
    print(f"Downloading {AUDIO_URL} ...")
    r = requests.get(AUDIO_URL, timeout=30)
    r.raise_for_status()
    with open(LOCAL_AUDIO, "wb") as f:
        f.write(r.content)
    print(f"Saved to {LOCAL_AUDIO}")


def transcribe(base_url: str, model: str, prompt: str | None = None) -> str:
    url = f"{base_url}/v1/audio/transcriptions"
    data = {"model": model}
    if prompt:
        data["prompt"] = prompt

    with open(LOCAL_AUDIO, "rb") as f:
        files = {"file": ("mary_had_lamb.ogg", f, "audio/ogg")}
        resp = requests.post(url, data=data, files=files, timeout=120)

    resp.raise_for_status()
    return resp.json()["text"]


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR prompt support")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="vLLM server base URL"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="Model name")
    args = parser.parse_args()

    download_audio()

    print("\n--- Test 1: Transcription WITHOUT prompt ---")
    no_prompt = transcribe(args.base_url, args.model)
    print(f"Result: {no_prompt}")

    prompt_text = "Listen for the words phonograph and fleece"
    print("\n--- Test 2: Transcription WITH prompt ---")
    print(f"Prompt used: '{prompt_text}'")
    with_prompt = transcribe(args.base_url, args.model, prompt=prompt_text)
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

    print("\n--- Tests completed successfully ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
