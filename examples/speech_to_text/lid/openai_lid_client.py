# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Language Identification (LID) demo using the FireRedLID model on vLLM.

FireRedLID is an audio encoder-decoder model that identifies the spoken
language of an audio clip. Unlike ASR models that output full transcriptions,
FireRedLID outputs at most 2 tokens representing the detected language
(e.g. "en", "zh mandarin").

Start the vLLM server:

    vllm serve PatchyTisa/FireRedLID-vllm

Then run this script:

    # Use the built-in sample audio
    python examples/speech_to_text/lid/openai_lid_client.py

    # Use your own audio file(s)
    python examples/speech_to_text/lid/openai_lid_client.py \
        --audio_paths audio_en.wav audio_zh.wav audio_fr.wav

    # Batch-identify multiple files in one run
    python examples/speech_to_text/lid/openai_lid_client.py \
        --audio_paths /path/to/dir/*.wav

Requirements:
- vLLM with audio support
- openai Python SDK
- kaldi_native_fbank (pulled in by the model)
"""

import argparse
import json
import os

from openai import OpenAI

from vllm.assets.audio import AudioAsset

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def identify_language(
    audio_path: str,
    client: OpenAI,
    model: str,
) -> str:
    """
    Send a single audio file to the vLLM transcription endpoint and return
    the detected language tag.

    FireRedLID re-uses the OpenAI-compatible ``/v1/audio/transcriptions``
    endpoint. The "transcription" it returns is actually the language label
    (e.g. ``"en"`` or ``"zh mandarin"``).
    """
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=f,
            model=model,
            response_format="json",
            temperature=0.0,
        )
    return result.text.strip()


def identify_language_raw(
    audio_path: str,
    model: str,
    api_base: str,
) -> str:
    """
    Same as :func:`identify_language` but uses raw HTTP so that the demo
    works without the ``openai`` SDK (useful for quick debugging).
    """
    import requests

    url = f"{api_base}/audio/transcriptions"
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f)}
        data = {
            "model": model,
            "response_format": "json",
        }
        resp = requests.post(url, files=files, data=data)
        resp.raise_for_status()
    return resp.json()["text"].strip()


def identify_language_streaming(
    audio_path: str,
    model: str,
    api_base: str,
) -> str:
    """
    Streaming variant – demonstrates the streaming transcription endpoint.
    For a 1-2 token output the stream finishes almost instantly, but this
    shows that the API path works end-to-end.
    """
    import requests

    url = f"{api_base}/audio/transcriptions"
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f)}
        data = {
            "stream": "true",
            "model": model,
            "response_format": "json",
        }
        response = requests.post(url, files=files, data=data, stream=True)
        response.raise_for_status()

        tokens: list[str] = []
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n"
        ):
            if not chunk:
                continue
            payload = json.loads(chunk[len("data: ") :].decode("utf-8"))
            choice = payload["choices"][0]
            delta = choice.get("delta", {}).get("content", "")
            if delta:
                tokens.append(delta)
            if choice.get("finish_reason") is not None:
                break

    return "".join(tokens).strip()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    api_base = args.api_base.rstrip("/")
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    model = client.models.list().data[0].id
    print(f"Model : {model}")
    print(f"Server: {api_base}\n")

    # Resolve audio paths ------------------------------------------------
    if args.audio_paths:
        audio_paths = args.audio_paths
    else:
        # Fall back to the built-in vLLM sample audios (both are English).
        audio_paths = [
            str(AudioAsset("mary_had_lamb").get_local_path()),
            str(AudioAsset("winning_call").get_local_path()),
        ]

    # Run LID for each file ----------------------------------------------
    print(f"{'Audio File':<50} {'Language (sync)':<20} {'Language (stream)'}")
    print("-" * 90)

    for path in audio_paths:
        basename = os.path.basename(path)

        # 1) Synchronous via OpenAI SDK
        lang_sync = identify_language(path, client, model)

        # 2) Streaming via raw HTTP
        lang_stream = identify_language_streaming(path, model, api_base)

        print(f"{basename:<50} {lang_sync:<20} {lang_stream}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FireRedLID – Language Identification demo via vLLM",
    )
    parser.add_argument(
        "--audio_paths",
        nargs="+",
        default=None,
        help=(
            "One or more audio files to identify. "
            "If omitted, uses vLLM's built-in sample audios."
        ),
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL (default: http://localhost:8000/v1)",
    )
    args = parser.parse_args()
    main(args)
