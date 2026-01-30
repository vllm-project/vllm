# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from urllib.parse import urljoin


def _to_file_url(path: Path) -> str:
    # vLLM expects a file:// URL for local media.
    return path.resolve().as_uri()


def _guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext in (".mp3",):
        return "audio/mpeg"
    if ext in (".m4a",):
        return "audio/mp4"
    if ext in (".mp4", ".m4v", ".mov", ".webm"):
        return "video/mp4"
    if ext in (".flac",):
        return "audio/flac"
    if ext in (".ogg", ".opus"):
        return "audio/ogg"
    return "application/octet-stream"


def _to_data_url(path: Path) -> str:
    mime = _guess_mime_type(path)
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one VibeVoice-ASR request to a running vLLM OpenAI server."
    )
    parser.add_argument("--base-url", default="http://localhost:8006")
    parser.add_argument("--model", default="vibevoice")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--prompt", default="Transcribe the audio.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--use-data-url",
        action="store_true",
        help=(
            "Send audio as a base64 data: URL (does not require "
            "--allowed-local-media-path)."
        ),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses (prints content as it arrives).",
    )
    parser.add_argument(
        "--wait-ready-secs",
        type=int,
        default=120,
        help="Wait for server /health before sending request (0 to disable).",
    )
    args = parser.parse_args()

    try:
        import requests
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Missing dependency `requests`. Install it first.") from exc

    audio_url = (
        _to_data_url(args.audio) if args.use_data_url else _to_file_url(args.audio)
    )
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful ASR assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    if args.stream:
        payload["stream"] = True

    base = args.base_url.rstrip("/") + "/"
    if args.wait_ready_secs > 0:
        health_url = urljoin(base, "health")
        deadline = time.time() + args.wait_ready_secs
        while True:
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() > deadline:
                raise SystemExit(
                    f"Server not ready after {args.wait_ready_secs}s: {health_url}"
                )
            time.sleep(1)

    url = urljoin(base, "v1/chat/completions")
    if not args.stream:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()

        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = None
        if text is not None:
            print("\n[transcript]\n" + text)
        return 0

    resp = requests.post(url, json=payload, stream=True, timeout=12000)
    resp.raise_for_status()

    printed = ""
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[len("data: ") :].strip()
        if data_str == "[DONE]":
            break
        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        try:
            delta = obj["choices"][0].get("delta") or {}
            chunk = delta.get("content")
        except Exception:
            chunk = None
        if chunk:
            printed += chunk
            print(chunk, end="", flush=True)

    if printed:
        print("\n\n[transcript]\n" + printed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
