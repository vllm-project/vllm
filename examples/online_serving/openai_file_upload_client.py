# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end example for the /v1/files upload endpoint.

Demonstrates the full lifecycle: upload a local multimodal file, list
and retrieve its metadata, reference it in multiple chat completions
via the `vllm-file://<id>` URL scheme, then delete. The "upload once,
reuse across turns" pattern is the whole point of the feature — inline
data URLs would re-transmit the bytes on every turn.

Launch the vLLM server with file uploads enabled:

    vllm serve allenai/Molmo2-8B \\
        --trust-remote-code --max-model-len 6144 \\
        --enable-file-uploads --file-upload-max-size-mb 128

Then run this script against a local video / image / audio file:

    python openai_file_upload_client.py path/to/clip.mp4
    python openai_file_upload_client.py photo.jpg "What breed is this dog?"
    python openai_file_upload_client.py speech.mp3

This avoids the two pain points of inline multimodal inputs:

- Base64 data URLs inflate payloads ~33% and can exceed shell ARG_MAX
  for videos over ~8 MB.
- `file://` URLs require `--allowed-local-media-path` AND the file to
  live on the server machine (useless when client and server are on
  different hosts).
"""

from __future__ import annotations

import mimetypes
import sys
from pathlib import Path

from openai import OpenAI
from utils import get_first_model

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Map sniffed/guessed MIME type prefix to the chat-completion content
# type the server expects. vLLM's multimodal loaders dispatch on this
# key to select the right decoder.
_CONTENT_TYPE_BY_PREFIX: dict[str, tuple[str, str]] = {
    "video/": ("video_url", "video_url"),
    "image/": ("image_url", "image_url"),
    "audio/": ("input_audio", "audio_url"),
}

# Default prompts per modality (overridden by a CLI arg if provided).
_DEFAULT_PROMPTS: dict[str, tuple[str, str]] = {
    "video/": (
        "Describe what happens in this clip in 2-3 sentences.",
        "What is the emotional tone of the scene?",
    ),
    "image/": (
        "Describe this image in one sentence.",
        "What colors dominate the composition?",
    ),
    "audio/": (
        "Transcribe the audio.",
        "What is the speaker's tone?",
    ),
}


def _detect_modality(path: Path) -> str:
    """Guess the content-type prefix (video/, image/, audio/) from the
    file extension. The server runs its own magic-byte sniffer and
    ignores the client-side guess, but we need the right chat-message
    content key to send the URL under."""
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        raise SystemExit(
            f"error: could not infer media type from extension {path.suffix!r}. "
            "Supported: video/image/audio files."
        )
    for prefix in _CONTENT_TYPE_BY_PREFIX:
        if mime.startswith(prefix):
            return prefix
    raise SystemExit(
        f"error: MIME type {mime!r} is not in the supported media allowlist "
        "(video/*, image/*, audio/*)."
    )


def _ask(model: str, prefix: str, file_id: str, prompt: str) -> str:
    """Send a single chat-completion request referencing the uploaded
    file, and return the model's text response."""
    content_type, url_key = _CONTENT_TYPE_BY_PREFIX[prefix]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": content_type,
                        url_key: {"url": f"vllm-file://{file_id}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_completion_tokens=150,
    )
    return response.choices[0].message.content or ""


def run(path: Path, custom_prompt: str | None) -> None:
    # Validate modality before connecting to the server — unsupported
    # file types fail fast with a clear error, without waiting on a
    # network round-trip.
    prefix = _detect_modality(path)
    model = get_first_model(client)

    # 1. Upload the local file once. The server streams it to disk and
    #    returns a 128-bit capability handle (file-<32 hex>).
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")
    print(f"uploaded:  {uploaded.id} ({uploaded.bytes} bytes, {path.name})")

    try:
        # 2. List files to confirm it's discoverable (also demonstrates
        #    the GET /v1/files endpoint).
        listing = client.files.list()
        ours = next((f for f in listing.data if f.id == uploaded.id), None)
        assert ours is not None, "uploaded file missing from listing"
        print(f"listed:    1 of {len(listing.data)} visible files")

        # 3. Retrieve metadata (demonstrates GET /v1/files/{id}).
        meta = client.files.retrieve(uploaded.id)
        print(
            f"metadata:  purpose={meta.purpose} status={meta.status} "
            f"expires_at={meta.expires_at}"
        )

        # 4. Reference by id in MULTIPLE chat completions. Same file,
        #    no re-upload. This is the reason /v1/files exists.
        prompts = (custom_prompt,) if custom_prompt else _DEFAULT_PROMPTS[prefix]
        for i, prompt in enumerate(prompts, start=1):
            answer = _ask(model, prefix, uploaded.id, prompt)
            print(f"\nturn {i}:")
            print(f"  prompt:  {prompt}")
            print(f"  answer:  {answer}")
    finally:
        # 5. Clean up. Files also auto-expire after
        #    --file-upload-ttl-seconds (default 1h) and are quota-
        #    bounded by --file-upload-max-total-gb (default 5 GB, LRU
        #    eviction). Explicit delete returns the capability immediately.
        client.files.delete(uploaded.id)
        print(f"\ndeleted:   {uploaded.id}")


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            f"usage: {sys.argv[0]} <path-to-media-file> [custom-prompt]",
            file=sys.stderr,
        )
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"error: {path} is not a file", file=sys.stderr)
        sys.exit(1)
    custom_prompt = sys.argv[2] if len(sys.argv) == 3 else None
    run(path, custom_prompt)


if __name__ == "__main__":
    main()
