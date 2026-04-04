# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An example showing how to upload a local multimodal file via the
/v1/files endpoint and reference it in a chat completion with the
`vllm-file://<id>` URL scheme.

Launch the vLLM server with file uploads enabled:

    vllm serve allenai/Molmo2-8B \
        --trust-remote-code --max-model-len 6144 \
        --enable-file-uploads --file-upload-max-size-mb 128

Then run this script against a local video/image/audio file:

    python openai_file_upload_client.py path/to/clip.mp4

This avoids the two pain points of inline multimodal inputs:

- Base64 data URLs inflate payloads ~33% and can exceed shell ARG_MAX
  for videos over ~8 MB.
- `file://` URLs require `--allowed-local-media-path` AND the file to
  live on the server machine (useless when client and server are on
  different hosts).
"""

from __future__ import annotations

import sys

from openai import OpenAI
from utils import get_first_model

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)


def run(path: str) -> None:
    model = get_first_model(client)

    # 1. Upload the local file once. The server streams it to disk and
    #    returns a 128-bit capability handle.
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")
    print(f"uploaded:  {uploaded.id}  ({uploaded.bytes} bytes)")

    # 2. Reference it by id in any number of chat completions using
    #    the `vllm-file://<id>` URL scheme. The same file can be used
    #    across multiple turns without re-uploading.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"vllm-file://{uploaded.id}"},
                    },
                    {
                        "type": "text",
                        "text": "Describe what happens in this clip in 2-3 sentences.",
                    },
                ],
            }
        ],
        max_completion_tokens=150,
    )
    print(f"response:  {response.choices[0].message.content}")

    # 3. Clean up. Files also auto-expire after
    #    --file-upload-ttl-seconds (default 1h) and are quota-bounded
    #    by --file-upload-max-total-gb (default 5 GB, LRU eviction).
    client.files.delete(uploaded.id)
    print(f"deleted:   {uploaded.id}")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-media-file>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
