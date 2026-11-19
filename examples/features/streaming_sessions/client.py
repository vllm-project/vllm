# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send image files to a server started with --enable-streaming --enforce-eager.

Usage:
    python client.py --base-url http://localhost:8000 frame1.jpg frame2.jpg
"""

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--system-prompt",
        default="Describe the most recent image in one short sentence.",
    )
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=150) as client:
        response = client.post(
            "/v1/streaming/sessions",
            json={
                "system_prompt": args.system_prompt,
                "retention": {"max_video_segments": 2, "max_session_tokens": 4000},
                "sampling": {"temperature": 0, "max_tokens": 32},
            },
        )
        response.raise_for_status()
        session_url = "/v1/streaming/sessions/" + response.json()["session_id"]
        try:
            for frame in args.frames:
                response = client.post(
                    session_url + "/frame", content=frame.read_bytes()
                )
                response.raise_for_status()
                print(json.dumps(response.json()))
        finally:
            response = client.delete(session_url)
            if response.status_code != 404:
                response.raise_for_status()


if __name__ == "__main__":
    main()
