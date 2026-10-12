# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise image content/order/repeat behavior across all decision types."""

import argparse
import json
import math
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import pybase64 as base64


def png(width, height, color):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = (b"\0" + bytes(color) * width) * height
    data = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )
    return base64.b64encode(
        data + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    ).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    questions = [
        {"type": "predicate", "name": "red", "instructions": "Is the first image red?"},
        {
            "type": "choice",
            "name": "color",
            "instructions": "What color is the first image?",
            "choices": [{"value": "red"}, {"value": "blue"}],
        },
        {
            "type": "score",
            "name": "brightness",
            "instructions": "Rate image brightness.",
            "levels": [{"label": "dark"}, {"label": "bright"}],
        },
    ]
    results = []
    for name, colors in [
        ("red", [(255, 0, 0)]),
        ("blue", [(0, 0, 255)]),
        ("red-blue", [(255, 0, 0), (0, 0, 255)]),
        ("blue-red", [(0, 0, 255), (255, 0, 0)]),
        ("repeat-red", [(255, 0, 0)]),
    ]:
        payload = {
            "model": args.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect the attached image(s)."}
                    ]
                    + [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64," + png(128, 128, c),
                        }
                        for c in colors
                    ],
                }
            ],
            "questions": questions,
        }
        start = time.monotonic()
        req = urllib.request.Request(
            url + "/v1/decisions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            response = json.load(r)
        assert [a["name"] for a in response["answers"]] == [
            "red",
            "color",
            "brightness",
        ]
        for a in response["answers"]:
            probs = (
                [a["probability"]]
                if a["type"] == "predicate"
                else [p["probability"] for p in a["probabilities"]]
            )
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
            if a["type"] != "predicate":
                assert abs(sum(probs) - 1) < 1e-6
        results.append(
            {"case": name, "seconds": time.monotonic() - start, "response": response}
        )
        (out / "progress.json").write_text(json.dumps(results, indent=2))
        print(name, "PASS", flush=True)
    by_case = {r["case"]: r["response"]["answers"] for r in results}
    for name in ("red", "red-blue"):
        assert by_case[name][1]["choice"] == "red"
    for name in ("blue", "blue-red"):
        assert by_case[name][1]["choice"] == "blue"
    for a, b in zip(by_case["red"], by_case["repeat-red"]):
        values = (
            lambda x: [x["probability"]]
            if x["type"] == "predicate"
            else [p["probability"] for p in x["probabilities"]]
        )
        assert max(abs(x - y) for x, y in zip(values(a), values(b))) < 1e-6
    (out / "COMPLETE.json").write_text(
        json.dumps(
            {
                "scope": (
                    "Image API functional checks; "
                    "not a quality or performance benchmark"
                ),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
