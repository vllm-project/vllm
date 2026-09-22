# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark an OpenAI-compatible generative OCR server on page images.

Start the server with the no-repeat n-gram processor enabled:

    vllm serve MODEL --logits-processors \
        vllm.v1.worker.gpu.sample.no_repeat_ngram:NoRepeatNGramState
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import pathlib
import statistics
import time
import urllib.request

import pybase64 as base64
from PIL import Image


def _image_data_url(path: pathlib.Path) -> str:
    with Image.open(path) as image:
        resampling = getattr(Image, "Resampling", Image)
        image = image.convert("RGB").resize((1036, 1036), resampling.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    payload = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{payload}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-dir", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-file", type=pathlib.Path)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    encoded = [(path.name, _image_data_url(path)) for path in paths]

    def request(item: tuple[str, str]) -> tuple[str, str]:
        name, image_url = item
        body = json.dumps(
            {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": "\nLayout Detection:"},
                        ],
                    },
                ],
                "temperature": 0,
                "top_p": 0.01,
                "top_k": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "repetition_penalty": 1,
                "vllm_xargs": {"no_repeat_ngram_size": 100},
                "max_completion_tokens": 2048,
                "skip_special_tokens": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{args.url.rstrip('/')}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
        return name, result["choices"][0]["message"]["content"]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        list(executor.map(request, encoded[: args.warmup]))
        durations = []
        final_outputs = []
        for round_index in range(args.rounds):
            started = time.perf_counter()
            final_outputs = list(executor.map(request, encoded))
            duration = time.perf_counter() - started
            durations.append(duration)
            print(
                json.dumps(
                    {
                        "round": round_index + 1,
                        "pages": len(encoded),
                        "seconds": duration,
                        "pages_per_second": len(encoded) / duration,
                    }
                )
            )

    canonical = json.dumps(final_outputs, ensure_ascii=False).encode()
    if args.output_file:
        args.output_file.write_bytes(canonical)
    print(
        json.dumps(
            {
                "mean_seconds": statistics.mean(durations),
                "mean_pages_per_second": len(encoded) / statistics.mean(durations),
                "output_sha256": hashlib.sha256(canonical).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
