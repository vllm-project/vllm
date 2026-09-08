#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 gate for the EC CPU-offload benchmark.

Confirms that a *single* `ec_both` instance actually reuses encoder outputs
from the CPU mmap region. Sends a pool of large images once (pass 1), then
replays the same images (pass 2), and reports, per pass:

  - `EC save:` / `EC load:` worker DEBUG lines (the batched DMAs)
  - `encoder inputs:` from the scheduler INFO iteration lines, i.e. encoder
    inputs actually *computed* (external EC loads are excluded from this stat)

The pool is sized so that pass 1 overflows the GPU encoder cache
(`max_num_batched_tokens` embeddings), so pass 2 must come from CPU rather
than from a GPU-resident hit. Each request carries a unique text prefix
*before* the image so KV prefix caching cannot shortcut the vision encoder.

Run inside the pod, alongside the server:

    venv-vllm/bin/python bench/phase0_hit_check.py \
        --log /vllm-workspace/logs/phase0_ec.log
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
import urllib.request

from ec_log_stats import read_slice, summarize


def synth_image(seed: int, width: int, height: int) -> str:
    """Deterministic low-frequency PNG as a base64 data URL.

    Low-frequency (not iid noise) so payloads stay small and compress like a
    real photo. Identical bytes for a given seed, which is what makes the
    server-side mm_hash recur across passes.
    """
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    rng = np.random.default_rng(seed)
    fx = rng.uniform(1.0, 6.0, 3)
    fy = rng.uniform(1.0, 6.0, 3)
    phase = rng.uniform(0, 6.28, 3)
    chans = []
    for c in range(3):
        v = np.sin(2 * math.pi * fx[c] * xx / width + phase[c]) * np.cos(
            2 * math.pi * fy[c] * yy / height
        )
        chans.append(((v + 1.0) * 127.5).astype(np.uint8))
    img = Image.fromarray(np.stack(chans, axis=-1), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(base_url: str, model: str, prefix: str, data_url: str, max_tokens: int) -> str:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": "Describe this image in one short sentence.",
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--log", required=True, help="server log path, for stat parsing")
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--width", type=int, default=1288)
    p.add_argument("--height", type=int, default=728)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--settle-s", type=float, default=3.0)
    args = p.parse_args()

    print(f"[phase0] building {args.num_images} images {args.width}x{args.height}")
    pool = [synth_image(i, args.width, args.height) for i in range(args.num_images)]
    print(f"[phase0] payload {len(pool[0]) / 1e6:.2f} MB per image (base64)")

    marks = [os.path.getsize(args.log)]
    outputs = []
    for pass_idx in (1, 2):
        t0 = time.monotonic()
        texts = []
        for i, data_url in enumerate(pool):
            # Unique prefix per (pass, image): keeps the image out of a
            # reusable KV prefix, so the encoder is genuinely required.
            texts.append(
                chat(
                    args.base_url,
                    args.model,
                    f"[req p{pass_idx}-{i} nonce {pass_idx * 7919 + i * 104729}] ",
                    data_url,
                    args.max_tokens,
                )
            )
        elapsed = time.monotonic() - t0
        outputs.append(texts)
        print(
            f"[phase0] pass {pass_idx}: {len(pool)} requests in {elapsed:.2f}s "
            f"({elapsed / len(pool) * 1000:.0f} ms/req)"
        )
        time.sleep(args.settle_s)  # let async stat/DMA-completion logs land
        marks.append(os.path.getsize(args.log))

    stats = [summarize(read_slice(args.log, marks[i], marks[i + 1])) for i in range(2)]

    print("\n[phase0] ---- per-pass server-side accounting ----")
    keys = list(stats[0])
    width = max(len(k) for k in keys)
    print(f"{'metric'.ljust(width)}  {'pass1':>14}  {'pass2':>14}")
    for k in keys:
        print(f"{k.ljust(width)}  {stats[0][k]:>14}  {stats[1][k]:>14}")

    identical = sum(a == b for a, b in zip(*outputs))
    print(f"\n[phase0] identical outputs across passes: {identical}/{len(pool)}")

    p1, p2 = stats
    gate_load = p2["ec_load_entries"] > 0
    gate_skip = p2["encoder_inputs_computed"] < p1["encoder_inputs_computed"]
    print(f"[phase0] GATE ec_load fired on pass 2: {'PASS' if gate_load else 'FAIL'}")
    print(f"[phase0] GATE encoder compute dropped: {'PASS' if gate_skip else 'FAIL'}")
    if p1["encoder_inputs_computed"]:
        avoided = 1 - p2["encoder_inputs_computed"] / p1["encoder_inputs_computed"]
        print(f"[phase0] encoder inputs avoided on pass 2: {avoided * 100:.1f}%")
    return 0 if (gate_load and gate_skip) else 1


if __name__ == "__main__":
    sys.exit(main())
