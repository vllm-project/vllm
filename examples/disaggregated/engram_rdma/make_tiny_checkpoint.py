# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a tiny deterministic Engram checkpoint for transport verification."""

import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    rows = 1024
    config = {
        "text_config": {
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [rows, rows],
            "engram_head_dim": 256,
        }
    }
    (root / "config.json").write_text(json.dumps(config))
    weight_map = {}
    for layer in (1, 14):
        shard = f"model-{layer:02}.safetensors"
        weight_name = f"layers.{layer}.engram.embed.weight"
        scale_name = f"layers.{layer}.engram.embed.scale"
        weight = bytes((i + layer) % 127 for i in range(rows * 256))
        scale = bytes(127 + i % 3 for i in range(rows * 8))
        header = {
            weight_name: {
                "dtype": "F8_E4M3",
                "shape": [rows, 256],
                "data_offsets": [0, len(weight)],
            },
            scale_name: {
                "dtype": "F8_E8M0",
                "shape": [rows, 8],
                "data_offsets": [len(weight), len(weight) + len(scale)],
            },
        }
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        (root / shard).write_bytes(struct.pack("<Q", len(raw)) + raw + weight + scale)
        weight_map[weight_name] = weight_map[scale_name] = shard
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


if __name__ == "__main__":
    main()
