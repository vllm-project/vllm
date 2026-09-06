# SPDX-License-Identifier: Apache-2.0
"""Prepare genuine reduced block-FP8 MoE checkpoints for day0 A/B validation."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    print(sys.executable, sys.prefix, flush=True)
    config = json.loads((args.source / "config.json").read_text())
    assert config["model_type"] == "qwen3_moe"
    assert config["quantization_config"]["weight_block_size"] == [128, 128]
    assert 0 < args.layers < config["num_hidden_layers"]
    original_layers = config["num_hidden_layers"]
    config["num_hidden_layers"] = args.layers
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    selected = {}
    for name, shard in index["weight_map"].items():
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None or int(match[1]) < args.layers:
            selected.setdefault(shard, []).append(name)
    args.output.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for shard, names in selected.items():
        with safe_open(str(args.source / shard), framework="pt", device="cpu") as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name)
    changed = [name for name in tensors if ".experts." in name and name.endswith(".down_proj.weight")]
    assert len(changed) == args.layers * config["num_experts"]
    for variant in ("a", "b"):
        target = args.output / variant
        target.mkdir()
        for source in args.source.iterdir():
            if source.is_file() and source.suffix in (".json", ".txt", ".model"):
                if source.name not in ("config.json", "model.safetensors.index.json"):
                    shutil.copy2(source, target / source.name)
        (target / "config.json").write_text(json.dumps(config, indent=2))
        if variant == "b":
            for name in changed:
                value = tensors[name]
                tensors[name] = (value.float() * 0.5).to(value.dtype)
        save_file(tensors, str(target / "model.safetensors"))
    manifest = {"source": str(args.source.resolve()), "original_layers": original_layers,
                "layers": args.layers, "quantization": config["quantization_config"],
                "tensor_count": len(tensors), "changed_names": changed,
                "change": "B halves stored FP8 MoE down_proj weights; scales unchanged"}
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {args.layers} layers, {len(tensors)} tensors; B changes {len(changed)} MoE weights", flush=True)


if __name__ == "__main__":
    main()
