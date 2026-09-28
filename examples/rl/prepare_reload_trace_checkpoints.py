# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare reduced dense/MoE safetensors A/B checkpoints for reload validation."""

import argparse
import json
import shutil
from pathlib import Path

import regex as re
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads((args.source / "config.json").read_text())
    if not 0 < args.layers <= config["num_hidden_layers"]:
        raise ValueError("Invalid reduced layer count")
    config["num_hidden_layers"] = args.layers
    args.output.mkdir(parents=True, exist_ok=False)
    for variant in ("a", "b"):
        target = args.output / variant
        target.mkdir()
        for path in args.source.iterdir():
            if path.is_file() and not (
                path.name.endswith((".safetensors", ".bin", ".index.json"))
                or path.name == "config.json"
            ):
                shutil.copy2(path, target / path.name)
        (target / "config.json").write_text(json.dumps(config, indent=2))
    weight_map = {}
    changed = []
    total_bytes = 0
    for path in sorted(args.source.glob("*.safetensors")):
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in sorted(source.keys()):
                layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if layer and int(layer.group(1)) >= args.layers:
                    continue
                tensors[name] = source.get_tensor(name)
                weight_map[name] = path.name
                total_bytes += tensors[name].numel() * tensors[name].element_size()
            if not tensors:
                continue
            save_file(tensors, args.output / "a" / path.name, metadata={"format": "pt"})
            for name, tensor in tensors.items():
                if name.endswith(".down_proj.weight"):
                    tensors[name] = (tensor.float() * 0.5).to(tensor.dtype)
                    changed.append(name)
            save_file(tensors, args.output / "b" / path.name, metadata={"format": "pt"})
    if not changed:
        raise ValueError("No down_proj weights were changed")
    for variant in ("a", "b"):
        (args.output / variant / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
            )
        )
    provenance = {
        "source": str(args.source.resolve()),
        "layers": args.layers,
        "changed_weights": changed,
        "operation": (
            "B down_proj weights = A weights * 0.5; all other tensors unchanged"
        ),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
