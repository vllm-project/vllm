# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare reduced A/B checkpoints for DeepSeek V4/V4.1 reload validation."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    config = json.loads((args.source / "config.json").read_text())
    if not 0 < args.layers <= config["num_hidden_layers"]:
        raise ValueError("Invalid reduced layer count")
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
        reduced_config = dict(config)
        reduced_config["num_hidden_layers"] = args.layers
        (target / "config.json").write_text(json.dumps(reduced_config, indent=2) + "\n")

    weight_map: dict[str, str] = {}
    changed: list[str] = []
    total_bytes = 0
    for path in sorted(args.source.glob("*.safetensors")):
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in sorted(source.keys()):
                marker = ".layers."
                if marker in name:
                    layer_index = int(name.split(marker, 1)[1].split(".", 1)[0])
                    if layer_index >= args.layers:
                        continue
                tensor = source.get_tensor(name)
                tensors[name] = tensor
                weight_map[name] = path.name
                total_bytes += tensor.numel() * tensor.element_size()
        if not tensors:
            continue

        save_file(
            tensors,
            args.output / "a" / path.name,
            metadata={"format": "pt"},
        )
        variant_b = dict(tensors)
        for name, tensor in tensors.items():
            if ".experts." in name and name.endswith(".w2.weight"):
                variant_b[name] = (tensor.float() * args.scale).to(tensor.dtype)
                changed.append(name)
        save_file(
            variant_b,
            args.output / "b" / path.name,
            metadata={"format": "pt"},
        )

    if not changed:
        raise ValueError("No DeepSeek expert w2 weights were changed")
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    for variant in ("a", "b"):
        (args.output / variant / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
    provenance = {
        "source": str(args.source.resolve()),
        "layers": args.layers,
        "changed_weights": changed,
        "operation": f"B expert w2 weights = A weights * {args.scale}",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
