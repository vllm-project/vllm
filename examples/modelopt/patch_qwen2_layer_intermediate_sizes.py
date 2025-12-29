#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

from safetensors import safe_open

GATE_PROJ_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight")


def _iter_safetensors_files(
    model_dir: str,
) -> Iterator[tuple[str, list[tuple[str, int]] | None]]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        file_to_keys: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for key, filename in index.get("weight_map", {}).items():
            match = GATE_PROJ_RE.match(key)
            if not match:
                continue
            layer_idx = int(match.group(1))
            file_to_keys[filename].append((key, layer_idx))
        for filename, keys in file_to_keys.items():
            yield os.path.join(model_dir, filename), keys
        return

    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(
            "No safetensors files found. Expected model.safetensors.index.json or "
            "one or more *.safetensors files."
        )
    for filename in files:
        yield os.path.join(model_dir, filename), None


def _collect_layer_sizes(model_dir: str) -> dict[int, int]:
    layer_sizes: dict[int, int] = {}
    for path, keys in _iter_safetensors_files(model_dir):
        with safe_open(path, framework="pt") as f:
            if keys is None:
                raw_keys = list(f.keys())
                filtered_keys: list[tuple[str, int]] = []
                for key in raw_keys:
                    match = GATE_PROJ_RE.match(key)
                    if match:
                        filtered_keys.append((key, int(match.group(1))))
                iter_keys = filtered_keys
            else:
                iter_keys = keys
            for key, layer_idx in iter_keys:
                shape = f.get_slice(key).get_shape()
                if len(shape) != 2:
                    raise ValueError(f"Unexpected shape for {key}: {shape}")
                intermediate_size = int(shape[0])
                if (
                    layer_idx in layer_sizes
                    and layer_sizes[layer_idx] != intermediate_size
                ):
                    raise ValueError(
                        f"Layer {layer_idx} has inconsistent sizes: "
                        f"{layer_sizes[layer_idx]} vs {intermediate_size}"
                    )
                layer_sizes[layer_idx] = intermediate_size
    if not layer_sizes:
        raise ValueError("No gate_proj weights found in safetensors files.")
    return layer_sizes


def _build_size_list(layer_sizes: dict[int, int], num_layers: int | None) -> list[int]:
    max_layer = max(layer_sizes.keys())
    inferred_layers = max_layer + 1
    if num_layers is None:
        num_layers = inferred_layers
    if num_layers != inferred_layers:
        raise ValueError(
            f"num_hidden_layers={num_layers} does not match inferred layer count "
            f"{inferred_layers}."
        )
    sizes = [0] * num_layers
    missing: list[int] = []
    for i in range(num_layers):
        if i not in layer_sizes:
            missing.append(i)
            continue
        sizes[i] = layer_sizes[i]
    if missing:
        raise ValueError(f"Missing gate_proj weights for layers: {missing}")
    return sizes


def _summarize_sizes(sizes: Iterable[int]) -> str:
    sizes_list = list(sizes)
    unique = sorted(set(sizes_list))
    return (
        f"layers={len(sizes_list)} "
        f"min={min(sizes_list)} max={max(sizes_list)} "
        f"unique={len(unique)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch Qwen2 config.json with layer_intermediate_sizes."
    )
    parser.add_argument("--model_dir", required=True, help="HF model directory")
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    layer_sizes = _collect_layer_sizes(args.model_dir)
    num_layers = config.get("num_hidden_layers")
    sizes = _build_size_list(layer_sizes, num_layers)

    config["layer_intermediate_sizes"] = sizes

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print(_summarize_sizes(sizes))


if __name__ == "__main__":
    main()
