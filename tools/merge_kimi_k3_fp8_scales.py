#!/usr/bin/env python3
"""Merge Kimi-K3 FP8 calibration rank shards into safetensors."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    shard_paths = sorted(args.input_dir.glob("kimi-k3-fp8-scales-pp*-tp*.json"))
    if not shard_paths:
        raise ValueError(f"No calibration shards found under {args.input_dir}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    reference = shards[0]
    common_fields = (
        "schema",
        "model",
        "revision",
        "checkpoint_id",
        "calibration_id",
        "tp_size",
        "pp_size",
        "fp8_dtype",
        "cache_mode",
        "local_heads",
        "qk_head_dim",
        "v_head_dim",
        "margin",
    )
    for shard in shards[1:]:
        for field in common_fields:
            if shard[field] != reference[field]:
                raise ValueError(f"Calibration shard mismatch for {field}")
    if not reference["checkpoint_id"] or not reference["calibration_id"]:
        raise ValueError("Calibration shards require checkpoint and run identities")

    tp_size = int(reference["tp_size"])
    expected_ranks = {
        (pp_rank, tp_rank)
        for pp_rank in range(int(reference["pp_size"]))
        for tp_rank in range(tp_size)
    }
    actual_ranks = {(int(shard["pp_rank"]), int(shard["tp_rank"])) for shard in shards}
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"Expected calibration ranks {sorted(expected_ranks)}, "
            f"got {sorted(actual_ranks)}"
        )

    per_layer: dict[int, dict[int, dict]] = {}
    for shard in shards:
        tp_rank = int(shard["tp_rank"])
        for layer_idx_text, layer in shard["layers"].items():
            layer_idx = int(layer_idx_text)
            ranks = per_layer.setdefault(layer_idx, {})
            if tp_rank in ranks:
                raise ValueError(
                    f"Duplicate calibration for layer {layer_idx}, TP rank {tp_rank}"
                )
            ranks[tp_rank] = layer

    fp8_max = torch.finfo(torch.float8_e4m3fnuz).max
    margin = float(reference["margin"])
    local_heads = int(reference["local_heads"])
    tensors = {}
    for layer_idx, ranks in sorted(per_layer.items()):
        if set(ranks) != set(range(tp_size)):
            raise ValueError(
                f"Incomplete TP calibration for layer {layer_idx}: "
                f"got ranks {sorted(ranks)}"
            )
        for tensor_name in ("q", "k", "v"):
            rank_maxima = []
            for rank in range(tp_size):
                maximum = torch.tensor(
                    ranks[rank][f"{tensor_name}_amax"],
                    dtype=torch.float32,
                )
                if maximum.shape != (local_heads,):
                    raise ValueError(
                        f"Invalid {tensor_name} maxima shape for layer {layer_idx}, "
                        f"TP rank {rank}: {tuple(maximum.shape)}"
                    )
                if not torch.isfinite(maximum).all() or not (maximum > 0).all():
                    raise ValueError(
                        f"Invalid {tensor_name} maxima for layer {layer_idx}, "
                        f"TP rank {rank}"
                    )
                rank_maxima.append(maximum)
            maxima = torch.cat(rank_maxima)
            tensors[f"layers.{layer_idx}.{tensor_name}_descale"] = (
                maxima * margin / fp8_max
            ).clamp_min(1.0e-12)

    metadata = {
        "schema": str(reference["schema"]),
        "model": reference["model"],
        "revision": reference["revision"],
        "checkpoint_id": reference["checkpoint_id"],
        "calibration_id": reference["calibration_id"],
        "tp_size": str(tp_size),
        "pp_size": str(reference["pp_size"]),
        "num_layers": str(len(per_layer)),
        "fp8_dtype": reference["fp8_dtype"],
        "cache_mode": reference["cache_mode"],
        "local_heads": str(local_heads),
        "qk_head_dim": str(reference["qk_head_dim"]),
        "v_head_dim": str(reference["v_head_dim"]),
        "margin": str(margin),
        "calibration_shards": str(len(shards)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    print(f"Saved {len(per_layer)} calibrated layers to {args.output}")


if __name__ == "__main__":
    main()
