# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from .config import MooncakeCompressionConfig


@dataclass(frozen=True)
class HeadSelection:
    low_counts: dict[int, int]
    local_orders: dict[int, tuple[int, ...]]


def _load_scores(path: str) -> np.ndarray:
    score_path = Path(path)
    suffix = score_path.suffix.lower()
    if suffix == ".npy":
        values = np.load(score_path)
    elif suffix in (".pt", ".pth"):
        loaded = torch.load(score_path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            loaded = loaded.get("scores", loaded.get("head_scores"))
        values = loaded.float().numpy()
    elif suffix == ".json":
        import json

        values = np.asarray(json.loads(score_path.read_text()), dtype=np.float32)
    else:
        values = np.genfromtxt(score_path, delimiter=",", dtype=np.float32)
        if values.ndim == 2:
            values = values[~np.all(np.isnan(values), axis=1)]
            values = values[:, ~np.all(np.isnan(values), axis=0)]
    values = np.asarray(values, dtype=np.float32)
    return values


def build_head_selection(
    specs: dict[str, KVCacheSpec],
    config: MooncakeCompressionConfig,
    tp_rank: int,
    tp_size: int,
    scores: np.ndarray | None = None,
) -> HeadSelection:
    fa_specs = [
        (name, spec)
        for name, spec in specs.items()
        if isinstance(spec, FullAttentionSpec)
    ]
    if config.split_type == "layer":
        return HeadSelection(
            low_counts={extract_layer_index(name): 0 for name, _ in fa_specs},
            local_orders={
                extract_layer_index(name): tuple(range(spec.num_kv_heads))
                for name, spec in fa_specs
            },
        )

    if scores is None:
        assert config.score_path is not None
        scores = _load_scores(config.score_path)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError(
            "Head scores must be a finite 2D array of layers and KV heads."
        )
    mask = np.zeros(scores.size, dtype=np.bool_)
    candidate_offsets: list[int] = []
    for layer_name, spec in fa_specs:
        layer_index = extract_layer_index(layer_name)
        if not 0 <= layer_index < scores.shape[0]:
            raise ValueError(f"Head scores have no row for layer {layer_index}.")
        row_start = layer_index * scores.shape[1]
        candidate_offsets.extend(range(row_start, row_start + scores.shape[1]))
    candidate_array = np.asarray(candidate_offsets, dtype=np.int64)
    low_count = round(candidate_array.size * config.hybrid_ratio)
    if low_count:
        flat_scores = scores.reshape(-1)
        selected = np.argsort(flat_scores[candidate_array], kind="stable")[:low_count]
        mask[candidate_array[selected]] = True
    mask = mask.reshape(scores.shape)

    low_counts: dict[int, int] = {}
    local_orders: dict[int, tuple[int, ...]] = {}
    for layer_name, spec in fa_specs:
        layer_index = extract_layer_index(layer_name)
        local_heads = spec.num_kv_heads
        score_heads = mask.shape[1]
        if score_heads == local_heads:
            start = 0
        elif score_heads == local_heads * tp_size:
            start = tp_rank * local_heads
        elif (
            local_heads == 1
            and 0 < score_heads < tp_size
            and tp_size % score_heads == 0
        ):
            start = tp_rank // (tp_size // score_heads)
        else:
            raise ValueError(
                f"Head score width {score_heads} is incompatible with "
                f"{local_heads} local KV heads and TP size {tp_size}."
            )
        local_mask = mask[layer_index, start : start + local_heads]
        low = np.flatnonzero(local_mask).tolist()
        high = np.flatnonzero(~local_mask).tolist()
        low_counts[layer_index] = len(low)
        local_orders[layer_index] = tuple(low + high)
    return HeadSelection(low_counts=low_counts, local_orders=local_orders)
