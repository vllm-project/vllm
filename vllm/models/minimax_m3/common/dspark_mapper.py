# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 nested DSpark checkpoint name mapping.

Official ``DSparkMiniMaxDraftModel`` shards live under
``language_model.model.dspark.layers.{i}.decoder_layer.*``. vLLM's draft module
expects ``layers.{i}.*`` with fused qkv / gate_up projections and ModelOpt
``weight_scale`` (not ``weight_scale_inv``).

This module is importable without the rest of vLLM so CPU tests can lock the
mapping independently of Attention / msgspec / uvloop.
"""

from __future__ import annotations

import regex as re

DSPARK_ORIG_TO_NEW_REGEX: dict[re.Pattern[str], str] = {
    re.compile(r"^language_model\.model\.dspark\."): "",
    re.compile(r"\.decoder_layer\."): ".",
    re.compile(r"(?<![A-Za-z0-9_])final_norm(?=\.|$)"): "norm",
    re.compile(r"weight_scale_inv"): "weight_scale",
}

DSPARK_ORIG_TO_NEW_STACKED: dict[str, tuple[str, str | int]] = {
    ".q_proj": (".qkv_proj", "q"),
    ".k_proj": (".qkv_proj", "k"),
    ".v_proj": (".qkv_proj", "v"),
    ".gate_proj": (".gate_up_proj", 0),
    ".up_proj": (".gate_up_proj", 1),
}


def map_dspark_weight_name(name: str) -> tuple[str, str | int | None] | None:
    """Map one official DSpark tensor name to the vLLM draft parameter.

    Returns ``(mapped_name, shard_id)``. ``shard_id`` is set for stacked q/k/v
    and gate/up weights.
    """
    mapped = name
    for pattern, replacement in DSPARK_ORIG_TO_NEW_REGEX.items():
        mapped = pattern.sub(replacement, mapped)

    shard_id: str | int | None = None
    for substr, (new_substr, new_shard_id) in DSPARK_ORIG_TO_NEW_STACKED.items():
        if substr in mapped:
            mapped = mapped.replace(substr, new_substr, 1)
            shard_id = new_shard_id
            break
    return mapped, shard_id
