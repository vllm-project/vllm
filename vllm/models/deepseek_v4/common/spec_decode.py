# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import regex as re
import torch

from vllm.config import SpeculativeConfig

_CONTEXT_WKV_RE = re.compile(r"^mtp\.(\d+)\.attn\.wkv\.(.+)$")


def duplicate_context_wkv_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_layers: int
) -> Iterable[tuple[str, torch.Tensor]]:
    """Load every draft layer's WKV into the cross-layer projection."""
    for name, weight in weights:
        yield name, weight
        match = _CONTEXT_WKV_RE.fullmatch(name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if layer_idx >= num_layers:
            continue
        stacked_weight = weight.detach()
        stacked_weight.shard_id = layer_idx
        yield f"context_wkv_proj.{match.group(2)}", stacked_weight


def needs_mtp_hidden_buffer(spec_config: SpeculativeConfig | None) -> bool:
    """Whether the drafter reads the target's pre-hc_head residual buffer.

    DSpark drafts from the target's aux hidden states, so it skips the
    (max_num_batched_tokens, hc_mult * hidden_size) buffer.
    """
    return (
        spec_config is not None
        and (spec_config.use_eagle() or spec_config.uses_draft_model())
        and not spec_config.use_dspark()
    )
