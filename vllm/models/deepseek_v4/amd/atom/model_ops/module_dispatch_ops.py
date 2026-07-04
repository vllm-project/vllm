# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Custom ops dispatching to module methods via `static_forward_context`.

Each op below takes a `layer_name: str` and looks up the owning module from
`compilation_config.static_forward_context[layer_name]`. The module's
internal state (sub-modules, persistent buffers, per-fwd metadata fetched
through `get_forward_context()`) stays out of the op's argument list, which:

  - hides dynamic-shape internals from Dynamo / fake-tensor mode (no graph
    break, no per-shape recompile)
  - bypasses functionalization for buffer mutations — the buffers aren't in
    the op's signature, so the pass needn't insert defensive clones
  - leaves CUDAGraph capture transparent — the op still launches the same
    CUDA work on the stream; only the Python dispatch is opaque to compile

Caller contract (per op): the module registered at `layer_name` must expose
the methods listed in each op's docstring.

Currently registered:
  - torch.ops.aiter.maybe_dual_stream_forward — V2/V3.2/V4 MoE
  - torch.ops.aiter.indexer_score_topk       — V4 sparse indexer
"""

import torch

from vllm.models.deepseek_v4.amd.atom.config import get_current_atom_config
from vllm.models.deepseek_v4.amd.atom.utils import envs
from vllm.models.deepseek_v4.amd.atom.utils.custom_register import direct_register_custom_op

# ---------------------------------------------------------------------------
# Dual-stream MoE dispatch (V2 / V3.2 / V4)
# ---------------------------------------------------------------------------
#
# Caller contract (the MoE module looked up by `layer_name`):
#   - `_use_dual_stream: bool`
#   - `single_stream_moe_forward(hidden_states) -> Tensor`
#   - `dual_stream_moe_forward(hidden_states) -> Tensor`
#
# Per-token gating: decode benefits from dual-stream, prefill doesn't —
# threshold from `envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD`.


def maybe_dual_stream_forward(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    threshold = envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD
    num_tokens = hidden_states.shape[0]
    if self._use_dual_stream and 0 < num_tokens <= threshold:
        return self.dual_stream_moe_forward(hidden_states)
    return self.single_stream_moe_forward(hidden_states)


def _maybe_dual_stream_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="maybe_dual_stream_forward",
    op_func=maybe_dual_stream_forward,
    # Op returns a fresh tensor; never writes into `hidden_states`. Declaring
    # `mutates_args=["hidden_states"]` (the V2 original) misleads the
    # functionalization pass into inserting defensive input clones.
    mutates_args=(),
    fake_impl=_maybe_dual_stream_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# Sparse indexer score + top-k (V2/V3.2/V4)
# ---------------------------------------------------------------------------
#
# Caller contract (the Indexer module looked up by `layer_name`):
#   - `indexer_score_topk(q_fp8, weights, topk) -> Tensor`   — real impl,
#     must return `[total_tokens, topk] int32` indices
#
# `topk` is on the op signature (not derived from the module) so the fake
# impl can size the output without any module lookup.
#
# Other inputs (block_tables, KV cache, per-fwd metadata) are read by the
# module from `self` or `get_forward_context().attn_metadata`.
#
# Why opaque (same rationale for V2 and V4):
#   - prefill paths allocate scratch tensors with shapes that depend on
#     per-fwd `total_committed` / `total_kv` — Dynamo's fake-tensor pass
#     can't size them without a graph break.
#   - decode paths write into module-owned buffers; keeping the buffers
#     out of the op signature avoids functionalization clones.


def indexer_score_topk(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    indexer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return indexer.indexer_score_topk(q_fp8, weights, topk)


def _indexer_score_topk_fake(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    return torch.empty(
        (q_fp8.shape[0], topk),
        dtype=torch.int32,
        device=q_fp8.device,
    )


direct_register_custom_op(
    op_name="indexer_score_topk",
    op_func=indexer_score_topk,
    # Output is a fresh tensor (per module contract). Internal buffer
    # mutations on the module are looked up via `layer_name`, not passed
    # in, so functionalization stays unaware and skips defensive clones.
    mutates_args=(),
    fake_impl=_indexer_score_topk_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
