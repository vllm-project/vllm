# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-uniform per-step routing and cache identity tracking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import CUDAGraphMode
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata

from .spec import DecodeConfig


@dataclass(frozen=True)
class StepDecision:
    use_mono: bool
    reason: str = ""


def decide_step(
    spec: DecodeConfig,
    rows: int,
    graph_mode: CUDAGraphMode,
    metadata: object,
) -> StepDecision:
    if rows not in spec.widths:
        return StepDecision(False, f"unsupported physical row count {rows}")
    if graph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        # a PIECEWISE capture is keyed by token count alone and replays mixed steps
        return StepDecision(False, f"{graph_mode.name} graph step")
    if not isinstance(metadata, MiniMaxM3SparseMetadata):
        return StepDecision(False, "sparse attention metadata missing")
    if metadata.decode is None or metadata.num_decodes == 0:
        return StepDecision(False, "not a decode step")
    if metadata.num_prefills != 0 or metadata.num_prefill_tokens != 0:
        return StepDecision(False, "mixed prefill/decode step")
    if metadata.decode.decode_query_len != 1:
        return StepDecision(False, "speculative verification is unsupported")
    if not 0 < metadata.num_decode_tokens <= rows:
        return StepDecision(False, "decode token count exceeds physical rows")
    if metadata.num_decodes != metadata.num_decode_tokens:
        return StepDecision(False, "plain decode requires one token per request")
    if (
        metadata.decode.seq_lens.numel() < metadata.num_decodes
        or metadata.decode.block_table.shape[0] < metadata.num_decodes
    ):
        return StepDecision(False, "decode tables do not cover every request")
    if (
        metadata.slot_mapping.dtype != torch.int64
        or metadata.slot_mapping.numel() < rows
    ):
        return StepDecision(False, "slot mapping does not cover physical rows")
    return StepDecision(True)
