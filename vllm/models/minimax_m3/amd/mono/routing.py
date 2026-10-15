# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-uniform per-step routing and cache identity tracking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import CUDAGraphMode
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata

from .spec import MiniMaxM3MonoSpec


@dataclass(frozen=True)
class StepDecision:
    use_mono: bool
    reason: str = ""


def decide_step(
    spec: MiniMaxM3MonoSpec,
    rows: int,
    graph_mode: CUDAGraphMode,
    metadata: object,
) -> StepDecision:
    if rows not in spec.widths:
        return StepDecision(False, f"unsupported graph width {rows}")
    if graph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        return StepDecision(False, f"unsupported CUDA graph mode {graph_mode}")
    if not isinstance(metadata, MiniMaxM3SparseMetadata):
        return StepDecision(False, "sparse attention metadata missing")
    if metadata.decode is None or metadata.num_decodes == 0:
        return StepDecision(False, "not a decode step")
    if metadata.num_prefills != 0 or metadata.num_prefill_tokens != 0:
        return StepDecision(False, "mixed prefill/decode step")
    if not 0 < metadata.num_decode_tokens <= rows:
        return StepDecision(False, "decode token count exceeds physical rows")
    if metadata.slot_mapping.numel() < rows:
        return StepDecision(False, "slot mapping does not cover physical rows")
    return StepDecision(True)


def tensor_identity(
    tensor: torch.Tensor,
) -> tuple[int, int, torch.dtype, tuple[int, ...]]:
    return tensor.data_ptr(), tensor.numel(), tensor.dtype, tensor.stride()
