# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rowwise FP8 (e4m3) quantization of a speculative-decoding draft LM head.

Drafters that share the target model's LM head (e.g. DSpark,
``has_own_lm_head=False``) pay a full ``[hidden, vocab]`` bf16 GEMM per draft
step just to propose tokens. On bandwidth-bound GPUs that weight read
dominates the draft loop. These helpers build a one-time rowwise-fp8 copy of
the (vocab-sharded) head and compute draft logits with a dynamic per-token
fp8 GEMM, halving the weight traffic.

This is draft-time only by construction: the verify pass never sees the fp8
weights, so accepted outputs are bitwise unchanged. The quantized logits are
used for the draft argmax/top-k proposal; a rare argmax flip only costs a
rejected draft token, never an incorrect sample.
"""

from typing import NamedTuple

import torch

# Finite max of float8_e4m3fn.
_FP8_MAX = 448.0


class Fp8DraftHead(NamedTuple):
    """Rowwise-quantized copy of a (possibly vocab-sharded) LM head."""

    # [num_local_vocab, hidden] fp8_e4m3, each row scaled to [-448, 448].
    weight_fp8: torch.Tensor
    # [1, num_local_vocab] dequant scale per output row (rowmax / 448),
    # kept in the activation dtype so the epilogue multiplies stay cheap.
    row_scale: torch.Tensor
    # fp32 scalar 1.0 for torch._scaled_mm (scaling is applied manually in
    # the epilogue because both operands use dynamic per-row scales).
    unit_scale: torch.Tensor


def fp8_draft_head_supported(device: torch.device | None = None) -> bool:
    """torch._scaled_mm needs fp8 tensor cores (SM89+) on CUDA devices."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def quantize_draft_head(weight: torch.Tensor) -> Fp8DraftHead:
    """Quantize a ``[num_local_vocab, hidden]`` head weight rowwise to fp8.

    Per-row (per vocab entry) symmetric scaling: ``w8 = w * (448 / rowmax)``
    stored as fp8_e4m3, ``row_scale = rowmax / 448`` for the epilogue. For a
    vocab-parallel (sharded) head, pass the local shard; row scales are
    per-local-row, so gather/argmax semantics downstream are unchanged.
    """
    with torch.no_grad():
        w = weight.detach()
        row_max = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6)
        weight_fp8 = (w.float() * (_FP8_MAX / row_max)).to(torch.float8_e4m3fn)
        row_scale = (row_max / _FP8_MAX).to(w.dtype).reshape(1, -1)
        unit_scale = torch.ones(1, dtype=torch.float32, device=w.device)
    return Fp8DraftHead(weight_fp8, row_scale, unit_scale)


def fp8_draft_head_logits(
    hidden_states: torch.Tensor,
    head: Fp8DraftHead,
) -> torch.Tensor:
    """Local (shard) draft logits via dynamic-fp8 x rowwise-fp8 GEMM.

    Activations are quantized with a per-token amax scale, then
    ``logits = _scaled_mm(a8, w8.T) * row_scale * (amax / 448)``. Output
    dtype and shape match ``lm_head.quant_method.apply``: the caller is
    responsible for the same TP gather / vocab-padding slice it would apply
    to the unquantized local logits. Contains no data-dependent control
    flow or allocations beyond the GEMM output, so it is safe to run inside
    a captured CUDA graph as long as ``head`` was materialized beforehand.
    """
    act_max = hidden_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    act_fp8 = (hidden_states * (_FP8_MAX / act_max)).to(torch.float8_e4m3fn)
    logits = torch._scaled_mm(
        act_fp8,
        head.weight_fp8.t(),
        scale_a=head.unit_scale,
        scale_b=head.unit_scale,
        out_dtype=hidden_states.dtype,
    )
    logits = logits * head.row_scale
    logits = logits * (act_max / _FP8_MAX).to(hidden_states.dtype)
    return logits
