# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bit-exact parity test for the q-quant-separation refactor in MLAAttention.

The refactor lifts the decode q-prep (q-absorb BMM + head-dim cat + per-tensor
FP8 quant) out of ``MLAAttention.forward_impl`` and into
``MLAAttention.forward()`` so each step appears as a discrete FX node. This
test pins the math: the lifted sequence
``QAbsorb(q_nope) -> cat -> reshape -> static FP8 quant`` must produce exactly
the same ``mqa_q`` tensor as the legacy in-place path
``BMM(q_nope, W_UK_T) -> _DecodeConcatQuantFP8(ql_nope, q_pe, q_scale)``.

Two cases are exercised: ``seq_len=1`` (the production decode hot path) and
``seq_len=7`` (a small multi-row batch that catches off-by-one shape bugs).
Both run at the canonical Kimi-K2.5 / DeepSeek-V3 dimensions.
"""

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _DecodeConcatQuantFP8,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Kimi-K2.5 / DeepSeek-V3 production shapes.
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_KV_LORA_RANK = 512
_NUM_HEADS = 128
_DTYPE = torch.bfloat16


def _legacy_inplace_mqa_q(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    W_UK_T: torch.Tensor,
    q_scale: torch.Tensor,
    decode_concat_quant: _DecodeConcatQuantFP8,
) -> torch.Tensor:
    """Reproduce the BMM + cat + FP8-quant block from forward_impl exactly."""
    # forward_impl: (B, N, P) -> (N, B, P)
    q_nope_nb = q_nope.transpose(0, 1)
    N, B, _ = q_nope_nb.shape
    _, _, L = W_UK_T.shape
    ql_nope_nb = q_nope_nb.new_empty((N, B, L))
    torch.bmm(q_nope_nb, W_UK_T, out=ql_nope_nb)
    # (N, B, L) -> (B, N, L)
    ql_nope = ql_nope_nb.transpose(0, 1)
    return decode_concat_quant(ql_nope, q_pe, q_scale)


def _lifted_mqa_q(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    W_UK_T: torch.Tensor,
    q_scale: torch.Tensor,
    quant_fp8: QuantFP8,
) -> torch.Tensor:
    """Reproduce the lifted q-prep that ``forward()`` emits as discrete FX nodes."""
    # Discrete FX node #1: q-absorb BMM, BF16 output, (B, N, L).
    q_nope_nb = q_nope.transpose(0, 1)
    N, B, _ = q_nope_nb.shape
    _, _, L = W_UK_T.shape
    ql_nope_nb = q_nope_nb.new_empty((N, B, L))
    torch.bmm(q_nope_nb, W_UK_T, out=ql_nope_nb)
    ql_nope = ql_nope_nb.transpose(0, 1)
    # Discrete FX node #2: head-dim concat.
    q_full = torch.cat((ql_nope, q_pe), dim=-1)
    # Discrete FX node #3: static per-tensor FP8 quant.
    q_flat = q_full.reshape(q_full.shape[0], -1)
    mqa_q_flat, _ = quant_fp8(q_flat, q_scale)
    return mqa_q_flat.view(q_full.shape)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
@pytest.mark.parametrize("seq_len", [1, 7])
@torch.inference_mode()
def test_lifted_q_prep_matches_inplace_decode_concat_quant(
    default_vllm_config,
    seq_len: int,
) -> None:
    device = "cuda:0"
    set_random_seed(0)
    torch.set_default_device(device)

    q_nope = torch.randn(
        seq_len, _NUM_HEADS, _QK_NOPE_HEAD_DIM, dtype=_DTYPE, device=device
    )
    q_pe = torch.randn(
        seq_len, _NUM_HEADS, _QK_ROPE_HEAD_DIM, dtype=_DTYPE, device=device
    )
    # forward_impl uses W_UK_T with layout (N, P, L); same dtype as activations.
    W_UK_T = torch.randn(
        _NUM_HEADS, _QK_NOPE_HEAD_DIM, _KV_LORA_RANK, dtype=_DTYPE, device=device
    )
    q_scale = torch.tensor(0.125, dtype=torch.float32, device=device)

    # The two CustomOp instances need to share the exact same quant kernel
    # dispatch so we don't introduce a spurious mismatch.
    decode_concat_quant = _DecodeConcatQuantFP8(
        static=True,
        group_shape=GroupShape.PER_TENSOR,
        compile_native=True,
    )
    quant_fp8 = QuantFP8(
        static=True,
        group_shape=GroupShape.PER_TENSOR,
        compile_native=True,
    )

    inplace = _legacy_inplace_mqa_q(q_nope, q_pe, W_UK_T, q_scale, decode_concat_quant)
    lifted = _lifted_mqa_q(q_nope, q_pe, W_UK_T, q_scale, quant_fp8)

    # The lift is supposed to be a pure reordering of the same ops, so the
    # two FP8 tensors must be bit-exact.
    assert inplace.dtype == lifted.dtype
    assert inplace.shape == lifted.shape
    torch.testing.assert_close(
        inplace.to(torch.float32),
        lifted.to(torch.float32),
        atol=0.0,
        rtol=0.0,
    )
