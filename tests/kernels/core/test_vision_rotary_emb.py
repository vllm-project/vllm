# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol

# yapf: disable
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    apply_rotary_pos_emb_vision_2c,
)

# yapf: enable
from vllm.platforms import current_platform

DTYPES = [torch.half, torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 120, 256]
NUM_HEADS = [8, 16]
BATCH_SIZES = [1, 2]
SEQ_LENS = [1024, 4096, 16384]
SEEDS = [0]
CUDA_DEVICES = ["cuda"]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_vision_rotary(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    # 2c Triton kernel only supports CUDA
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    # Qwen2-VL uses rotary over half the head dim
    rotary_dim = head_size // 2
    rope = Qwen2VisionRotaryEmbedding(rotary_dim)
    rope = rope.to(dtype=torch.float32, device=torch.get_default_device())
    freqs = rope(seq_len)  # (seqlen, rotary_dim/2)

    # Inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
    k = torch.randn_like(q)

    # 1c path: apply to q and k separately
    out_q_1c = apply_rotary_pos_emb_vision(q, freqs)
    out_k_1c = apply_rotary_pos_emb_vision(k, freqs)

    # 2c path: apply to q and k together
    out_q_2c, out_k_2c = apply_rotary_pos_emb_vision_2c(q, k, freqs)

    torch.testing.assert_close(
        out_q_2c,
        out_q_1c,
        atol=get_default_atol(out_q_2c),
        rtol=get_default_rtol(out_q_2c),
    )
    torch.testing.assert_close(
        out_k_2c,
        out_k_1c,
        atol=get_default_atol(out_k_2c),
        rtol=get_default_rtol(out_k_2c),
    )
