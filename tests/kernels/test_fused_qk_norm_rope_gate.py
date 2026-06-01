# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.qwen3_next import FusedQKNormRopeGate
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Real Qwen3.5 config (see Qwen3_5Config defaults in
# vllm/transformers_utils/configs/qwen3_5.py): TP=1 single-rank shapes.
NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 32768
ROPE_THETA = 10000.0

DTYPES = [torch.bfloat16, torch.float16]
SEEDS = [13]
NUM_TOKENS = [1, 4, 37]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="FusedQKNormRopeGate Triton kernel requires CUDA/ROCm",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_matches_native(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    seed: int,
    num_tokens: int,
):
    torch.set_default_device(device)
    set_random_seed(seed)

    q_gate = torch.randn(
        num_tokens, NUM_Q_HEADS * 2 * HEAD_DIM, dtype=dtype, device=device
    )
    k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=dtype, device=device)
    q_weight = torch.empty(HEAD_DIM, dtype=dtype, device=device).normal_(
        mean=0.0, std=0.1
    )
    k_weight = torch.empty(HEAD_DIM, dtype=dtype, device=device).normal_(
        mean=0.0, std=0.1
    )

    # FusedQKNormRopeGate only fuses the NeoX-style RoPE path.
    rope = RotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=ROPE_THETA,
        is_neox_style=True,
        dtype=dtype,
    ).to(device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    op = FusedQKNormRopeGate(
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        eps=RMS_NORM_EPS,
    )

    q_ref, k_ref, gate_ref = op.forward_native(
        q_gate, k, q_weight, k_weight, rope.cos_sin_cache, positions
    )
    q_out, k_out, gate_out = op.forward_cuda(
        q_gate, k, q_weight, k_weight, rope.cos_sin_cache, positions
    )

    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (1e-2, 1e-2)
    torch.testing.assert_close(q_out, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, k_ref, atol=atol, rtol=rtol)
    # gate is a verbatim copy of the source slice — must match bit-exactly.
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)
