# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Qwen3.6 TP=1 attention geometry.
HEAD_DIM = 256
ROTARY_DIM = 64
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 262144
ROPE_THETA = 10000000.0
DTYPE = torch.bfloat16
SEED = 13
MROPE_SECTION = (11, 11, 10)
ROPE_CASES = [
    pytest.param(24, 4, None, id="rope"),
    pytest.param(16, 2, MROPE_SECTION, id="interleaved-mrope"),
]


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_rmsnorm_rope_gate Triton kernel requires CUDA/ROCm",
)
@pytest.mark.parametrize("num_q_heads,num_kv_heads,mrope_section", ROPE_CASES)
@pytest.mark.parametrize("num_tokens", [1, 4, 37])
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_matches_reference(
    default_vllm_config,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    mrope_section: tuple[int, int, int] | None,
) -> None:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.set_default_device(device)
    set_random_seed(SEED)

    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * HEAD_DIM, dtype=DTYPE, device=device
    )
    k = torch.randn(num_tokens, num_kv_heads * HEAD_DIM, dtype=DTYPE, device=device)

    q_norm = GemmaRMSNorm(HEAD_DIM, eps=RMS_NORM_EPS).to(device, dtype=DTYPE)
    k_norm = GemmaRMSNorm(HEAD_DIM, eps=RMS_NORM_EPS).to(device, dtype=DTYPE)
    q_norm.weight.normal_(std=0.1)
    k_norm.weight.normal_(std=0.1)

    q_gate_heads = q_gate.view(num_tokens, num_q_heads, 2 * HEAD_DIM)
    q = q_gate_heads[..., :HEAD_DIM]
    gate_ref = q_gate_heads[..., HEAD_DIM:].reshape(num_tokens, num_q_heads * HEAD_DIM)
    q_ref = q_norm.forward_native(q)
    k_ref = k_norm.forward_native(k.view(num_tokens, num_kv_heads, HEAD_DIM))
    assert isinstance(q_ref, torch.Tensor)
    assert isinstance(k_ref, torch.Tensor)
    q_ref = q_ref.reshape(num_tokens, num_q_heads * HEAD_DIM)
    k_ref = k_ref.reshape(num_tokens, num_kv_heads * HEAD_DIM)

    if mrope_section is None:
        rope = RotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
        ).to(device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        rope = MRotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
            mrope_section=list(mrope_section),
            mrope_interleaved=True,
        ).to(device)
        positions = torch.arange(3 * num_tokens, dtype=torch.long, device=device).view(
            3, num_tokens
        )
        assert torch.unique(positions[:, 0]).numel() == 3

    q_ref, k_ref = rope.forward_native(positions, q_ref, k_ref)
    assert k_ref is not None

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_norm.weight,
        k_norm.weight,
        rope.cos_sin_cache,
        positions,
        RMS_NORM_EPS,
        num_q_heads,
        num_kv_heads,
        HEAD_DIM,
        ROTARY_DIM,
        mrope_section=mrope_section,
        norm_beta=1.0,
    )

    # The built-in reference performs RoPE in BF16, while the fused kernel
    # promotes the BF16-normalized values to FP32 for RoPE before storing BF16.
    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)
