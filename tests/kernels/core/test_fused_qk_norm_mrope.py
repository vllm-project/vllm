# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
MROPE_INTERLEAVED = [False, True]  # Qwen3-VL uses interleaved
EPS_VALUES = [1e-5, 1e-6]
# -1 auto-selects; 1 is the base kernel, 2/4/8 the multi-token-head kernel.
TOKEN_HEADS_PER_WARP = [-1, 1, 2, 4, 8]
SEEDS = [13]
CUDA_DEVICES = ["cuda:0"]

# (num_heads, num_kv_heads, head_dim, mrope_section). mrope_section sums to
# head_dim // 2 (full rotary).
HEAD_CONFIGS = [
    (16, 4, 128, [16, 24, 24]),
    (16, 2, 64, [8, 12, 12]),
    (8, 8, 256, [32, 48, 48]),
]


def _apply_qk_norm_mrope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: MRotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    assert isinstance(q_by_head, torch.Tensor)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    assert isinstance(k_by_head, torch.Tensor)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_mrope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("mrope_interleaved", MROPE_INTERLEAVED)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("token_heads_per_warp", TOKEN_HEADS_PER_WARP)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,mrope_section", HEAD_CONFIGS)
@torch.inference_mode()
def test_fused_qk_norm_mrope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    mrope_interleaved: bool,
    eps: float,
    token_heads_per_warp: int,
    seed: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mrope_section: list[int],
):
    torch.set_default_device(device)
    set_random_seed(seed)
    # Odd count spanning several blocks, so partial blocks and chunks are hit.
    num_tokens = 67

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    # mRoPE positions: [3, num_tokens] (time/height/width), independent streams.
    # Use a strided (non-contiguous) view like Qwen3-VL passes at runtime, to
    # also exercise the kernel's internal contiguous() handling.
    positions = torch.randint(
        0, 1000, (3, num_tokens * 2), dtype=torch.long, device=device
    )[:, :num_tokens]

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data

    def make_rope(rope_dtype: torch.dtype) -> MRotaryEmbedding:
        return MRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=is_neox,
            dtype=rope_dtype,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        ).to(device)

    rope = make_rope(dtype)

    # The reference runs in fp32: a half-precision reference rounds between
    # the norm and the rotation and is itself off by more than the tolerance
    # on a few elements once there are enough of them.
    ref_q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=torch.float32)
    ref_k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=torch.float32)
    ref_q_norm.weight.data.copy_(q_weight)
    ref_k_norm.weight.data.copy_(k_weight)
    ref_result = _apply_qk_norm_mrope(
        qkv=qkv_base.float(),
        positions=positions,
        q_norm=ref_q_norm,
        k_norm=ref_k_norm,
        rope=make_rope(torch.float32),
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    opcheck(
        torch.ops._C.fused_qk_norm_mrope,
        (
            qkv_fused.clone(),
            num_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions,
            mrope_section[0],
            mrope_section[1],
            mrope_interleaved,
            token_heads_per_warp,
        ),
    )

    ops.fused_qk_norm_mrope(
        qkv_fused,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        is_neox,
        positions,
        mrope_section[0],
        mrope_section[1],
        mrope_interleaved,
        token_heads_per_warp,
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (4e-2, 4e-2)

    torch.testing.assert_close(
        qkv_fused.float(),
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )

    # Every kernel variant must produce the same bits, so the output does not
    # depend on which one the token count dispatches to.
    qkv_base_kernel = qkv_base.clone()
    ops.fused_qk_norm_mrope(
        qkv_base_kernel,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        is_neox,
        positions,
        mrope_section[0],
        mrope_section[1],
        mrope_interleaved,
        1,
    )
    assert torch.equal(qkv_fused, qkv_base_kernel)
