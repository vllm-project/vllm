# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
SEEDS = [13]
PARTIAL_ROPE = [True, False]
CUDA_DEVICES = ["cuda:0"]


def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5, 0.25])
@torch.inference_mode()
def test_fused_qk_norm_rope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    seed: int,
    rotary_ratio: float,
):
    torch.set_default_device(device)
    set_random_seed(seed)
    num_heads, num_kv_heads, head_dim = 16, 4, 128
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data
    rotary_dim = int(head_dim * rotary_ratio)
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)

    ref_result = _apply_qk_norm_rope(
        qkv=qkv_base,
        positions=positions,
        q_norm=q_norm,
        k_norm=k_norm,
        rope=rope,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    opcheck(
        torch.ops._C.fused_qk_norm_rope,
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
            positions.view(-1),
        ),
    )

    torch.ops._C.fused_qk_norm_rope(
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
        positions.view(-1),
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_fused,
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.optional
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.skipif(
    os.getenv("VLLM_TEST_LARGE_FUSED_QK_NORM_ROPE_OVERFLOW") != "1",
    reason="large 4GiB fused_qk_norm_rope overflow regression is manual",
)
@pytest.mark.parametrize(
    ("forced_token_heads_per_warp", "num_heads_q", "num_heads_k", "head_dim"),
    [
        pytest.param(1, 8, 8, 256, id="base_head_dim_256"),
        pytest.param(4, 32, 32, 64, id="ntokenheads_head_dim_64"),
    ],
)
@torch.inference_mode()
def test_fused_qk_norm_rope_large_qkv_offsets(
    default_vllm_config,
    forced_token_heads_per_warp: int,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
):
    torch.set_default_device("cuda:0")

    num_heads_v = 0
    eps = 1e-6
    rotary_dim = 64
    dtype = torch.float16

    row_dim = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    overflow_token = (2**31) // row_dim
    num_tokens = overflow_token + 2

    qkv_bytes = num_tokens * row_dim * torch.empty((), dtype=dtype).element_size()
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < qkv_bytes + 2 * 1024**3:
        pytest.skip(
            "large fused_qk_norm_rope overflow regression needs about 6GiB "
            f"free CUDA memory, got {free_bytes / 2**30:.2f}GiB"
        )

    probes = torch.tensor(
        [0, overflow_token - 1, overflow_token, overflow_token + 1],
        device="cuda",
        dtype=torch.long,
    )
    qkv = torch.empty((num_tokens, row_dim), device="cuda", dtype=dtype)
    qkv.zero_()
    # Keep the 4GiB manual test fast while making probed rows meaningful.
    qkv.index_copy_(
        0,
        probes,
        torch.randn((probes.numel(), row_dim), device="cuda", dtype=dtype),
    )

    q_weight = torch.ones((head_dim,), device="cuda", dtype=dtype)
    k_weight = torch.ones((head_dim,), device="cuda", dtype=dtype)
    cos_sin_cache = torch.zeros((1, rotary_dim), device="cuda", dtype=torch.float32)
    cos_sin_cache[:, : rotary_dim // 2] = 1.0
    position_ids = torch.zeros((num_tokens,), device="cuda", dtype=torch.int64)

    before = qkv.index_select(0, probes).clone()

    def reference_rows(rows: torch.Tensor) -> torch.Tensor:
        rows_f = rows.float()
        q_size = num_heads_q * head_dim
        k_size = num_heads_k * head_dim

        q = rows_f[:, :q_size].view(rows.shape[0], num_heads_q, head_dim)
        k = rows_f[:, q_size : q_size + k_size].view(
            rows.shape[0], num_heads_k, head_dim
        )
        v = rows_f[:, q_size + k_size :]

        q = q / torch.sqrt((q * q).mean(dim=-1, keepdim=True) + eps)
        k = k / torch.sqrt((k * k).mean(dim=-1, keepdim=True) + eps)
        return torch.cat(
            [q.reshape(rows.shape[0], q_size), k.reshape(rows.shape[0], k_size), v],
            dim=-1,
        ).to(dtype)

    expected = reference_rows(before)

    torch.ops._C.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        True,
        position_ids,
        forced_token_heads_per_warp,
    )
    torch.cuda.synchronize()

    after = qkv.index_select(0, probes)
    torch.testing.assert_close(after, expected, atol=3e-3, rtol=3e-3)
