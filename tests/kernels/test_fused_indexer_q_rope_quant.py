# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for fused_indexer_q_rope_quant.

Compares the fused Triton kernel against the unfused reference flow used by
the DeepseekV4 indexer in model_tracking:
    q_rot = ops.rotary_embedding(positions, q, None, head_dim, cos_sin_cache,
                                 is_neox_style=False,
                                 rope_dim_offset=head_dim - rope_dim)
    q_fp8, q_scale = per_token_group_quant_fp8(q_rot, head_dim, use_ue8m0=True)
    weights_out = weights * q_scale * softmax_scale * head_scale

Expects bit-exact equality on both q_fp8 and weights_out.
"""

import contextlib
import importlib
from unittest import mock

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.models.deepseek_v4.common.ops import fused_indexer_q_rope_quant
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

HEAD_DIM = 128
ROPE_DIM = 64
MAX_POS = 4096

CUTEDSL_MODULE = "vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl"


def quantize_to_mxfp4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP4 quantization.

    Args:
        x: [..., head_dim] where head_dim is divisible by 32
    Returns:
        packed: [..., head_dim//2]  uint8   2 E2M1 nibbles/byte, low nibble = even index
        scales: [..., head_dim//32] uint8   1 ue8m0 byte
    """
    MXFP4_BLOCK_SIZE = 32
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    n_blocks = head_dim // MXFP4_BLOCK_SIZE

    x_f32 = x.float().reshape(-1, n_blocks, MXFP4_BLOCK_SIZE)

    # Per-block ue8m0 scale: 2^ceil(log2(amax / 6.0)), stored as byte = exp + 127
    # 6 * 2^-126 is from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py#L163
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=6 * (2**-126))
    log2_ratio = (amax * (1.0 / 6.0)).log2().ceil().clamp(-127.0, 127.0)
    scale = log2_ratio.exp2()
    ue8m0 = (log2_ratio + 127.0).to(torch.uint8)  # [*, n_blocks]

    # E2M1 round-to-nearest-even: midpoints round to the even code.
    # E2M1 values: [0.00, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00]
    # boundaries:  [   0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00]
    x_scaled = (x_f32 / scale).clamp(-6.0, 6.0)
    abs_x = x_scaled.abs()
    code = torch.zeros_like(abs_x, dtype=torch.int32)
    code = torch.where(abs_x > 0.25, 1, code)
    code = torch.where(abs_x >= 0.75, 2, code)
    code = torch.where(abs_x > 1.25, 3, code)
    code = torch.where(abs_x >= 1.75, 4, code)
    code = torch.where(abs_x > 2.5, 5, code)
    code = torch.where(abs_x >= 3.5, 6, code)
    code = torch.where(abs_x > 5.0, 7, code)
    sign = ((x_scaled.view(torch.int32) >> 31) & 1).to(torch.uint8)
    nibble = code.to(torch.uint8) | (sign << 3)

    # Pack: even-index element → low nibble, odd-index → high nibble
    nibble_flat = nibble.reshape(-1, head_dim)
    packed = (nibble_flat[:, 0::2] | (nibble_flat[:, 1::2] << 4)).contiguous()
    packed = packed.reshape(*orig_shape[:-1], head_dim // 2)

    scales = ue8m0.view(*orig_shape[:-1], n_blocks)
    return packed, scales


def _reference(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    n_head: int,
    use_fp4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = q.clone()
    ops.rotary_embedding(
        positions,
        q_rot,
        None,
        HEAD_DIM,
        cos_sin_cache,
        False,  # is_neox_style=False → GPT-J interleaved
        HEAD_DIM - ROPE_DIM,  # rope_dim_offset → rotate the tail
        False,
    )

    if use_fp4:
        q_packed, ue8m0 = quantize_to_mxfp4(q_rot.view(-1, n_head, HEAD_DIM))
        # Pack 4 ue8m0 bytes into 1 int32
        q_scale = ue8m0.view(torch.int32).squeeze(-1)
        # FP4 path: q_scale stays separate (cannot be folded into a per-token scalar)
        weights_out = weights.to(torch.float32) * softmax_scale * head_scale
        return (q_packed, q_scale), weights_out

    else:
        q_fp8, q_scale = per_token_group_quant_fp8(
            q_rot.view(-1, HEAD_DIM).contiguous(),
            HEAD_DIM,
            use_ue8m0=True,
        )
        q_fp8 = q_fp8.view(-1, n_head, HEAD_DIM)
        q_scale = q_scale.view(-1, n_head)

        weights_out = weights.to(torch.float32) * q_scale * softmax_scale * head_scale
        return q_fp8, weights_out


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 257, 1023])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "use_fp4",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not (
                    current_platform.is_cuda()
                    and current_platform.is_device_capability_family(100)
                ),
                reason="MXFP4 indexer cache requires an SM100-family GPU",
            ),
        ),
    ],
)
@pytest.mark.parametrize("use_cutedsl", [False, True])
@pytest.mark.parametrize("n_head", [32, 64])
@torch.inference_mode()
def test_fused_indexer_q_rope_quant_matches_unfused(
    num_tokens, cache_dtype, use_fp4, use_cutedsl, n_head
):
    if use_cutedsl and not has_cutedsl():
        pytest.skip("cutedsl (cutlass) not installed")

    device = "cuda"
    torch.manual_seed(0)

    q = torch.randn(num_tokens, n_head, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=cache_dtype, device=device)
    weights = torch.randn(num_tokens, n_head, dtype=torch.bfloat16, device=device)
    softmax_scale = HEAD_DIM**-0.5
    # head_scale must be an exact power of two for bit-exactness to be a fair
    # expectation: the triton kernel multiplies by softmax_scale and head_scale
    # separately (two roundings) while the cutedsl kernels fold them into one
    # fp32 scalar first (one rounding). Those agree only when one factor is a
    # pow2. n_head**-0.5 is a pow2 at 64 heads but not at 32.
    head_scale = 0.125

    q_quant_ref, weights_ref = _reference(
        positions,
        q,
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        n_head,
        use_fp4,
    )
    # use_cutedsl=False: force the triton path even when cutedsl is installed
    # by patching the dispatcher's has_cutedsl() binding to return False.
    cutedsl_patch = (
        mock.patch(
            "vllm.models.deepseek_v4.common.ops.fused_indexer_q.has_cutedsl",
            return_value=False,
        )
        if not use_cutedsl
        else contextlib.nullcontext()
    )
    with cutedsl_patch:
        q_quant_fused, weights_fused = fused_indexer_q_rope_quant(
            positions,
            q.clone(),
            cos_sin_cache,
            weights,
            softmax_scale,
            head_scale,
            use_fp4,
        )

    if use_fp4:
        q_quant_ref, q_scale_ref = q_quant_ref
        q_quant_fused, q_scale_fused = q_quant_fused

        assert torch.equal(q_scale_ref, q_scale_fused), (
            f"q_scale mismatch: "
            f"{(q_scale_ref != q_scale_fused).sum().item()} "
            f"/ {q_scale_ref.numel()} bytes differ"
        )

    # fp8 tensors aren't directly comparable via torch.equal — reinterpret as int8.
    ref_bits = q_quant_ref.view(torch.int8)
    fused_bits = q_quant_fused.view(torch.int8)
    assert torch.equal(ref_bits, fused_bits), (
        f"q_quant_fused mismatch: "
        f"{(ref_bits != fused_bits).sum().item()} / {ref_bits.numel()} bytes differ"
    )

    assert weights_fused.dtype == torch.float32
    assert torch.equal(weights_ref, weights_fused), (
        f"weights mismatch: max abs diff "
        f"{(weights_ref - weights_fused).abs().max().item()}"
    )


@pytest.mark.skipif(not has_cutedsl(), reason="cutedsl (cutlass) not installed")
@pytest.mark.parametrize(
    "use_fp4",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not (
                    current_platform.is_cuda()
                    and current_platform.is_device_capability_family(100)
                ),
                reason="MXFP4 indexer cache requires an SM100-family GPU",
            ),
        ),
    ],
)
@pytest.mark.parametrize("n_head", [32, 64])
@torch.inference_mode()
def test_cutedsl_indexer_q_writes_stay_within_num_tokens(use_fp4, n_head):
    """The cutedsl kernels must not store past ``num_tokens``.

    The launcher rounds ``num_tokens * threads_per_token`` up to whole 128-thread
    blocks, so the trailing block is only partially in range unless
    ``threads_per_token`` is itself a multiple of the block size. With 32 heads
    and coarsen=4 (picked for >= 512 tokens) a token needs 64 threads, so an odd
    token count leaves half of the last block addressing token ``num_tokens``.
    Every output gets one extra sentinel row that the kernel must leave alone.
    """
    mod = importlib.import_module(CUTEDSL_MODULE)
    device = "cuda"
    torch.manual_seed(0)
    num_tokens = 935

    sentinel_u8, sentinel_f32 = 0x55, -12345.0
    # positions/q are padded too, so an out-of-bounds *load* stays addressable
    # and the assertions below are about stores only.
    positions = torch.zeros(num_tokens + 1, dtype=torch.int64, device=device)
    q = torch.randn(
        num_tokens + 1, n_head, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=torch.float32, device=device)
    weights = torch.randn(num_tokens + 1, n_head, dtype=torch.bfloat16, device=device)
    weights_out = torch.full(
        (num_tokens + 1, n_head), sentinel_f32, dtype=torch.float32, device=device
    )
    outputs = {"weights_out": weights_out}

    if use_fp4:
        outputs["q_packed"] = torch.full(
            (num_tokens + 1, n_head, HEAD_DIM // 2),
            sentinel_u8,
            dtype=torch.uint8,
            device=device,
        )
        outputs["q_scale"] = torch.full(
            (num_tokens + 1, n_head, HEAD_DIM // 32),
            sentinel_u8,
            dtype=torch.uint8,
            device=device,
        )
        mod.fused_indexer_q_rope_quant_mxfp4_cutedsl(
            positions[:num_tokens],
            q[:num_tokens],
            cos_sin_cache,
            weights[:num_tokens],
            1.0,
            1.0,
            outputs["q_packed"][:num_tokens],
            outputs["q_scale"][:num_tokens],
            weights_out[:num_tokens],
        )
    else:
        outputs["q_fp8"] = torch.full(
            (num_tokens + 1, n_head, HEAD_DIM),
            sentinel_u8,
            dtype=torch.uint8,
            device=device,
        )
        mod.fused_indexer_q_rope_quant_fp8_cutedsl(
            positions[:num_tokens],
            q[:num_tokens],
            cos_sin_cache,
            weights[:num_tokens],
            1.0,
            1.0,
            outputs["q_fp8"][:num_tokens].view(torch.float8_e4m3fn),
            weights_out[:num_tokens],
        )
    torch.accelerator.synchronize()

    clobbered = {}
    for name, tensor in outputs.items():
        sentinel = sentinel_f32 if tensor.dtype == torch.float32 else sentinel_u8
        n_changed = (tensor[num_tokens] != sentinel).sum().item()
        if n_changed:
            clobbered[name] = n_changed
    assert not clobbered, (
        f"kernel wrote past token {num_tokens - 1} into the sentinel row: {clobbered}"
    )


def _indexer_k_reference(
    k_pre, positions, cos_sin_cache, rms_norm_weight, compress_ratio, use_fp4
):
    k = k_pre.float()
    k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1e-20)
    k = (k * rms_norm_weight.float()).to(torch.bfloat16).float()
    group_positions = positions // compress_ratio * compress_ratio
    cos, sin = cos_sin_cache[group_positions].float().chunk(2, dim=-1)
    even, odd = k[:, 64::2], k[:, 65::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    k = torch.cat((k[:, :64], rotated.flatten(1)), dim=-1).to(torch.bfloat16)
    if use_fp4:
        return quantize_to_mxfp4(k)
    exponent = k.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448
    scale = exponent.log2().ceil().exp2()
    values = (k.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.view(torch.uint8), scale.view(torch.uint8)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA kernel")
@pytest.mark.parametrize("num_tokens", [1, 17, 257])
@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("use_fp4", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_indexer_k_inserts_only_valid_groups_into_padded_pages(
    num_tokens, compress_ratio, use_fp4, cache_dtype
):
    """Insert group ends and preserve skipped slots, graph rows, and page padding."""
    from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )

    if use_fp4 and not current_platform.is_device_capability_family(100):
        pytest.skip("MXFP4 indexer cache requires an SM100-family GPU")
    torch.manual_seed(17)
    device = "cuda"
    k_pre = torch.randn(num_tokens + 3, 128, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    positions = torch.arange(7, num_tokens + 10, device=device)
    slots = torch.randperm(2 * num_tokens, device=device)[:num_tokens].clone()
    slots[5::7] = -1
    valid = (slots >= 0) & ((positions[:num_tokens] + 1) % compress_ratio == 0)
    k_pre[3::7] = 0
    k_pre[8::11] *= 1e-10
    k_pre[:num_tokens][~valid] = float("nan")
    k_pre[num_tokens:] = float("nan")
    positions[num_tokens:] = MAX_POS + 10
    angles = torch.randn(MAX_POS, 32, device=device)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(cache_dtype)
    block_size = 16
    value_bytes = 64 if use_fp4 else 128
    page_bytes = block_size * (value_bytes + 4)
    num_blocks = (2 * num_tokens + block_size - 1) // block_size + 1
    storage = torch.full(
        (num_blocks, page_bytes + 128), 0xA5, device=device, dtype=torch.uint8
    )
    cache = storage.as_strided(
        (num_blocks, block_size, value_bytes + 4),
        (storage.stride(0), value_bytes + 4, 1),
    )
    expected = storage.clone()
    values, scales = _indexer_k_reference(
        k_pre[:num_tokens][valid],
        positions[:num_tokens][valid],
        cos_sin,
        norm_weight,
        compress_ratio,
        use_fp4,
    )
    block_ids = slots[valid] // block_size
    block_offsets = slots[valid] % block_size
    value_offsets = block_offsets[:, None] * value_bytes + torch.arange(
        value_bytes, device=device
    )
    scale_offsets = block_size * value_bytes + block_offsets[:, None] * 4
    scale_offsets = scale_offsets + torch.arange(4, device=device)
    expected[block_ids[:, None], value_offsets] = values
    expected[block_ids[:, None], scale_offsets] = scales

    indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        norm_weight,
        1e-20,
        cache,
        slots,
        compress_ratio,
        use_fp4,
    )
    # Fused RoPE can change zero signs without changing their numerical values.
    actual_values = storage[block_ids[:, None], value_offsets]
    if use_fp4:
        for shift in (0, 4):
            zero_mask = (actual_values >> shift) & 7 == 0
            actual_values &= ~(zero_mask.to(torch.uint8) << (shift + 3))
            zero_mask = (values >> shift) & 7 == 0
            values &= ~(zero_mask.to(torch.uint8) << (shift + 3))
    else:
        actual_values[actual_values == 128] = 0
        values[values == 128] = 0
    storage[block_ids[:, None], value_offsets] = actual_values
    expected[block_ids[:, None], value_offsets] = values
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA kernel")
@pytest.mark.parametrize("use_fp4", [False, True])
@torch.inference_mode()
def test_indexer_k_cuda_graph_replay_reads_current_projection(use_fp4):
    """Replay must consume updated keys and slot mapping through the captured API."""
    from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )

    if use_fp4 and not current_platform.is_device_capability_family(100):
        pytest.skip("MXFP4 indexer cache requires an SM100-family GPU")
    torch.manual_seed(18)
    k_pre = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(19, device="cuda")
    cos_sin = torch.zeros(32, 64, device="cuda")
    cos_sin[:, :32] = 1
    slots = torch.arange(17, device="cuda")
    cache = torch.full(
        (2, 16, (64 if use_fp4 else 128) + 4),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )

    def run():
        indexer_k_norm_rope_store(
            k_pre,
            positions,
            cos_sin,
            norm_weight,
            1e-20,
            cache,
            slots,
            2,
            use_fp4,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    k_pre.neg_()
    slots.add_(4)
    cache.fill_(0xA5)
    graph.replay()
    replayed = cache.clone()
    cache.fill_(0xA5)
    run()
    torch.testing.assert_close(cache, replayed, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm cache layout")
@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("compress_ratio", [1, 2])
@torch.inference_mode()
def test_indexer_k_store_roundtrips_through_rocm_gather(block_size, compress_ratio):
    """ROCm reads the indexer K cache back with the 16x16-tiled value layout.

    ``cp_gather_indexer_k_quant_cache_triton`` (prefill) and aiter's
    ``Preshuffle=True`` paged kernel (decode) both pick that layout whenever
    ``block_size > 1``, so a row-major store would feed the indexer permuted
    key bytes and randomize its top-k. Assert the write/read pair is exact.
    """
    from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
    )

    torch.manual_seed(0)
    device = "cuda"
    num_tokens = 3 * block_size + 5
    num_blocks = num_tokens // block_size + 2
    k_pre = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device=device)
    angles = torch.randn(MAX_POS, ROPE_DIM // 2, device=device)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    cache = torch.zeros(
        num_blocks, block_size, HEAD_DIM + 4, device=device, dtype=torch.uint8
    )
    slots = torch.arange(num_tokens, device=device)

    indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        norm_weight,
        1e-20,
        cache,
        slots,
        compress_ratio,
        False,
    )

    k_fp8 = torch.zeros(num_tokens, HEAD_DIM, device=device, dtype=torch.float8_e4m3fn)
    k_scale = torch.zeros(num_tokens, 4, device=device, dtype=torch.uint8)
    cp_gather_indexer_k_quant_cache_triton(
        cache,
        k_fp8,
        k_scale,
        torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1),
        torch.tensor([0, num_tokens], device=device, dtype=torch.int32),
        torch.zeros(num_tokens, device=device, dtype=torch.int32),
    )

    expected_values, expected_scale = _indexer_k_reference(
        k_pre, positions, cos_sin, norm_weight, compress_ratio, False
    )
    emitted = (positions + 1) % compress_ratio == 0
    gathered = k_fp8.view(torch.uint8)
    gathered = torch.where(gathered == 128, gathered.new_zeros(()), gathered)
    expected_values = torch.where(
        expected_values == 128, expected_values.new_zeros(()), expected_values
    )
    torch.testing.assert_close(
        gathered[emitted], expected_values[emitted], rtol=0, atol=0
    )
    torch.testing.assert_close(
        k_scale[emitted], expected_scale[emitted], rtol=0, atol=0
    )
