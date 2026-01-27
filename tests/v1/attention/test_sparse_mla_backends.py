# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the FlashMLA sparse backend utilities."""

import math
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.test_mla_backends import (
    BATCH_SPECS,
    BatchSpec,
    MockAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm import _custom_ops as ops
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import split_prefill_chunks
from vllm.v1.attention.ops import flashmla

SPARSE_BACKEND_BATCH_SPECS = {
    name: BATCH_SPECS[name]
    for name in [
        "mixed_small",
        "mixed_medium",
        "small_prefill",
        "medium_prefill",
        "single_prefill",
    ]
}

SPARSE_BACKEND_BATCH_SPECS["large_q_prefill"] = BatchSpec(
    seq_lens=[1024] * 2, query_lens=[256] * 2
)
SPARSE_BACKEND_BATCH_SPECS["large_q_pure_prefill"] = BatchSpec(
    seq_lens=[256] * 2, query_lens=[256] * 2
)


def _float_to_e8m0_truncate(f: float) -> float:
    """Simulate SM100's float -> e8m0 -> bf16 scale conversion.
    e8m0 format only stores the exponent (power of 2).
    cudaRoundZero truncates toward zero, meaning we round down to the
    nearest power of 2.
    """
    if f <= 0:
        return 0.0
    # e8m0 = floor(log2(f)), then 2^(e8m0)
    # This is equivalent to truncating to the nearest power of 2 below f
    exp = math.floor(math.log2(f))
    return 2.0**exp


def _dequantize_fp8_ds_mla_entry(
    cache_slice: torch.Tensor,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a single fp8_ds_mla cache entry back to latent + rope.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion path.
    """

    # The first kv_lora_rank bytes store FP8 latent values with one scale per
    # 128 element tile written as float32 right after the latent payload.
    scales = cache_slice.view(torch.float32)[kv_lora_rank // 4 : kv_lora_rank // 4 + 4]
    latent = torch.empty(kv_lora_rank, dtype=torch.float16, device=cache_slice.device)
    for tile_idx in range(4):
        tile_start = tile_idx * 128
        tile_end = tile_start + 128
        scale_val = float(scales[tile_idx].item())
        if simulate_sm100_e8m0_scales:
            # Simulate the lossy float -> e8m0 -> bf16 conversion
            scale_val = _float_to_e8m0_truncate(scale_val)
        ops.convert_fp8(
            latent[tile_start:tile_end],
            cache_slice[tile_start:tile_end],
            scale_val,
            kv_dtype="fp8",
        )
    latent = latent.to(dtype)

    rope_offset = kv_lora_rank // 2 + 8
    rope_vals = cache_slice.view(dtype)[rope_offset : rope_offset + rope_dim]
    return latent, rope_vals.clone()


def _quantize_dequantize_fp8_ds_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    block_size: int,
    scale: torch.Tensor,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-trip kv_c/k_pe though the fp8_ds_mla cache layout.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion in dequantization.
    """

    if kv_c.numel() == 0:
        return kv_c.clone(), k_pe.clone()

    kv_lora_rank = kv_c.shape[-1]
    rope_dim = k_pe.shape[-1]
    num_tokens = kv_c.shape[0]
    num_blocks = max(1, math.ceil(num_tokens / block_size))
    entry_size = kv_lora_rank + 4 * 4 + 2 * rope_dim

    tmp_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=kv_c.device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=kv_c.device)

    ops.concat_and_cache_mla(
        kv_c, k_pe, tmp_cache, slot_mapping, kv_cache_dtype="fp8_ds_mla", scale=scale
    )

    dequant_kv_c = torch.empty_like(kv_c)
    dequant_k_pe = torch.empty_like(k_pe)

    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        cache_slice = tmp_cache[block_idx, block_offset]
        latent, rope_vals = _dequantize_fp8_ds_mla_entry(
            cache_slice,
            kv_lora_rank,
            rope_dim,
            kv_c.dtype,
            simulate_sm100_e8m0_scales=simulate_sm100_e8m0_scales,
        )
        dequant_kv_c[token_idx] = latent
        dequant_k_pe[token_idx] = rope_vals

    return dequant_kv_c, dequant_k_pe


@pytest.mark.parametrize("batch_name", list(SPARSE_BACKEND_BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["fp8_ds_mla", "auto"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_sparse_backend_decode_correctness(
    default_vllm_config,
    dist_init,
    batch_name,
    kv_cache_dtype,
    tensor_parallel_size,
    workspace_init,
):
    if current_platform.is_rocm():
        pytest.skip("ROCm does not support fp8_ds_mla data type for kv cache.")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for sparse MLA decode test")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_spec = SPARSE_BACKEND_BATCH_SPECS[batch_name]

    # Model hyper-parameters (kept intentionally small for the unit test)
    total_num_heads = 128
    # Compute per-rank heads for simulated TP
    num_heads = max(1, total_num_heads // tensor_parallel_size)

    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 2048

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)
    block_size = 64

    # Note: We use TP=1 to avoid multi-GPU requirements in CI.
    # The test simulates head partitioning via mocked methods below.
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=max_seqlen,
        num_gpu_blocks=max(2048, cdiv(total_cache_tokens, block_size) + 1),
        block_size=block_size,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.dtype = dtype
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: num_heads,
        model_config,
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    torch.manual_seed(0)

    scale = 1.0 / math.sqrt(head_size)

    # Shared MLA projection weights to keep reference and backend in sync
    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    # Build synthetic decode-only workload
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []

    kv_cache_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len

        q_c = torch.rand(
            q_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        # SM100 (Blackwell) uses float -> e8m0 -> bf16 scale conversion
        # which truncates scales to powers of 2. Simulate this in reference.
        is_sm100 = torch.cuda.get_device_capability()[0] >= 10
        kv_c_full, k_pe_full = _quantize_dequantize_fp8_ds_mla(
            kv_c_full,
            k_pe_full.squeeze(1),
            block_size=vllm_config.cache_config.block_size,
            scale=kv_cache_scale,
            simulate_sm100_e8m0_scales=is_sm100,
        )

        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)

        k_mqa = torch.cat([kv_c_full, k_pe_full], dim=-1)
        k_mqa = k_mqa.unsqueeze(1).expand(-1, num_heads, -1)
        v_mqa = kv_c_full.unsqueeze(1).expand(-1, num_heads, -1)

        attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=device)
        causal_mask = torch.tril(torch.ones(q_len, q_len, device=device))
        attn_mask[:, ctx_len:] = causal_mask

        q_sdpa_in = q_mqa.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_mqa.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_mqa.unsqueeze(0).transpose(1, 2)

        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, scale=scale
        )
        sdpa_out = sdpa_out.transpose(1, 2).squeeze(0)

        sdpa_out = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)
        reference_outputs.append(sdpa_out.flatten(start_dim=-2))

        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[ctx_len:])
        all_k_pe_vllm.append(k_pe_full[ctx_len:])
        kv_c_contexts.append(kv_c_full[: ctx_len + 1])
        k_pe_contexts.append(k_pe_full[: ctx_len + 1])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
    sdpa_reference = torch.cat(reference_outputs, dim=0)

    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    vllm_config.model_config.hf_config.index_topk = topk_tokens

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        vllm_config.cache_config.block_size,
        device,
        arange_block_indices=True,
    )

    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=vllm_config.cache_config.block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        scale=kv_cache_scale,
    )

    builder_cls = FlashMLASparseBackend.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
    seg_lengths = np.diff(starts)
    positions = np.arange(starts[-1], dtype=np.int32) - np.repeat(
        starts[:-1], seg_lengths
    )
    seq_lengths = np.asarray(common_attn_metadata.seq_lens.cpu(), dtype=np.int32)
    prefix_lengths = seq_lengths - seg_lengths
    positions += np.repeat(prefix_lengths, seg_lengths)

    pos_gpu = torch.as_tensor(positions, device=device, dtype=torch.int32)
    topk = metadata.topk_tokens
    debug_indices = torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0)
    token_positions = pos_gpu.unsqueeze(1)
    causal_mask = debug_indices <= token_positions
    debug_indices = torch.where(
        causal_mask, debug_indices, torch.full_like(debug_indices, -1)
    )

    # FlashMLASparseImpl now reads top-k indices from the indexer-provided
    # buffer, so emulate that contract with a simple namespace mock.
    debug_indices = debug_indices.expand(metadata.num_actual_tokens, -1).clone()
    mock_indexer = SimpleNamespace(topk_indices_buffer=debug_indices)

    ok, reason = flashmla.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
    )

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T.contiguous())

    impl_cls = FlashMLASparseBackend.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=mock_kv_b_proj,
            indexer=mock_indexer,
        )

        impl.process_weights_after_loading(dtype)

    layer = MockAttentionLayer(device)
    out_buffer = torch.empty(
        metadata.num_actual_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        backend_output = impl.forward(
            layer,
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            metadata,
            output=out_buffer,
        )

    assert backend_output.shape == sdpa_reference.shape
    assert backend_output.dtype == sdpa_reference.dtype
    assert torch.isfinite(backend_output).all()

    # FP8 quantization introduces some error, but should be within reasonable bounds
    # BF16 (auto) should be very accurate, FP8 allows slightly more tolerance
    if kv_cache_dtype == "fp8_ds_mla":
        torch.testing.assert_close(backend_output, sdpa_reference, rtol=0.05, atol=0.05)
    else:
        torch.testing.assert_close(backend_output, sdpa_reference, rtol=0.01, atol=0.01)


def _triton_convert_reference_impl(
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    num_topk_tokens: int,
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation for triton_convert_req_index_to_global_index."""
    num_tokens = req_ids.shape[0]
    max_blocks_per_req = block_table.shape[1]
    result = torch.empty(
        num_tokens, num_topk_tokens, dtype=torch.int32, device=req_ids.device
    )

    for token_id in range(num_tokens):
        req_id = req_ids[token_id].item()

        # Determine if this token uses workspace or paged cache
        use_prefill_workspace = False
        workspace_start = 0
        if HAS_PREFILL_WORKSPACE and prefill_workspace_request_ids is not None:
            assert prefill_workspace_starts is not None
            prefill_req_id = prefill_workspace_request_ids[token_id].item()
            if prefill_req_id >= 0:
                use_prefill_workspace = True
                workspace_start = prefill_workspace_starts[prefill_req_id].item()

        for idx_id in range(num_topk_tokens):
            token_idx = token_indices[token_id, idx_id].item()

            if token_idx == -1:
                result[token_id, idx_id] = -1
            elif use_prefill_workspace:
                # Prefill + using prefill workspace: map to workspace offset
                result[token_id, idx_id] = workspace_start + token_idx
            else:
                # Decode: map to paged cache
                block_id = token_idx // block_size
                if block_id >= max_blocks_per_req:
                    result[token_id, idx_id] = -1
                else:
                    block_num = block_table[req_id, block_id].item()
                    offset = token_idx % block_size
                    result[token_id, idx_id] = block_num * block_size + offset

    return result


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("num_topk_tokens", [128, 256, 512])
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_decode_only(
    block_size, num_topk_tokens
):
    device = torch.device("cuda")
    num_tokens = 8
    num_requests = 4
    max_blocks_per_req = 10

    req_id = torch.randint(
        0, num_requests, (num_tokens,), dtype=torch.int32, device=device
    )
    block_table = torch.randint(
        0, 100, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )

    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (num_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_with_prefill_workspace(block_size):
    device = torch.device("cuda")
    num_requests = 4
    max_blocks_per_req = 8
    num_topk_tokens = 128

    # First 6 tokens are decode (reqs 0, 1), last 6 are prefill (reqs 2, 3)
    req_id = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.int32, device=device
    )
    prefill_workspace_request_ids = torch.tensor(
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
    )

    # Workspace starts for the 2 prefill reqs: req 2 starts at 0, req 3 starts at 100
    prefill_workspace_starts = torch.tensor([0, 100], dtype=torch.int32, device=device)

    block_table = torch.randint(
        0, 50, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )
    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (req_id.shape[0], num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.parametrize(
    "seq_lens,max_buf,expected",
    [
        # Basic split: totals per chunk ≤ max_buf
        (torch.tensor([2, 3, 4, 2]), 5, [(0, 2), (2, 3), (3, 4)]),
        # Exact fits should split between items when adding the next would overflow
        (torch.tensor([5, 5, 5]), 5, [(0, 1), (1, 2), (2, 3)]),
        # All requests fit in a single chunk
        (torch.tensor([1, 1, 1]), 10, [(0, 3)]),
        # Large buffer
        (torch.tensor([4, 4, 4]), 100, [(0, 3)]),
    ],
)
def test_split_prefill_chunks(seq_lens, max_buf, expected):
    out = split_prefill_chunks(seq_lens, max_buf)
    assert out == expected
