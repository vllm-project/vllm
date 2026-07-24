# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the sparse MLA backends and utilities."""

import math
from types import MethodType, SimpleNamespace

import pytest
import torch

from tests.v1.attention.test_mla_backends import (
    BATCH_SPECS,
    BatchSpec,
    MockSparseMLAAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm import _custom_ops as ops
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadataBuilder,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform

# TODO: Integrate ROCMAiterMLASparseBackend for ROCm.
# The ROCm sparse MLA backend (rocm_aiter_mla_sparse.py) has a compatible
# forward_mqa interface but needs validation on ROCm hardware.
if not current_platform.is_cuda():
    pytest.skip(
        "Sparse MLA backend tests currently only support CUDA. "
        "ROCm support requires integrating ROCMAiterMLASparseBackend.",
        allow_module_level=True,
    )

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.indexer import split_indexer_prefill_chunks
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
    split_prefill_chunks,
)
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

DEVICE_TYPE = current_platform.device_type


def test_flashinfer_sparse_mla_accepts_zero_local_tokens() -> None:
    impl = object.__new__(FlashInferMLASparseImpl)
    impl.num_heads = 4
    impl.kv_lora_rank = 512
    q = torch.empty((0, 4, 192), device=DEVICE_TYPE)

    output, lse = impl.forward_mqa(
        q,
        torch.empty((1,), device=DEVICE_TYPE),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert output.shape == (0, 4, 512)
    assert output.device == q.device
    assert lse is None


def test_flashmla_sparse_mla_accepts_zero_local_tokens() -> None:
    impl = object.__new__(FlashMLASparseImpl)
    impl.num_heads = 4
    impl.kv_lora_rank = 512
    impl.q_concat_buffer = torch.empty(
        (0, 4, 576), dtype=torch.bfloat16, device=DEVICE_TYPE
    )
    q_nope = torch.empty((0, 4, 512), dtype=torch.bfloat16, device=DEVICE_TYPE)
    q_pe = torch.empty((0, 4, 64), dtype=torch.bfloat16, device=DEVICE_TYPE)

    output, lse = impl.forward_mqa(
        (q_nope, q_pe),
        torch.empty((1,), device=DEVICE_TYPE),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert output.shape == (0, 4, 512)
    assert output.device == q_nope.device
    assert lse is None


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


@pytest.mark.parametrize(
    "backend_cls",
    [FlashMLASparseBackend, FlashInferMLASparseTRTLLMBackend],
    ids=["FlashMLA", "FlashInferTRTLLM"],
)
@pytest.mark.parametrize("batch_name", list(SPARSE_BACKEND_BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_ds_mla"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize(("q_scale", "k_scale"), [(1.0, 1.0), (2.0, 3.0)])
def test_sparse_backend_decode_correctness(
    default_vllm_config,
    dist_init,
    backend_cls,
    batch_name,
    kv_cache_dtype,
    tensor_parallel_size,
    block_size,
    workspace_init,
    q_scale: float,
    k_scale: float,
):
    if kv_cache_dtype not in backend_cls.supported_kv_cache_dtypes:
        pytest.skip(f"{backend_cls.get_name()} does not support {kv_cache_dtype}")

    if (
        backend_cls == FlashMLASparseBackend
        and kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_ds_mla"
    ):
        pytest.skip(
            "FlashMLA Sparse Attention backend fp8 only supports "
            "fp8_ds_mla kv-cache dtype"
        )

    supported_block_sizes = backend_cls.get_supported_kernel_block_sizes()
    if block_size not in supported_block_sizes:
        pytest.skip(
            f"{backend_cls.get_name()} does not support block_size={block_size}"
        )

    if backend_cls == FlashMLASparseBackend:
        ok, reason = flashmla.is_flashmla_sparse_supported()
        if not ok:
            pytest.skip(reason)
    elif backend_cls == FlashInferMLASparseTRTLLMBackend:
        device_capability = current_platform.get_device_capability()
        if device_capability is None or not backend_cls.supports_compute_capability(
            device_capability
        ):
            pytest.skip("FlashInferMLASparseTRTLLMBackend requires SM 10.x capability")

    batch_spec = SPARSE_BACKEND_BATCH_SPECS[batch_name]
    use_fp8_ds_mla_quantization = kv_cache_dtype == "fp8_ds_mla"

    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16

    # Model hyper-parameters (kept intentionally small for the unit test)
    total_num_heads = 128
    # Compute per-rank heads for simulated TP
    num_heads = max(1, total_num_heads // tensor_parallel_size)

    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 128

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)

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

    # Pre-compute positions and sparse indices for all tokens.
    # We need these BEFORE computing the reference to use sparse attention masks.
    total_query_tokens = sum(query_lens)
    positions = []
    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len
        for q_idx in range(q_len):
            positions.append(ctx_len + q_idx)

    # Create sparse indices with UNIQUE per-token offsets to catch bugs where
    # the kernel uses wrong indices for some tokens (e.g., due to incorrect
    # tensor shapes like [1, num_tokens, ...] instead of [num_tokens, 1, ...]).
    # Also include -1 masked indices to verify the kernel handles them correctly.
    sparse_indices = torch.empty(
        total_query_tokens, topk_tokens, dtype=torch.int32, device=device
    )
    for tok_idx in range(total_query_tokens):
        max_valid_idx = positions[tok_idx]
        offset = tok_idx * 7  # Prime number for varied offsets
        # Use only half the topk indices as valid, mask the rest with -1
        # This tests that the kernel correctly ignores -1 indices
        num_valid = min(topk_tokens // 2, max_valid_idx + 1)
        if num_valid > 0:
            valid_range = torch.arange(num_valid, device=device, dtype=torch.int32)
            tok_indices = (valid_range + offset) % (max_valid_idx + 1)
            # Pad with -1 for the remaining positions
            tok_indices = torch.cat(
                [
                    tok_indices,
                    torch.full(
                        (topk_tokens - num_valid,), -1, device=device, dtype=torch.int32
                    ),
                ]
            )
        else:
            tok_indices = torch.full(
                (topk_tokens,), -1, device=device, dtype=torch.int32
            )
            tok_indices[0] = 0  # At least one valid index
        sparse_indices[tok_idx] = tok_indices

    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []

    kv_cache_scale = torch.tensor(k_scale, dtype=torch.float32, device=device)
    global_token_idx = 0

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

        if use_fp8_ds_mla_quantization:
            is_sm100 = torch.cuda.get_device_capability()[0] >= 10
            kv_c_full, k_pe_squeezed = _quantize_dequantize_fp8_ds_mla(
                kv_c_full,
                k_pe_full.squeeze(1),
                block_size=block_size,
                scale=kv_cache_scale,
                simulate_sm100_e8m0_scales=is_sm100,
            )
            k_pe_full = k_pe_squeezed.unsqueeze(1)

        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)

        k_mqa = torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
        v_mqa = kv_c_full

        # Compute sparse SDPA reference per query token using its sparse indices
        for q_idx in range(q_len):
            tok_sparse_idx = sparse_indices[global_token_idx]
            valid_mask = tok_sparse_idx >= 0
            valid_indices = tok_sparse_idx[valid_mask].long()

            q_tok = q_mqa[q_idx : q_idx + 1]  # [1, num_heads, head_dim]
            k_sparse = k_mqa[valid_indices]  # [num_valid, head_dim]
            v_sparse = v_mqa[valid_indices]  # [num_valid, kv_lora_rank]

            k_sparse = k_sparse.unsqueeze(1).expand(-1, num_heads, -1)
            v_sparse = v_sparse.unsqueeze(1).expand(-1, num_heads, -1)

            # SDPA: [1, num_heads, 1, head_dim] x [1, num_heads, num_valid, head_dim]
            q_sdpa_in = q_tok.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_sparse.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_sparse.unsqueeze(0).transpose(1, 2)

            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in, k_sdpa_in, v_sdpa_in, scale=scale
            )
            sdpa_out = sdpa_out.transpose(1, 2).squeeze(
                0
            )  # [1, num_heads, kv_lora_rank]

            sdpa_out = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)
            reference_outputs.append(sdpa_out.flatten(start_dim=-2))

            global_token_idx += 1

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
        kv_cache_dtype=kv_cache_dtype,
        scale=kv_cache_scale,
    )

    # The sparse builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; register a mock layer carrying one.
    from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=num_heads,
        scale=scale,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=prefill_backend)
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    # Use the pre-computed sparse_indices for the mock indexer
    mock_indexer = SimpleNamespace(topk_indices_buffer=sparse_indices)

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

    impl_cls = backend_cls.get_impl_cls()
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

        # Create mock sparse MLA layer with weight matrices
        mock_layer = MockSparseMLAAttentionLayer(
            impl=impl,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            W_UK=W_UK,
            W_UV=W_UV,
            q_scale=q_scale,
            k_scale=k_scale,
        )

    out_buffer = torch.empty(
        metadata.num_actual_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        backend_output = mock_layer.forward_impl(
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            metadata,
            out_buffer,
        )

    assert backend_output.shape == sdpa_reference.shape
    assert backend_output.dtype == sdpa_reference.dtype
    assert torch.isfinite(backend_output).all()

    # FP8 quantization introduces some error, but should be within reasonable bounds
    # BF16 (auto) should be very accurate, FP8 allows slightly more tolerance
    if kv_cache_dtype.startswith("fp8"):
        torch.testing.assert_close(
            backend_output, sdpa_reference, rtol=0.065, atol=0.05
        )
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
    device = torch.device(DEVICE_TYPE)
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
    # A second decode row also carries out-of-bounds paged-cache indices.
    token_indices[6, 110:120] = max_blocks_per_req * block_size

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
    device = torch.device(DEVICE_TYPE)
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

    # Decode rows must reject paged-cache indices beyond the BlockTable.
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    # Dense prefill workspaces use global token IDs and are not bounded by the
    # owner-local BlockTable width. Include both the DCP4 32K boundary and a
    # 64K-history token, plus a sentinel that must remain invalid.
    token_indices[6, 110:114] = torch.tensor(
        [max_blocks_per_req * block_size, 32768, 65535, -1],
        dtype=torch.int32,
        device=device,
    )

    result, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
        return_valid_counts=True,
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
    expected_valid_counts = (reference_result >= 0).sum(dim=1, dtype=torch.int32)
    torch.testing.assert_close(valid_counts, expected_valid_counts, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_rejects_req_id_longer_than_token_indices():
    """Guard against the #47327 regression: the kernel grid is sized by
    req_id but the output is allocated like token_indices, so a full-batch
    req_id combined with an MQA-subset token_indices wrote past the end of
    the output buffer. The wrapper must reject the length mismatch instead
    of corrupting memory."""
    device = torch.device(DEVICE_TYPE)
    num_topk_tokens = 128
    block_size = 64
    block_table = torch.arange(40, dtype=torch.int32, device=device).view(4, 10)

    # Full batch: 2 decode tokens + 10 prefill tokens
    req_id_full = torch.tensor(
        [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=torch.int32, device=device
    )
    num_mqa_tokens = 2
    token_indices = torch.randint(
        0,
        block_size * 10,
        (num_mqa_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    with pytest.raises(AssertionError, match="must cover the same tokens"):
        triton_convert_req_index_to_global_index(
            req_id_full,
            block_table,
            token_indices,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk_tokens,
        )

    # The sliced call is the intended usage and must match the reference.
    result = triton_convert_req_index_to_global_index(
        req_id_full[:num_mqa_tokens],
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
    )
    reference = _triton_convert_reference_impl(
        req_id_full[:num_mqa_tokens],
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
    )
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_flashmla_forward_bf16_kv_slices_req_id_to_mqa_tokens():
    """Guard against the #47327 regression: when the dense-MHA prefill split
    is active, forward_mqa only receives the leading decode tokens, but
    _forward_bf16_kv passed the full-batch req_id_per_token to the index
    conversion, making it write past the end of its output buffer. The call
    site must slice req_id_per_token to the MQA tokens."""
    device = torch.device(DEVICE_TYPE)
    num_topk_tokens = 128
    block_size = 64
    num_batch_tokens = 12
    num_mqa_tokens = 2

    attn_metadata = SimpleNamespace(
        req_id_per_token=torch.tensor(
            [0, 1] + [2] * 5 + [3] * 5, dtype=torch.int32, device=device
        ),
        block_table=torch.arange(40, dtype=torch.int32, device=device).view(4, 10),
        block_size=block_size,
    )
    assert attn_metadata.req_id_per_token.shape[0] == num_batch_tokens

    q = torch.zeros(num_mqa_tokens, 4, 576, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(40 * block_size, 576, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(
        0,
        block_size * 10,
        (num_mqa_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    captured = {}

    def _stub_kernel(q, kv, indices, lengths):
        captured["indices"] = indices
        return torch.zeros(q.shape[0], q.shape[1], 512, dtype=q.dtype, device=q.device)

    stub_impl = SimpleNamespace(_bf16_flash_mla_kernel=_stub_kernel)

    out = FlashMLASparseImpl._forward_bf16_kv(
        stub_impl, q, kv_cache, topk_indices, attn_metadata
    )

    assert out.shape[0] == num_mqa_tokens
    assert captured["indices"].shape[0] == num_mqa_tokens
    reference = _triton_convert_reference_impl(
        attn_metadata.req_id_per_token[:num_mqa_tokens],
        attn_metadata.block_table,
        topk_indices,
        block_size,
        num_topk_tokens,
    )
    torch.testing.assert_close(captured["indices"], reference, rtol=0, atol=0)


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


PREFILL_BATCH_SPECS = {
    "short_dense_mha": BatchSpec(seq_lens=[64, 128], query_lens=[64, 128]),
    "short_context_dense_mha": BatchSpec(seq_lens=[128, 160], query_lens=[64, 32]),
}


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="Sparse MLA forward_mha requires FA4 (SM100+)",
)
@pytest.mark.parametrize("batch_name", list(PREFILL_BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
def test_sparse_backend_prefill_correctness(
    default_vllm_config,
    dist_init,
    batch_name,
    kv_cache_dtype,
    workspace_init,
):
    """Test single-pass FA4 dense forward_mha for sparse MLA prefill."""
    backend_cls = FlashMLASparseBackend
    batch_spec = PREFILL_BATCH_SPECS[batch_name]

    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 64

    num_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 512

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)

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
        lambda self, parallel_config: num_heads, model_config
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

    torch.manual_seed(42)

    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Compute dense reference outputs.
    total_query_tokens = sum(query_lens)
    sparse_indices = torch.zeros(
        total_query_tokens, topk_tokens, dtype=torch.int32, device=device
    )

    all_q, all_kv_c_new, all_k_pe_new = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []

    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len

        q_mha = torch.rand(
            q_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        # Decompress all KV for reference
        kv_b_weight = torch.cat([W_UK, W_UV], dim=-1).view(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
        )
        kv_decompressed = (kv_c_full @ kv_b_weight).view(
            s_len, num_heads, qk_nope_head_dim + v_head_dim
        )
        k_nope_all, v_all = kv_decompressed.split(
            [qk_nope_head_dim, v_head_dim], dim=-1
        )
        k_pe_expanded = k_pe_full.expand(-1, num_heads, -1)
        k_all = torch.cat([k_nope_all, k_pe_expanded], dim=-1)

        for j in range(q_len):
            attend_end = ctx_len + j + 1
            q_tok = q_mha[j : j + 1]  # (1, H, D_qk)
            k_attend = k_all[:attend_end]  # (N, H, D_qk)
            v_attend = v_all[:attend_end]  # (N, H, D_v)

            q_sdpa = q_tok.unsqueeze(0).transpose(1, 2).float()
            k_sdpa = k_attend.unsqueeze(0).transpose(1, 2).float()
            v_sdpa = v_attend.unsqueeze(0).transpose(1, 2).float()

            out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, scale=scale
            )
            out = out.transpose(1, 2).squeeze(0)  # (1, H, D_v)
            reference_outputs.append(out.to(dtype).flatten(start_dim=-2))

        all_q.append(q_mha)
        all_kv_c_new.append(kv_c_full[ctx_len:])
        all_k_pe_new.append(k_pe_full[ctx_len:])
        kv_c_contexts.append(kv_c_full)
        k_pe_contexts.append(k_pe_full)

    query_cat = torch.cat(all_q, dim=0)
    kv_c_cat = torch.cat(all_kv_c_new, dim=0)
    k_pe_cat = torch.cat(all_k_pe_new, dim=0)
    ref_output = torch.cat(reference_outputs, dim=0)

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
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
    )

    # The sparse builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; register a mock layer carrying one.
    from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=num_heads,
        scale=scale,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=prefill_backend)
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    # Drive the queries through the dense-MHA prefill path directly (the routing
    # threshold would otherwise classify these short queries as MQA decodes).
    builder.reorder_batch_threshold = 1
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    mock_indexer = SimpleNamespace(topk_indices_buffer=sparse_indices)

    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1).view(
        kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
    )

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T.contiguous())

    impl_cls = backend_cls.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
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

    out_buffer = torch.empty(
        total_query_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        impl.forward_mha(
            q=query_cat,
            kv_c_normed=kv_c_cat,
            k_pe=k_pe_cat,
            kv_c_and_k_pe_cache=kv_cache,
            attn_metadata=metadata,
            k_scale=torch.tensor(1.0, device=device),
            output=out_buffer,
        )

    assert out_buffer.shape == ref_output.shape
    assert torch.isfinite(out_buffer).all(), "Non-finite values in output"
    torch.testing.assert_close(out_buffer, ref_output, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "seq_lens,query_lens,workspace_size,max_logits_bytes,expected",
    [
        (
            torch.tensor([0]),
            torch.tensor([0]),
            100,
            1000,
            [],
        ),
        # Logits constraint triggers split (M*N exceeds budget)
        # req0: M=10, N=100 -> 1000 elems (4000 bytes) - fits in 5000
        # req1: adding M=10, N=100 -> new_M=20, new_N=200 -> 4000 elems > 1250
        (
            torch.tensor([100, 100, 100]),
            torch.tensor([10, 10, 10]),
            1000,  # workspace allows all
            5000,  # 1250 float32 elems -> forces split
            [
                (slice(0, 1), slice(0, 10)),
                (slice(1, 2), slice(0, 10)),
                (slice(2, 3), slice(0, 10)),
            ],
        ),
        # Both constraints satisfied - all fit in one chunk
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            10000,  # 2500 elems, M*N = 15*30 = 450 < 2500
            [(slice(0, 3), slice(0, 15))],
        ),
        # Workspace constraint triggers first
        (
            torch.tensor([50, 50, 50]),
            torch.tensor([1, 1, 1]),
            50,  # workspace only fits one at a time
            1000000,  # logits budget is huge
            [
                (slice(0, 1), slice(0, 1)),
                (slice(1, 2), slice(0, 1)),
                (slice(2, 3), slice(0, 1)),
            ],
        ),
        # Greedy filling: first two fit, third doesn't
        # req0: M=5, N=10 -> 50 elems
        # req0+1: M=10, N=20 -> 200 elems <= 250
        # req0+1+2: M=15, N=30 -> 450 elems > 250
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            1000,  # 250 elems
            [(slice(0, 2), slice(0, 10)), (slice(2, 3), slice(0, 5))],
        ),
    ],
)
def test_split_indexer_prefill_chunks(
    seq_lens, query_lens, workspace_size, max_logits_bytes, expected
):
    out = split_indexer_prefill_chunks(
        seq_lens,
        query_lens,
        workspace_size,
        max_logits_bytes,
    )
    assert out == expected


def test_split_indexer_prefill_chunks_single_request_overflow():
    """Test that single request exceeding budget is sub-chunked on query dim."""
    seq_lens = torch.tensor([1000, 50])
    query_lens = torch.tensor([100, 5])

    out = split_indexer_prefill_chunks(seq_lens, query_lens, 2000, 1000)
    # max_logits_elems = 250, N=1000 -> max_q = 1 -> 100 query sub-chunks
    expected = [(slice(0, 1), slice(i, i + 1)) for i in range(100)]
    # req1: M=5, N=50 -> 250 elems fits budget
    expected.append((slice(1, 2), slice(0, 5)))
    assert out == expected


def test_triton_convert_returns_valid_counts():
    """Test that return_valid_counts correctly counts non-negative indices."""
    device = torch.device(DEVICE_TYPE)
    num_tokens = 8
    num_requests = 2
    max_blocks_per_req = 10
    block_size = 64
    num_topk_tokens = 128

    req_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)
    block_table = torch.arange(
        num_requests * max_blocks_per_req, dtype=torch.int32, device=device
    ).view(num_requests, max_blocks_per_req)

    # Create token indices with varying numbers of valid entries
    # Token 0: 64 valid, 64 invalid (-1)
    # Token 1: 32 valid, 96 invalid
    # Token 2: 128 valid (all)
    # Token 3: 1 valid, 127 invalid
    # etc.
    token_indices = torch.full(
        (num_tokens, num_topk_tokens), -1, dtype=torch.int32, device=device
    )
    expected_valid = []
    for i in range(num_tokens):
        num_valid = [64, 32, 128, 1, 64, 32, 128, 1][i]
        token_indices[i, :num_valid] = torch.arange(
            num_valid, dtype=torch.int32, device=device
        ) % (block_size * max_blocks_per_req)
        expected_valid.append(num_valid)

    expected_valid_tensor = torch.tensor(
        expected_valid, dtype=torch.int32, device=device
    )

    # Test with return_valid_counts=True
    result, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=True,
    )

    torch.testing.assert_close(valid_counts, expected_valid_tensor, rtol=0, atol=0)

    # Test that return_valid_counts=False returns only the indices
    result_only = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=False,
    )
    assert isinstance(result_only, torch.Tensor)
    torch.testing.assert_close(result_only, result, rtol=0, atol=0)


def test_flashmla_cache_dtype_aliases_use_ds_layout():
    from vllm.model_executor.layers.attention.mla_attention import (
        _canonicalize_sparse_mla_kv_cache_dtype,
    )

    # kv-cache dtype aliases are canonicalized to fp8_ds_mla before the layer
    # stores kv_cache_dtype, so they cannot bypass the gate.
    for alias in ("fp8", "fp8_e4m3"):
        assert (
            _canonicalize_sparse_mla_kv_cache_dtype(FlashMLASparseBackend, alias)
            == "fp8_ds_mla"
        )


def test_flashmla_fp8_metadata_reuses_common_batch_split():
    builder = SimpleNamespace(
        device=torch.device(DEVICE_TYPE),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8)),
    )
    common_metadata = SimpleNamespace(
        num_actual_tokens=1,
        seq_lens_cpu_upper_bound=torch.tensor([1]),
        query_start_loc_cpu=torch.tensor([0, 1]),
        block_table_tensor=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
    )
    metadata = FlashMLASparseMetadata(
        num_reqs=1,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], device=DEVICE_TYPE),
        slot_mapping=torch.tensor([0], device=DEVICE_TYPE),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        req_id_per_token=torch.zeros(1, dtype=torch.int32, device=DEVICE_TYPE),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
    )

    fp8_metadata = FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode(
        builder, common_metadata, metadata
    )

    assert fp8_metadata.num_decodes == 0
    assert fp8_metadata.num_prefills == 1
    assert fp8_metadata.num_decode_tokens == 0
    assert fp8_metadata.num_prefill_tokens == 1


def test_flashmla_common_metadata_requires_uniform_decodes():
    common_metadata = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=6,
        query_start_loc_cpu=torch.tensor([0, 1, 3, 6]),
        is_prefilling=None,
    )

    split = split_decodes_and_prefills(
        common_metadata,
        decode_threshold=128,
        require_uniform=FlashMLASparseMetadataBuilder.require_uniform_decodes,
    )

    assert split == (1, 2, 1, 5)


def test_flashmla_fp8_metadata_excludes_zero_token_decode_padding(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse.get_mla_metadata",
        lambda: (object(), None),
    )
    builder = SimpleNamespace(
        device=torch.device(DEVICE_TYPE),
        dummy_block_table=torch.zeros(7, 1, device=DEVICE_TYPE),
        max_model_len_tensor=torch.zeros(7, device=DEVICE_TYPE),
    )
    query_start_loc_cpu = torch.tensor([0, 110, 220, 330, 440, 550, 660, 660])
    common_metadata = SimpleNamespace(
        num_actual_tokens=660,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.arange(7, device=DEVICE_TYPE),
    )
    metadata = FlashMLASparseMetadata(
        num_reqs=7,
        max_query_len=110,
        max_seq_len=110,
        num_actual_tokens=660,
        query_start_loc=query_start_loc_cpu.to(DEVICE_TYPE),
        slot_mapping=torch.arange(660, device=DEVICE_TYPE),
        block_table=torch.zeros(7, 1, dtype=torch.int32, device=DEVICE_TYPE),
        req_id_per_token=torch.zeros(660, dtype=torch.int32, device=DEVICE_TYPE),
        num_decodes=7,
        num_prefills=0,
        num_decode_tokens=660,
    )

    fp8_metadata = FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode(
        builder, common_metadata, metadata
    )

    assert fp8_metadata.num_decodes == 6
    assert fp8_metadata.num_decode_tokens == 660
    assert fp8_metadata.decode is not None
    assert fp8_metadata.decode.decode_query_len == 110
    torch.testing.assert_close(
        fp8_metadata.decode.seq_lens, torch.arange(6, device=DEVICE_TYPE)
    )


@pytest.mark.parametrize("use_mixed_batch", [False, True])
def test_flashmla_fp8_paths_accept_decode_subset(monkeypatch, use_mixed_batch: bool):
    num_decode_tokens = 2
    num_batch_tokens = 5
    q = torch.empty(num_decode_tokens, 2, 3, device=DEVICE_TYPE)
    topk_indices = torch.empty(num_decode_tokens, 4, device=DEVICE_TYPE)
    kernel_q_shapes = []

    def convert_indices(*args, **kwargs):  # noqa: ARG001
        assert not kwargs.get("HAS_PREFILL_WORKSPACE", False)
        if not kwargs.get("return_valid_counts", False):
            return topk_indices
        valid_counts = torch.full(
            (num_decode_tokens,), 4, dtype=torch.int32, device=DEVICE_TYPE
        )
        return topk_indices, valid_counts

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_convert_req_index_to_global_index",
        convert_indices,
    )

    def run_kernel(**kwargs):
        kernel_q_shapes.append(kwargs["q"].shape)
        return kwargs["q"][..., :1], None

    if use_mixed_batch:
        fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=object(),  # type: ignore[arg-type]
            dummy_block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
            cache_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
        )
    else:
        FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
        fp8_metadata = FP8Meta(
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_batch_tokens - num_decode_tokens,
            decode=FP8Meta.Decode(
                seq_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
                kernel_metadata=object(),  # type: ignore[arg-type]
                decode_query_len=num_decode_tokens,
            ),
            prefill=FP8Meta.Prefill(
                request_ids=torch.empty(
                    num_batch_tokens, dtype=torch.int32, device=DEVICE_TYPE
                ),
                workspace_starts=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
                chunks=[],
            ),
        )
    metadata = SimpleNamespace(
        fp8_extra_metadata=fp8_metadata,
        fp8_use_mixed_batch=use_mixed_batch,
        num_actual_tokens=num_batch_tokens,
        req_id_per_token=torch.empty(
            num_batch_tokens, dtype=torch.int32, device=DEVICE_TYPE
        ),
        block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        block_size=64,
    )
    impl = SimpleNamespace(
        kv_cache_dtype="fp8_ds_mla",
        topk_indices_buffer=topk_indices,
        num_heads=2,
        kv_lora_rank=1,
        _fp8_flash_mla_kernel=run_kernel,
    )
    impl._forward_fp8_kv_mixed_batch = MethodType(
        FlashMLASparseImpl._forward_fp8_kv_mixed_batch, impl
    )
    impl._forward_fp8_kv_separate_prefill_decode = MethodType(
        FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode, impl
    )

    output, lse = FlashMLASparseImpl.forward_mqa(
        impl,
        q,
        torch.empty(0, device=DEVICE_TYPE),
        metadata,
        None,
    )

    assert kernel_q_shapes == [(1, num_decode_tokens, 2, 3)]
    assert output.shape == (num_decode_tokens, 2, 1)
    assert lse is None


def test_flashmla_direct_owner_history_uses_cached_peer_slots(monkeypatch):
    num_tokens = 3
    topk = 4
    q = torch.empty(num_tokens, 2, 3, device=DEVICE_TYPE)
    logical_topk = torch.arange(
        num_tokens * topk, dtype=torch.int32, device=DEVICE_TYPE
    ).view(num_tokens, topk)
    peer_topk = logical_topk + 100
    calls = []

    class FakeOwnerPeerSlotCache:
        def get(self, rows, block_table, **kwargs):
            calls.append((rows, block_table, kwargs))
            return peer_topk, torch.full(
                (num_tokens,), topk, dtype=torch.int32, device=DEVICE_TYPE
            )

    def reject_generic_conversion(*args, **kwargs):  # noqa: ARG001
        pytest.fail("owner history must not use the replicated slot conversion")

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_convert_req_index_to_global_index",
        reject_generic_conversion,
    )

    captured_indices = []

    def run_kernel(**kwargs):
        captured_indices.append(kwargs["topk_indices"])
        return kwargs["q"][..., :1], None

    metadata = SimpleNamespace(
        fp8_extra_metadata=FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=object(),  # type: ignore[arg-type]
            dummy_block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
            cache_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
        ),
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE_TYPE),
        block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        block_size=64,
        cp_kv_cache_interleave_size=64,
    )
    impl = SimpleNamespace(
        dcp_world_size=4,
        _fp8_flash_mla_kernel=run_kernel,
    )
    layer = SimpleNamespace(
        pcp_owner_history_direct=True,
        pcp_peer_block_stride=17,
        owner_peer_slot_cache=FakeOwnerPeerSlotCache(),
    )

    output = FlashMLASparseImpl._forward_fp8_kv_mixed_batch(
        impl,
        q,
        torch.empty(0, device=DEVICE_TYPE),
        logical_topk,
        metadata,
        layer,
    )

    assert output.shape == (num_tokens, 2, 1)
    assert len(calls) == 1
    assert calls[0][0] == num_tokens
    assert calls[0][2] == {
        "dcp_size": 4,
        "blocks_per_peer": 17,
        "cp_kv_cache_interleave_size": 64,
        "block_size": 64,
    }
    torch.testing.assert_close(captured_indices[0], peer_topk.unsqueeze(0))


def test_flashmla_owner_decode_preserves_per_request_schedule(monkeypatch):
    num_tokens = 2
    topk = 4
    q = torch.empty(num_tokens, 2, 3, device=DEVICE_TYPE)
    logical_topk = torch.arange(
        num_tokens * topk, dtype=torch.int32, device=DEVICE_TYPE
    ).view(num_tokens, topk)
    peer_topk = logical_topk + 200

    class FakeOwnerPeerSlotCache:
        def get(self, rows, block_table, **kwargs):  # noqa: ARG002
            assert rows == num_tokens
            return peer_topk, torch.full(
                (num_tokens,), topk, dtype=torch.int32, device=DEVICE_TYPE
            )

    def reject_generic_conversion(*args, **kwargs):  # noqa: ARG001
        pytest.fail("owner decode must not use replicated slot conversion")

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_convert_req_index_to_global_index",
        reject_generic_conversion,
    )
    captured = []

    def run_kernel(**kwargs):
        captured.append((kwargs["q"].shape, kwargs["topk_indices"]))
        return kwargs["q"][..., :1], None

    KernelMeta = FlashMLASparseMetadata.FP8KernelMetadata
    FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
    metadata = SimpleNamespace(
        fp8_extra_metadata=FP8Meta(
            num_decodes=num_tokens,
            num_prefills=0,
            num_decode_tokens=num_tokens,
            num_prefill_tokens=0,
            decode=FP8Meta.Decode(
                seq_lens=torch.full(
                    (num_tokens,), topk, dtype=torch.int32, device=DEVICE_TYPE
                ),
                kernel_metadata=KernelMeta(
                    scheduler_metadata=object(),  # type: ignore[arg-type]
                    dummy_block_table=torch.empty(
                        num_tokens, 1, dtype=torch.int32, device=DEVICE_TYPE
                    ),
                    cache_lens=torch.empty(
                        num_tokens, dtype=torch.int32, device=DEVICE_TYPE
                    ),
                ),
                decode_query_len=1,
            ),
        ),
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE_TYPE),
        block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        block_size=64,
        cp_kv_cache_interleave_size=64,
    )
    impl = SimpleNamespace(
        dcp_world_size=4,
        num_heads=2,
        kv_lora_rank=1,
        _fp8_flash_mla_kernel=run_kernel,
    )
    layer = SimpleNamespace(
        pcp_owner_history_direct=True,
        pcp_peer_block_stride=19,
        owner_peer_slot_cache=FakeOwnerPeerSlotCache(),
    )

    output = FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode(
        impl,
        q,
        torch.empty(0, device=DEVICE_TYPE),
        logical_topk,
        metadata,
        layer,
    )

    assert output.shape == (num_tokens, 2, 1)
    assert captured[0][0] == (num_tokens, 1, 2, 3)
    torch.testing.assert_close(captured[0][1], peer_topk.view(num_tokens, 1, topk))


def test_flashmla_owner_prefill_caches_translation_but_materializes_each_layer(
    monkeypatch,
):
    num_tokens = 2
    topk = 3
    total_history_tokens = 4
    q = torch.empty(num_tokens, 2, 3, device=DEVICE_TYPE)
    logical_topk = torch.arange(
        num_tokens * topk, dtype=torch.int32, device=DEVICE_TYPE
    ).view(num_tokens, topk)
    workspace_topk = torch.tensor(
        [[0, 2, 3], [1, 3, -1]], dtype=torch.int32, device=DEVICE_TYPE
    )
    workspace_lengths = torch.tensor([3, 2], dtype=torch.int32, device=DEVICE_TYPE)
    local_block_table = torch.tensor([[2, 5]], dtype=torch.int32, device=DEVICE_TYPE)
    translated_block_table = torch.tensor(
        [[2, 19, 36, 53, 5, 22, 39, 56]],
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    workspace_starts = torch.zeros(1, dtype=torch.int32, device=DEVICE_TYPE)

    FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
    chunk = FP8Meta.Prefill.Chunk(
        tokens_slice=slice(0, num_tokens),
        block_table=local_block_table,
        req_start_idx=0,
        workspace_starts=workspace_starts,
        chunk_tot_seqlen=total_history_tokens,
    )
    metadata = SimpleNamespace(
        fp8_extra_metadata=FP8Meta(
            num_decodes=0,
            num_prefills=1,
            num_decode_tokens=0,
            num_prefill_tokens=num_tokens,
            prefill=FP8Meta.Prefill(
                request_ids=torch.zeros(
                    num_tokens, dtype=torch.int32, device=DEVICE_TYPE
                ),
                workspace_starts=workspace_starts,
                chunks=[chunk],
            ),
        ),
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE_TYPE),
        block_table=local_block_table,
        block_size=64,
        cp_kv_cache_interleave_size=64,
    )

    conversion_calls = []

    def convert_indices(*args, **kwargs):  # noqa: ARG001
        conversion_calls.append(kwargs)
        return workspace_topk, workspace_lengths

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_convert_req_index_to_global_index",
        convert_indices,
    )

    translation_calls = []

    def translate(owner_block_tables, **kwargs):
        translation_calls.append((owner_block_tables, kwargs))
        return translated_block_table

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "build_rotated_dcp_peer_block_table",
        translate,
    )

    peer_cache = torch.empty(0, device=DEVICE_TYPE)
    materialization_calls = []

    def materialize(src, dst, block_table, starts, num_reqs):
        materialization_calls.append((src, dst, block_table, starts, num_reqs))

    monkeypatch.setattr(
        ops,
        "cp_gather_and_upconvert_fp8_kv_cache",
        materialize,
    )

    kernel_calls = []

    def run_bf16_kernel(kernel_q, workspace, indices, lengths):
        kernel_calls.append((kernel_q, workspace, indices, lengths))
        return kernel_q[..., :1]

    prefill_workspace = torch.empty(
        total_history_tokens, 576, dtype=torch.bfloat16, device=DEVICE_TYPE
    )
    impl = SimpleNamespace(
        dcp_world_size=4,
        num_heads=2,
        kv_lora_rank=1,
        prefill_bf16_workspace=prefill_workspace,
        _bf16_flash_mla_kernel=run_bf16_kernel,
    )
    layer = SimpleNamespace(
        pcp_owner_history_direct=True,
        pcp_peer_block_stride=17,
        owner_peer_slot_cache=object(),
    )

    for _ in range(2):
        output = FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode(
            impl,
            q,
            peer_cache,
            logical_topk,
            metadata,
            layer,
        )
        assert output.shape == (num_tokens, 2, 1)

    assert len(translation_calls) == 1
    owner_tables, translation_kwargs = translation_calls[0]
    assert owner_tables.shape == (4, 1, 2)
    assert owner_tables.stride(0) == 0
    assert translation_kwargs == {
        "local_rank": 0,
        "peer_block_stride": 17,
        "cp_kv_cache_interleave_size": 64,
        "block_size": 64,
    }
    assert len(materialization_calls) == 2
    assert all(call[0] is peer_cache for call in materialization_calls)
    assert all(call[2] is translated_block_table for call in materialization_calls)
    assert all(call[4] == 1 for call in materialization_calls)
    assert all(call["HAS_PREFILL_WORKSPACE"] for call in conversion_calls)
    assert len(kernel_calls) == 2
    torch.testing.assert_close(kernel_calls[0][2], workspace_topk)
    torch.testing.assert_close(kernel_calls[0][3], workspace_lengths)


def test_flashmla_owner_materialization_reuses_reserved_padded_query(monkeypatch):
    buffer_tokens = 4
    num_tokens = 2
    num_heads = 2
    padded_heads = 4
    head_dim = 3
    q_padded_buffer = torch.full(
        (buffer_tokens, padded_heads, head_dim),
        float("nan"),
        device=DEVICE_TYPE,
    )
    # Exercise a later pure-prefill chunk. Its query is a nonzero row slice of
    # the initialization-time padded workspace and must not allocate fallback
    # storage in every attention layer.
    q = q_padded_buffer[1:3, :num_heads, :]
    q.fill_(1)

    captured_q = []

    def sparse_fwd(kernel_q, *args, **kwargs):  # noqa: ARG001
        captured_q.append(kernel_q)
        output = kernel_q.new_zeros((num_tokens, padded_heads, 1))
        return output, None, None

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse.flash_mla_sparse_fwd",
        sparse_fwd,
    )

    impl = SimpleNamespace(
        num_heads=num_heads,
        prefill_padding=padded_heads,
        q_padded_buffer=q_padded_buffer,
        softmax_scale=1.0,
    )
    output = FlashMLASparseImpl._bf16_flash_mla_kernel(
        impl,
        q,
        torch.empty((4, 576), device=DEVICE_TYPE),
        torch.zeros((num_tokens, 1), dtype=torch.int32, device=DEVICE_TYPE),
    )

    assert output.shape == (num_tokens, num_heads, 1)
    assert captured_q[0].data_ptr() == q_padded_buffer[1].data_ptr()
    torch.testing.assert_close(
        captured_q[0][:, :num_heads],
        torch.ones_like(captured_q[0][:, :num_heads]),
    )
    torch.testing.assert_close(
        captured_q[0][:, num_heads:],
        torch.zeros_like(captured_q[0][:, num_heads:]),
    )


def test_flashmla_materialization_does_not_reinterpret_packed_query_as_padded(
    monkeypatch,
):
    num_tokens = 2
    num_heads = 2
    padded_heads = 4
    head_dim = 3
    q_padded_buffer = torch.full(
        (num_tokens, padded_heads, head_dim),
        float("nan"),
        device=DEVICE_TYPE,
    )
    q = q_padded_buffer.view(-1)[: num_tokens * num_heads * head_dim].view(
        num_tokens, num_heads, head_dim
    )
    q.copy_(torch.arange(q.numel(), device=DEVICE_TYPE, dtype=q.dtype).view_as(q))
    expected = q.clone()

    captured_q = []

    def sparse_fwd(kernel_q, *args, **kwargs):  # noqa: ARG001
        captured_q.append(kernel_q)
        output = kernel_q.new_zeros((num_tokens, padded_heads, 1))
        return output, None, None

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse.flash_mla_sparse_fwd",
        sparse_fwd,
    )

    impl = SimpleNamespace(
        num_heads=num_heads,
        prefill_padding=padded_heads,
        q_padded_buffer=q_padded_buffer,
        softmax_scale=1.0,
    )
    FlashMLASparseImpl._bf16_flash_mla_kernel(
        impl,
        q,
        torch.empty((4, 576), device=DEVICE_TYPE),
        torch.zeros((num_tokens, 1), dtype=torch.int32, device=DEVICE_TYPE),
    )

    assert captured_q[0].data_ptr() != q_padded_buffer.data_ptr()
    torch.testing.assert_close(captured_q[0][:, :num_heads], expected)
    torch.testing.assert_close(
        captured_q[0][:, num_heads:],
        torch.zeros_like(captured_q[0][:, num_heads:]),
    )


@pytest.mark.parametrize(
    "owner_prefill_mode,num_heads,num_prefills,num_decodes,expect_mixed",
    [
        ("auto", 64, 1, 0, True),
        ("auto", 16, 1, 0, True),
        ("direct", 64, 1, 0, True),
        ("materialize", 64, 1, 0, False),
        ("materialize", 16, 1, 0, False),
        ("materialize", 64, 1, 1, True),
        ("materialize", 64, 0, 1, False),
    ],
)
def test_flashmla_direct_pcp_selects_requested_prefill_path(
    monkeypatch,
    owner_prefill_mode: str,
    num_heads: int,
    num_prefills: int,
    num_decodes: int,
    expect_mixed: bool,
):
    monkeypatch.setenv("VLLM_PCP_OWNER_PREFILL_MODE", owner_prefill_mode)
    metadata = SimpleNamespace(num_prefills=num_prefills, num_decodes=num_decodes)
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,  # noqa: ARG005
    )
    builder = object.__new__(FlashMLASparseMetadataBuilder)
    builder.num_heads = num_heads
    builder.use_peer_pcp_fp8 = True
    builder.use_fp8_kv_cache = True
    mixed_metadata = object()
    separate_metadata = object()
    builder._build_fp8_mixed_decode_prefill = MethodType(  # type: ignore[method-assign]
        lambda self, common: mixed_metadata,
        builder,  # noqa: ARG005
    )
    builder._build_fp8_separate_prefill_decode = MethodType(  # type: ignore[method-assign]
        lambda self, common, metadata: separate_metadata,  # noqa: ARG005
        builder,
    )

    result = FlashMLASparseMetadataBuilder.build(
        builder,
        common_prefix_len=0,
        common_attn_metadata=object(),  # type: ignore[arg-type]
    )

    assert result.fp8_use_mixed_batch is expect_mixed
    expected_metadata = mixed_metadata if expect_mixed else separate_metadata
    assert result.fp8_extra_metadata is expected_metadata
