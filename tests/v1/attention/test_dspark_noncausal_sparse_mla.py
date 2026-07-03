# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for DSpark non-causal sliding-window MLA via sparse indices.

DSpark drafts a block of N tokens whose attention is NON-CAUSAL within the block:
every block token attends to the sliding window of context AND to all block
tokens (including ones at later positions than itself).

We can implement this using the existing sparse-MLA pathway by expanding the window size
to include the rest of the block tokens: instead of setting topk indices to the 127
previous tokens, we expand it to the next power of 2 (256) and include up to
swa_size + block_size - 1 topk indices, so that each query attends to the rest. The
remaining slots are filled with padding.

The sparse-MLA decode kernels (FlashMLA on SM90/SM100, FlashInfer TRTLLM on
SM100/SM120) are index-driven: each query attends over exactly the slots in its
index list, with no causal mask (see ``flash_mla_with_kvcache(..., indices=...)``
and ``_forward_decode``'s "attend only by generated indices"). The existing
``test_sparse_mla_backends`` suite already validates arbitrary index lists, but
only ones whose entries are <= the query's own position. This test suite specifically
ensures correctness of the non-causal attention case.

This reuses the harness/helpers of ``test_sparse_mla_backends.py`` (same model
shapes, fp8_ds_mla round-trip, mock indexer, MockSparseMLAAttentionLayer); only
the index construction differs.
"""

import math
from types import MethodType, SimpleNamespace

import pytest
import torch

from tests.v1.attention.test_mla_backends import (
    BatchSpec,
    MockSparseMLAAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.test_sparse_mla_backends import (
    _quantize_dequantize_fp8_ds_mla,
)
from tests.v1.attention.utils import (
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        "DSpark non-causal sparse MLA tests currently only support CUDA.",
        allow_module_level=True,
    )

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.ops import flashmla

DEVICE_TYPE = current_platform.device_type

# (window, block_size, topk_width). topk_width must be a multiple of the kernel's
# B_TOPK (= padded query-head count, 64 or 128); we use 128-multiples to cover
# both. The "wide" case needs window + block > 128 -> width must grow past 128.
_DSPARK_CONFIGS = {
    "small_block": (8, 4, 128),
    "full_window_block": (128, 5, 256),
}


def _build_dspark_noncausal_indices(
    seq_lens: list[int],
    query_lens: list[int],
    window: int,
    topk_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-token sparse indices for the DSpark non-causal block.

    For a request with context length ``ctx`` and a query block of ``q_len``
    tokens (block positions ``ctx .. ctx+q_len-1``), EVERY block query attends to
    the same set: the trailing ``window`` context positions plus all block
    positions, i.e. the contiguous range ``[max(ctx-window,0) .. ctx+q_len-1]``.
    This is non-causal: an early block query's list contains later block tokens
    (future-pointing). The list is padded to ``topk_width`` with ``-1``.
    """
    total_query_tokens = sum(query_lens)
    sparse_indices = torch.full(
        (total_query_tokens, topk_width), -1, dtype=torch.int32, device=device
    )
    gt = 0
    for s_len, q_len in zip(seq_lens, query_lens):
        ctx_len = s_len - q_len
        lo = max(ctx_len - window, 0)
        hi = ctx_len + q_len  # exclusive: window context + the full block
        idx_list = torch.arange(lo, hi, dtype=torch.int32, device=device)
        n = idx_list.numel()
        assert n <= topk_width, (
            f"index list ({n}) exceeds aligned topk width ({topk_width})"
        )
        for _ in range(q_len):
            sparse_indices[gt, :n] = idx_list
            gt += 1
    return sparse_indices


def _run_sparse_backend_vs_sdpa(
    backend_cls,
    seq_lens: list[int],
    query_lens: list[int],
    sparse_indices: torch.Tensor,
    kv_cache_dtype: str,
    block_size: int,
    num_heads: int,
    device: torch.device,
    force_future_dominance: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a sparse-MLA backend with the given per-token indices and compute a
    dense per-token SDPA reference over the SAME indices.

    Mirrors ``test_sparse_mla_backends.test_sparse_backend_decode_correctness``
    but with externally-supplied (non-causal) ``sparse_indices``.

    ``num_heads`` selects the kernel's B_TOPK (= padded q-head count): 128 -> 128,
    64 -> 64. The aligned widths (128/256) are multiples of both, so num_heads=64
    exercises the head64 decode path that the SM100 alignment assert guards.

    ``force_future_dominance`` scales the LAST block token's latent KV so it
    dominates the softmax for every query that attends to it. With random data the
    few future block tokens carry negligible attention mass (especially with a wide
    window), so causal and non-causal outputs coincide; this knob makes the
    future-token contribution provably large for the differentiation test. It is
    OFF for the correctness test (which needs sensitivity to all tokens).

    Returns (backend_output, noncausal_reference, causal_reference). The causal
    reference restricts each query to indices <= its own absolute position.
    """
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    topk_tokens = sparse_indices.shape[1]
    dtype = torch.bfloat16
    use_fp8_ds_mla_quantization = kv_cache_dtype == "fp8_ds_mla"

    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim

    max_seqlen = max(seq_lens)
    total_cache_tokens = sum(seq_lens)

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

    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(head_size)

    # Shared MLA projection weights, used by both reference and backend.
    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []
    # Causal counterpart of the reference: same index lists, but each query is
    # restricted to indices <= its own absolute position (drops future-pointing
    # block tokens). Used to prove the non-causal result is genuinely different.
    causal_reference_outputs = []

    kv_cache_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    global_token_idx = 0

    for s_len, q_len in zip(seq_lens, query_lens):
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

        if force_future_dominance:
            # Scale the last block token's latent KV so its key/value dominate the
            # softmax for any query attending to it. 4x in the latent dot makes its
            # pre-softmax score exceed the others by a wide margin, so non-causal
            # queries (which include it) diverge sharply from causal ones (which,
            # for all but the last query, exclude it). Applied before quantization
            # so cache and reference stay consistent.
            kv_c_full[s_len - 1] = kv_c_full[s_len - 1] * 4.0 + 2.0

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

        # Per-token sparse SDPA reference over the supplied (non-causal) indices.
        def _sparse_sdpa(idx_tensor, q_tok, k_mqa=k_mqa, v_mqa=v_mqa):
            k_sparse = k_mqa[idx_tensor].unsqueeze(1).expand(-1, num_heads, -1)
            v_sparse = v_mqa[idx_tensor].unsqueeze(1).expand(-1, num_heads, -1)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_tok.unsqueeze(0).transpose(1, 2),
                k_sparse.unsqueeze(0).transpose(1, 2),
                v_sparse.unsqueeze(0).transpose(1, 2),
                scale=scale,
            )
            out = out.transpose(1, 2).squeeze(0)
            out = torch.einsum("qnl,lnv->qnv", out, W_UV)
            return out.flatten(start_dim=-2)

        for q_idx in range(q_len):
            tok_sparse_idx = sparse_indices[global_token_idx]
            valid_indices = tok_sparse_idx[tok_sparse_idx >= 0].long()

            q_tok = q_mqa[q_idx : q_idx + 1]
            reference_outputs.append(_sparse_sdpa(valid_indices, q_tok))

            # Causal: drop indices pointing past this query's own position.
            abs_pos = ctx_len + q_idx
            causal_indices = valid_indices[valid_indices <= abs_pos]
            causal_reference_outputs.append(_sparse_sdpa(causal_indices, q_tok))
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
    causal_reference = torch.cat(causal_reference_outputs, dim=0)

    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    vllm_config.model_config.hf_config.index_topk = topk_tokens

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, device, arange_block_indices=True
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
        scale=kv_cache_scale,
    )

    builder = backend_cls.get_builder_cls()(
        kv_cache_spec, ["placeholder"], vllm_config, device
    )
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

    with set_current_vllm_config(vllm_config):
        impl = backend_cls.get_impl_cls()(
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
            q_scale=1.0,
            k_scale=1.0,
        )

    out_buffer = torch.empty(
        metadata.num_actual_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )
    with torch.inference_mode():
        backend_output = mock_layer.forward_impl(
            query_vllm, kv_c_vllm, k_pe_vllm, kv_cache, metadata, out_buffer
        )
    return backend_output, sdpa_reference, causal_reference


def _skip_if_backend_unavailable(backend_cls, kv_cache_dtype: str, block_size: int):
    if kv_cache_dtype not in backend_cls.supported_kv_cache_dtypes:
        pytest.skip(f"{backend_cls.get_name()} does not support {kv_cache_dtype}")
    if (
        backend_cls is FlashMLASparseBackend
        and kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_ds_mla"
    ):
        pytest.skip("FlashMLA Sparse fp8 only supports fp8_ds_mla kv-cache dtype")
    if block_size not in backend_cls.get_supported_kernel_block_sizes():
        pytest.skip(
            f"{backend_cls.get_name()} does not support block_size={block_size}"
        )
    if backend_cls is FlashMLASparseBackend:
        ok, reason = flashmla.is_flashmla_sparse_supported()
        if not ok:
            pytest.skip(reason)
    elif backend_cls is FlashInferMLASparseTRTLLMBackend:
        cap = current_platform.get_device_capability()
        if cap is None or not backend_cls.supports_compute_capability(cap):
            pytest.skip("FlashInferMLASparseTRTLLMBackend requires SM 10.x capability")


@pytest.mark.parametrize(
    "backend_cls",
    [FlashMLASparseBackend, FlashInferMLASparseTRTLLMBackend],
    ids=["FlashMLA", "FlashInferTRTLLM"],
)
@pytest.mark.parametrize("config_name", list(_DSPARK_CONFIGS.keys()))
# Per backend, the skip logic routes fp8 to the supported flavor: FlashMLA tests
# auto + fp8_ds_mla (and skips per-tensor "fp8", which it aliases to ds_mla);
# FlashInfer TRTLLM tests auto + per-tensor "fp8" (and skips fp8_ds_mla, which it
# does not implement). So both backends get a bf16 case and an fp8 case.
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_ds_mla", "fp8"])
@pytest.mark.parametrize("block_size", [64])
# h_q=128 -> B_TOPK=128; h_q=64 -> B_TOPK=64 (covers the head64 decode path the
# SM100 alignment assert specifically guards). Aligned widths (128/256) satisfy both.
@pytest.mark.parametrize("num_heads", [128, 64], ids=["h128", "h64"])
def test_dspark_noncausal_sparse_mla_matches_sdpa(
    default_vllm_config,
    dist_init,
    workspace_init,
    backend_cls,
    config_name,
    kv_cache_dtype,
    block_size,
    num_heads,
):
    """Non-causal (window ∪ block, future-pointing) per-token indices must match
    a dense SDPA reference over the same indices, for both sparse-MLA backends."""
    _skip_if_backend_unavailable(backend_cls, kv_cache_dtype, block_size)

    window, block, topk_width = _DSPARK_CONFIGS[config_name]
    device = torch.device(DEVICE_TYPE)

    # Decode-style batch: each request has `block` query tokens and enough
    # context for a full sliding window.
    seq_lens = [window + block + 123, window + block + 50]
    query_lens = [block, block]

    sparse_indices = _build_dspark_noncausal_indices(
        seq_lens, query_lens, window, topk_width, device
    )

    # Sanity: the construction must actually be non-causal (an early block query
    # must reference a later block position than itself).
    ctx0 = seq_lens[0] - query_lens[0]
    first_query_valid = sparse_indices[0][sparse_indices[0] >= 0]
    assert int(first_query_valid.max()) >= ctx0 + query_lens[0] - 1, (
        "expected the first block query to attend to a future block token"
    )

    backend_output, sdpa_reference, _ = _run_sparse_backend_vs_sdpa(
        backend_cls,
        seq_lens,
        query_lens,
        sparse_indices,
        kv_cache_dtype,
        block_size,
        num_heads,
        device,
    )

    assert backend_output.shape == sdpa_reference.shape
    assert backend_output.dtype == sdpa_reference.dtype
    assert torch.isfinite(backend_output).all()
    if kv_cache_dtype.startswith("fp8"):
        rtol, atol = 0.065, 0.05
    else:
        rtol, atol = 0.01, 0.01
    torch.testing.assert_close(backend_output, sdpa_reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "backend_cls",
    [FlashMLASparseBackend, FlashInferMLASparseTRTLLMBackend],
    ids=["FlashMLA", "FlashInferTRTLLM"],
)
@pytest.mark.parametrize("config_name", list(_DSPARK_CONFIGS.keys()))
@pytest.mark.parametrize("block_size", [64])
def test_dspark_noncausal_differs_from_causal(
    default_vllm_config,
    dist_init,
    workspace_init,
    backend_cls,
    config_name,
    block_size,
):
    """Differentiation guard: prove the backend genuinely attends to the
    future-pointing indices (not silently applying a causal mask, and not merely
    coinciding with a causal result because future tokens carry little weight).

    With random data the few future block tokens are a negligible fraction of the
    attended set (especially with a wide window), so causal and non-causal outputs
    are numerically indistinguishable -- that is correct physics, not a backend
    bug. To make the check meaningful we use ``force_future_dominance`` so the last
    block token dominates the softmax: the backend must then match the non-causal
    reference and diverge sharply from the causal one. bf16 (``auto``) suffices;
    the property is dtype-independent and fp8 correctness is covered above.
    """
    _skip_if_backend_unavailable(backend_cls, "auto", block_size)

    window, block, topk_width = _DSPARK_CONFIGS[config_name]
    device = torch.device(DEVICE_TYPE)
    seq_lens = [window + block + 123, window + block + 50]
    query_lens = [block, block]

    sparse_indices = _build_dspark_noncausal_indices(
        seq_lens, query_lens, window, topk_width, device
    )

    backend_output, sdpa_reference, causal_reference = _run_sparse_backend_vs_sdpa(
        backend_cls,
        seq_lens,
        query_lens,
        sparse_indices,
        "auto",
        block_size,
        128,
        device,
        force_future_dominance=True,
    )

    # The two references must be clearly distinguishable for the check to mean
    # anything (dominance guarantees this).
    ref_gap = (sdpa_reference - causal_reference).abs().max().item()
    assert ref_gap > 0.1, (
        f"non-causal and causal references are too close (gap={ref_gap}); "
        "force_future_dominance did not create a separable scenario"
    )

    # Backend must track the NON-causal reference, not the causal one.
    torch.testing.assert_close(backend_output, sdpa_reference, rtol=0.01, atol=0.01)
    causal_err = (backend_output - causal_reference).abs().max().item()
    assert causal_err > 0.1, (
        f"non-causal backend output matches the causal reference "
        f"(max abs diff={causal_err}); future-pointing indices are not attended to"
    )
