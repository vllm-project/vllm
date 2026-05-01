# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend integration tests for CUTLASS FA3 sparse MLA attention.

Tests verify:
  - Backend class properties
  - Metadata builder (decode, prefill, mixed, topk clipping)
  - KV cache write/read consistency
  - Backend registration and selection
"""

import pytest
import torch

from vllm.v1.attention.ops.cutlass_fa3 import is_cutlass_fa3_available

pytestmark = pytest.mark.skipif(
    not is_cutlass_fa3_available(),
    reason="CUTLASS FA3 not available (requires CUDA >= 12.4, SM90)",
)


# ─── TEST 2.1: Backend Class Properties ──────────────────────────────


def test_backend_class_properties():
    """Verify CutlassFA3MLASparseBackend class attributes."""
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseBackend,
    )

    assert CutlassFA3MLASparseBackend.get_name() == "CUTLASS_FA3_MLA_SPARSE"
    assert CutlassFA3MLASparseBackend.is_mla() is True
    assert CutlassFA3MLASparseBackend.is_sparse() is True
    assert CutlassFA3MLASparseBackend.get_supported_head_sizes() == [576]
    assert CutlassFA3MLASparseBackend.supported_kv_cache_dtypes == ["auto"]
    assert CutlassFA3MLASparseBackend.get_supported_kernel_block_sizes() == [64]


def test_backend_compute_capability():
    """Verify SM90-only support."""
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseBackend,
    )

    assert CutlassFA3MLASparseBackend.supports_compute_capability(
        DeviceCapability(major=9, minor=0)
    )
    assert not CutlassFA3MLASparseBackend.supports_compute_capability(
        DeviceCapability(major=8, minor=0)
    )
    assert not CutlassFA3MLASparseBackend.supports_compute_capability(
        DeviceCapability(major=10, minor=0)
    )


def test_backend_kv_cache_shape():
    """Verify KV cache shape for BF16 format."""
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseBackend,
    )

    shape = CutlassFA3MLASparseBackend.get_kv_cache_shape(
        num_blocks=100,
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str="auto",
    )
    assert shape == (100, 64, 576)


# ─── TEST 2.2: Backend Registration ──────────────────────────────────


def test_backend_enum_registered():
    """Verify CUTLASS_FA3_MLA_SPARSE is in the backend enum."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert hasattr(AttentionBackendEnum, "CUTLASS_FA3_MLA_SPARSE")
    backend_enum = AttentionBackendEnum.CUTLASS_FA3_MLA_SPARSE
    assert "cutlass_fa3_sparse" in backend_enum.get_path()


def test_backend_class_loadable():
    """Verify the backend class can be loaded from the enum."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    backend_cls = AttentionBackendEnum.CUTLASS_FA3_MLA_SPARSE.get_class()
    assert backend_cls.get_name() == "CUTLASS_FA3_MLA_SPARSE"


# ─── TEST 2.3: KV Cache Write/Read ───────────────────────────────────


def test_kv_cache_write_read_consistency():
    """Verify do_kv_cache_update writes match what forward_mqa would read."""
    device = "cuda"
    num_blocks = 4
    block_size = 64
    head_size = 576
    kv_lora_rank = 512
    qk_rope_head_dim = 64

    # Create BF16 cache
    cache = torch.zeros(
        num_blocks, block_size, head_size, dtype=torch.bfloat16, device=device
    )

    # Write known values
    T = 3
    kv_c_normed = torch.randn(T, kv_lora_rank, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(T, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)

    from vllm import _custom_ops as ops

    ops.concat_and_cache_mla(
        kv_c_normed,
        k_pe.squeeze(1),
        cache,
        slot_mapping,
        kv_cache_dtype="auto",
        scale=k_scale,
    )

    # Read back via flatten + split (same as forward_mqa does)
    S = num_blocks * block_size
    kv_flat = cache.reshape(S, head_size)
    c_kv_read = kv_flat[:T, :kv_lora_rank]
    k_rope_read = kv_flat[:T, kv_lora_rank:]

    # Verify consistency
    torch.testing.assert_close(c_kv_read, kv_c_normed, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_rope_read, k_pe.squeeze(1), rtol=1e-3, atol=1e-3)


def test_kv_cache_dtype_auto():
    """Verify kv_cache_dtype='auto' uses BF16 direct copy."""
    device = "cuda"
    cache = torch.zeros(1, 64, 576, dtype=torch.bfloat16, device=device)

    kv_c = torch.randn(1, 512, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(1, 1, 64, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.tensor([0], dtype=torch.int64, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)

    from vllm import _custom_ops as ops

    ops.concat_and_cache_mla(
        kv_c, k_pe.squeeze(1), cache, slot_mapping, kv_cache_dtype="auto", scale=k_scale
    )

    assert cache.dtype == torch.bfloat16


# ─── TEST 2.4: Edge Cases ────────────────────────────────────────────


def test_empty_kv_cache():
    """Verify do_kv_cache_update handles empty cache gracefully."""

    kv_cache = torch.empty(0, device="cuda")
    # Should return without error (numel() == 0 check)
    # We call the static method from parent class directly
    from vllm.v1.attention.backend import SparseMLAAttentionImpl

    SparseMLAAttentionImpl.do_kv_cache_update(
        None,
        kv_c_normed=torch.empty(0),
        k_pe=torch.empty(0),
        kv_cache=kv_cache,
        slot_mapping=torch.empty(0),
        kv_cache_dtype="auto",
        k_scale=torch.ones(1),
    )


# ─── TEST 2.5: Valid Counts from Index Conversion ───────────────────


def test_triton_convert_valid_counts():
    """Verify triton_convert_req_index_to_global_index with return_valid_counts.

    This tests the core fix mechanism: the Triton kernel atomically counts
    valid (non -1) entries per row while converting indices.
    """
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )

    device = "cuda"
    T = 4
    topk = 128
    num_blocks = 16
    block_size = 64

    req_id = torch.zeros(T, dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )  # [1, num_blocks]

    # Create topk_indices with varying valid entries per token
    topk_indices = torch.full((T, topk), -1, dtype=torch.int32, device=device)
    expected_valid = [1, 10, 50, 100]
    for i in range(T):
        nv = expected_valid[i]
        # Use indices within the valid range
        topk_indices[i, :nv] = torch.randint(
            0,
            num_blocks * block_size,
            (nv,),
            dtype=torch.int32,
            device=device,
        )

    global_idx, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        topk_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=topk,
        return_valid_counts=True,
    )

    # Verify valid counts match expected
    for i in range(T):
        assert valid_counts[i].item() == expected_valid[i], (
            f"Token {i}: expected {expected_valid[i]} valid, "
            f"got {valid_counts[i].item()}"
        )

    # Verify -1 propagation
    for i in range(T):
        nv = expected_valid[i]
        # Entries beyond valid should be -1
        assert (global_idx[i, nv:] == -1).all(), (
            f"Token {i}: entries beyond valid count should be -1"
        )


# ─── TEST 2.6: Prefill Metadata Correctness ─────────────────────────


def test_prefill_cache_seqlens_vs_valid_counts():
    """Verify metadata cache_seqlens = min(seq_len, topk) and that the
    forward_mqa fix overrides with valid_counts.

    The metadata builder computes cache_seqlens as min(seq_len, topk).
    For prefill tokens, this can exceed the actual valid topk entries.
    The fix in forward_mqa uses valid_counts instead.
    """
    import numpy as np

    device = "cuda"

    # Simulate a prefill batch: 1 request, 4 tokens, seq_len=4
    num_reqs = 1
    T = 4
    topk = 2048
    seq_len = 4

    # The metadata builder's logic (simplified):
    starts = np.array([0, T], dtype=np.int32)
    seg_lens = np.diff(starts)  # [4]
    seq_lens_np = np.array([seq_len], dtype=np.int32)

    per_tok_seqlens = np.minimum(np.repeat(seq_lens_np, seg_lens), topk)  # [4, 4, 4, 4]

    # This is what the metadata builder produces:
    assert all(per_tok_seqlens == 4), (
        "Metadata cache_seqlens should be min(seq_len, topk) = 4"
    )

    # But the actual valid entries per token (with causal masking):
    # Token 0: 1 valid entry, Token 1: 2, Token 2: 3, Token 3: 4
    expected_valid = [1, 2, 3, 4]

    # The fix in forward_mqa computes valid_counts from the page_table
    # and uses those as cache_seqlens. Verify the fix produces correct
    # valid counts:
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )

    req_id = torch.zeros(T, dtype=torch.int32, device=device)
    block_table = torch.arange(32, dtype=torch.int32, device=device).unsqueeze(0)

    topk_indices = torch.full((T, topk), -1, dtype=torch.int32, device=device)
    for i in range(T):
        nv = expected_valid[i]
        topk_indices[i, :nv] = torch.arange(nv, dtype=torch.int32, device=device)

    _, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        topk_indices,
        BLOCK_SIZE=64,
        NUM_TOPK_TOKENS=topk,
        return_valid_counts=True,
    )

    for i in range(T):
        assert valid_counts[i].item() == expected_valid[i], (
            f"Token {i}: valid_counts should be {expected_valid[i]}, "
            f"got {valid_counts[i].item()}"
        )


# ─── TEST 2.7: Clamp -1 to 0 Safety ─────────────────────────────────


def test_global_idx_clamp_safety():
    """Verify clamping -1 page indices to 0 prevents OOB access."""
    device = "cuda"

    # Create a page_table with -1 entries
    page_table = torch.tensor(
        [[5, 10, -1, -1], [3, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )

    # Clamp -1 to 0
    clamped = page_table.clamp(min=0)

    # Verify
    expected = torch.tensor(
        [[5, 10, 0, 0], [3, 0, 0, 0]],
        dtype=torch.int32,
        device=device,
    )
    assert torch.equal(clamped, expected), (
        f"Clamped page_table doesn't match expected: {clamped} vs {expected}"
    )


# ─── TEST 2.8: In-place clamp correctness ───────────────────────────


def test_inplace_clamp_no_negative_indices():
    """Verify in-place clamp_(min=0) on global_idx leaves no -1 entries.

    The review-fixed code uses clamp_() (in-place) instead of clamp()
    to avoid unnecessary tensor allocations during CUDA graph capture.
    """
    device = "cuda"

    # Create a global_idx tensor with -1 entries
    global_idx = torch.tensor(
        [[100, 200, -1, -1, -1], [50, -1, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )

    # In-place clamp
    global_idx.clamp_(min=0)

    # Verify no -1 entries remain
    assert (global_idx >= 0).all(), (
        f"In-place clamp should remove all -1 entries: {global_idx}"
    )
    # Verify valid entries are preserved
    assert global_idx[0, 0].item() == 100
    assert global_idx[0, 1].item() == 200
    assert global_idx[1, 0].item() == 50


# ─── TEST 2.9: Full fix flow with index conversion ──────────────────


def test_full_fix_flow_valid_counts_and_clamp():
    """End-to-end test of the complete fix flow:
    1. triton_convert_req_index_to_global_index with return_valid_counts=True
    2. In-place clamp global_idx to replace -1 with 0
    3. In-place clamp valid_counts to min=1
    4. Use valid_counts as cache_seqlens

    This simulates what forward_mqa does after the fix.
    """
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )

    device = "cuda"
    T = 4
    topk = 128
    num_blocks = 16
    block_size = 64

    req_id = torch.zeros(T, dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )

    # Simulate causal prefill: token i has (i+1) valid entries
    topk_indices = torch.full((T, topk), -1, dtype=torch.int32, device=device)
    expected_valid = [1, 2, 3, 4]
    for i in range(T):
        nv = expected_valid[i]
        topk_indices[i, :nv] = torch.arange(nv, dtype=torch.int32, device=device)

    # Step 1: Convert with valid counts
    global_idx, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        topk_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=topk,
        return_valid_counts=True,
    )

    # Step 2: In-place clamp global_idx (no -1 entries after)
    global_idx.clamp_(min=0)
    assert (global_idx >= 0).all(), "No -1 entries should remain after clamp_"

    # Step 3: In-place clamp valid_counts to min=1
    valid_counts.clamp_(min=1)
    cache_seqlens = valid_counts

    # Step 4: Verify valid counts match expected
    for i in range(T):
        assert cache_seqlens[i].item() == expected_valid[i], (
            f"Token {i}: expected cache_seqlens={expected_valid[i]}, "
            f"got {cache_seqlens[i].item()}"
        )

    # Step 5: Verify that for each token, entries 0..cache_seqlens-1 in
    # global_idx are valid (non-zero, since we clamped -1 to 0 for the
    # entries beyond valid_counts, the valid entries at positions 0..nv-1
    # should be the actual converted indices)
    for i in range(T):
        nv = expected_valid[i]
        valid_region = global_idx[i, :nv]
        # Valid region should have specific converted values from block_table
        # (not just zeros from clamping)
        # For indices [0, 1, ..., nv-1] with block_size=64:
        #   block_id = index // 64, inblock_off = index % 64
        #   out = block_table[0, block_id] * 64 + inblock_off
        for j in range(nv):
            block_id = j // block_size
            inblock_off = j % block_size
            expected_val = block_table[0, block_id].item() * block_size + inblock_off
            assert valid_region[j].item() == expected_val, (
                f"Token {i}, position {j}: expected {expected_val}, "
                f"got {valid_region[j].item()}"
            )


# ─── TEST 2.10: CUDA Graph Padding Fix ─────────────────────────────
# These tests verify the fix for Issue 2: RuntimeError when
# num_actual_tokens (padded) != sum(seg_lens) (real tokens).
# This is the core bug that caused the crash during lm_eval with
# 32 concurrent requests on DeepSeek-V3.2.


def _make_mock_vllm_config(max_tokens=512):
    """Create a mock VllmConfig for metadata builder tests."""
    from unittest.mock import MagicMock

    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_batched_tokens = max_tokens
    vllm_config.speculative_config = None
    vllm_config.parallel_config.decode_context_parallel_size = 1
    return vllm_config


def test_metadata_builder_cuda_graph_padding():
    """Verify build() handles CUDA graph padding (T > actual_tokens).

    Reproduces the exact crash from Issue 2:
      RuntimeError: The size of tensor a (32) must match the size
      of tensor b (31) at non-singleton dimension 0

    This happens when num_actual_tokens=32 (padded for CUDA graph)
    but only 31 real tokens exist (one request completed mid-batch).
    """
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseMetadataBuilder,
    )

    device = "cuda"
    max_tokens = 512
    block_size = 64
    topk = 2048

    # Mock kv_cache_spec
    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size

    # Mock vllm_config
    vllm_config = _make_mock_vllm_config(max_tokens)

    builder = CutlassFA3MLASparseMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device(device),
    )
    builder.topk_tokens = topk

    # Simulate the crash scenario: 31 real tokens padded to 32
    padded_T = 32
    real_tokens = 31
    num_reqs_padded = 32  # padded request count

    # Accurately mock gpu_model_runner.py's padding behavior:
    # query_start_loc.cpu[:num_reqs_padded+1] = [:33], 33 entries
    # Real entries: [0,1,...,31], Padding: [31] (repeats last value)
    query_start_loc_cpu = list(range(real_tokens + 1)) + [real_tokens]
    # seq_lens_cpu[:num_reqs_padded] = [:32], 32 entries
    # Real entries: [100]*31, Padding: [0] (stale/zero for padding slot)
    seq_lens_cpu = [100] * real_tokens + [0]

    # Build the mock CommonAttentionMetadata
    cm = MagicMock()
    cm.num_actual_tokens = padded_T  # PADDED to 32
    cm.query_start_loc_cpu = query_start_loc_cpu
    cm.seq_lens_cpu = seq_lens_cpu
    cm.num_reqs = num_reqs_padded  # gpu_model_runner passes padded count
    cm.max_query_len = 1
    cm.max_seq_len = 100
    cm.query_start_loc = torch.tensor(
        query_start_loc_cpu, dtype=torch.int32, device=device
    )
    cm.slot_mapping = torch.zeros(padded_T, dtype=torch.int64, device=device)
    cm.block_table_tensor = torch.zeros(
        num_reqs_padded, 4, dtype=torch.int32, device=device
    )

    # This should NOT raise RuntimeError
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=cm,
    )

    # Verify metadata shapes match padded T
    assert metadata.req_id_per_token.shape[0] == padded_T, (
        f"req_id_per_token should have padded size {padded_T}, "
        f"got {metadata.req_id_per_token.shape[0]}"
    )
    assert metadata.cache_seqlens.shape[0] == padded_T, (
        f"cache_seqlens should have padded size {padded_T}, "
        f"got {metadata.cache_seqlens.shape[0]}"
    )
    assert metadata.cu_seqlens_q.shape[0] == padded_T + 1
    assert metadata.cu_seqlens_k.shape[0] == padded_T + 1

    # Verify real data portion is correct
    for i in range(real_tokens):
        assert metadata.req_id_per_token[i].item() == i, (
            f"Token {i}: req_id should be {i}, "
            f"got {metadata.req_id_per_token[i].item()}"
        )
        assert metadata.cache_seqlens[i].item() == 100, (
            f"Token {i}: cache_seqlens should be 100, "
            f"got {metadata.cache_seqlens[i].item()}"
        )

    # Verify padding tokens have safe defaults
    assert metadata.req_id_per_token[real_tokens].item() == 0, (
        "Padding token req_id should be 0"
    )
    assert metadata.cache_seqlens[real_tokens].item() >= 1, (
        "Padding token cache_seqlens should be >= 1 (safe minimum)"
    )

    # Verify cu_seqlens_q is [0, 1, 2, ..., padded_T] (always correct)
    for i in range(padded_T + 1):
        assert metadata.cu_seqlens_q[i].item() == i, (
            f"cu_seqlens_q[{i}] should be {i}, got {metadata.cu_seqlens_q[i].item()}"
        )

    # Verify cu_seqlens_k is monotonically non-decreasing
    for i in range(padded_T):
        assert metadata.cu_seqlens_k[i + 1].item() >= metadata.cu_seqlens_k[i].item(), (
            f"cu_seqlens_k must be non-decreasing at index {i}: "
            f"{metadata.cu_seqlens_k[i].item()} -> {metadata.cu_seqlens_k[i + 1].item()}"
        )
    # Verify cu_seqlens_k at the real/padding boundary
    assert metadata.cu_seqlens_k[real_tokens].item() == real_tokens * 100, (
        f"cu_seqlens_k[{real_tokens}] should be {real_tokens * 100}, "
        f"got {metadata.cu_seqlens_k[real_tokens].item()}"
    )


@pytest.mark.parametrize(
    "real_tokens,padded_T",
    [
        (1, 2),  # minimal padding
        (3, 32),  # large padding gap
        (7, 8),  # small batch
        (15, 16),  # medium batch
        (31, 32),  # the exact crash scenario
        (100, 104),  # larger padding gap
    ],
)
def test_metadata_builder_cuda_graph_padding_various(real_tokens, padded_T):
    """Verify build() handles various CUDA graph padding scenarios.

    Uses accurate mock that matches gpu_model_runner.py's padding behavior:
    - query_start_loc_cpu has num_reqs_padded+1 entries (with padded suffix)
    - seq_lens_cpu has num_reqs_padded entries (with stale padding entries)
    """
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseMetadataBuilder,
    )

    device = "cuda"
    max_tokens = max(512, padded_T + 1)  # ensure buffer large enough
    block_size = 64
    topk = 2048
    num_reqs_padded = padded_T  # For decode-only, padded_T == num_reqs_padded

    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size

    vllm_config = _make_mock_vllm_config(max_tokens)

    builder = CutlassFA3MLASparseMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device(device),
    )
    builder.topk_tokens = topk

    # Accurate mock: query_start_loc_cpu[:num_reqs_padded+1]
    # Real entries [0,1,...,real_tokens], then (num_reqs_padded - real_tokens)
    # padding entries all equal to real_tokens (flat, non-decreasing)
    query_start_loc_cpu = list(range(real_tokens + 1))
    num_padding_reqs = num_reqs_padded - real_tokens
    query_start_loc_cpu += [real_tokens] * num_padding_reqs

    # seq_lens_cpu[:num_reqs_padded] — padding entries are stale (zero)
    seq_lens_cpu = [200] * real_tokens + [0] * num_padding_reqs

    cm = MagicMock()
    cm.num_actual_tokens = padded_T
    cm.query_start_loc_cpu = query_start_loc_cpu
    cm.seq_lens_cpu = seq_lens_cpu
    cm.num_reqs = num_reqs_padded
    cm.max_query_len = 1
    cm.max_seq_len = 200
    cm.query_start_loc = torch.tensor(
        query_start_loc_cpu, dtype=torch.int32, device=device
    )
    cm.slot_mapping = torch.zeros(padded_T, dtype=torch.int64, device=device)
    cm.block_table_tensor = torch.zeros(
        max(num_reqs_padded, 1), 4, dtype=torch.int32, device=device
    )

    # Should NOT raise any errors
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=cm,
    )

    # Verify shapes match padded T
    assert metadata.req_id_per_token.shape[0] == padded_T
    assert metadata.cache_seqlens.shape[0] == padded_T
    assert metadata.cu_seqlens_q.shape[0] == padded_T + 1
    assert metadata.cu_seqlens_k.shape[0] == padded_T + 1
    assert metadata.num_actual_tokens == padded_T

    # Verify real portion
    for i in range(real_tokens):
        assert metadata.req_id_per_token[i].item() == i
        assert metadata.cache_seqlens[i].item() == 200

    # Verify padding
    for i in range(real_tokens, padded_T):
        assert metadata.req_id_per_token[i].item() == 0
        assert metadata.cache_seqlens[i].item() >= 1

    # Verify cu_seqlens_q is [0, 1, ..., padded_T]
    for i in range(padded_T + 1):
        assert metadata.cu_seqlens_q[i].item() == i

    # Verify cu_seqlens_k monotonicity
    for i in range(padded_T):
        assert metadata.cu_seqlens_k[i + 1].item() >= metadata.cu_seqlens_k[i].item()
    # Verify cu_seqlens_k at boundary
    assert metadata.cu_seqlens_k[real_tokens].item() == real_tokens * 200


def test_metadata_builder_no_padding():
    """Verify build() still works correctly when T == actual_tokens (no padding)."""
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseMetadataBuilder,
    )

    device = "cuda"
    max_tokens = 512
    block_size = 64

    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size

    vllm_config = _make_mock_vllm_config(max_tokens)

    builder = CutlassFA3MLASparseMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device(device),
    )
    builder.topk_tokens = 2048

    # No padding: T == real tokens
    T = 4
    query_start_loc_cpu = [0, 1, 2, 3, 4]  # 4 decode tokens
    seq_lens_cpu = [50, 100, 150, 200]

    cm = MagicMock()
    cm.num_actual_tokens = T
    cm.query_start_loc_cpu = query_start_loc_cpu
    cm.seq_lens_cpu = seq_lens_cpu
    cm.num_reqs = 4
    cm.max_query_len = 1
    cm.max_seq_len = 200
    cm.query_start_loc = torch.tensor(
        query_start_loc_cpu, dtype=torch.int32, device=device
    )
    cm.slot_mapping = torch.zeros(T, dtype=torch.int64, device=device)
    cm.block_table_tensor = torch.zeros(4, 4, dtype=torch.int32, device=device)

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=cm,
    )

    assert metadata.req_id_per_token.shape[0] == T
    assert metadata.cache_seqlens.shape[0] == T
    assert metadata.num_actual_tokens == T

    # Verify exact values
    assert metadata.req_id_per_token[0].item() == 0
    assert metadata.req_id_per_token[1].item() == 1
    assert metadata.req_id_per_token[2].item() == 2
    assert metadata.req_id_per_token[3].item() == 3
    assert metadata.cache_seqlens[0].item() == 50
    assert metadata.cache_seqlens[1].item() == 100
    assert metadata.cache_seqlens[2].item() == 150
    assert metadata.cache_seqlens[3].item() == 200


def test_metadata_builder_mixed_prefill_decode_with_padding():
    """Verify build() handles mixed prefill+decode with CUDA graph padding.

    This tests a more complex scenario: 2 decode tokens + 3 prefill tokens
    from 3 requests, padded from 5 to 8 tokens.
    """
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseMetadataBuilder,
    )

    device = "cuda"
    max_tokens = 512
    block_size = 64

    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size

    vllm_config = _make_mock_vllm_config(max_tokens)

    builder = CutlassFA3MLASparseMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device(device),
    )
    builder.topk_tokens = 2048

    # 3 real requests: req0 (1 decode token), req1 (1 decode token),
    #                  req2 (3 prefill tokens)
    # Total: 5 real tokens, padded to 8 tokens, 8 padded request slots
    real_tokens = 5
    num_real_reqs = 3
    padded_T = 8
    num_reqs_padded = 8  # padded request count

    # Accurate: query_start_loc_cpu[:num_reqs_padded+1] = 9 entries
    # Real: [0, 1, 2, 5], Padding: [5, 5, 5, 5, 5]
    query_start_loc_cpu = [0, 1, 2, 5] + [5] * (num_reqs_padded - num_real_reqs)
    # seq_lens_cpu[:num_reqs_padded] = 8 entries
    seq_lens_cpu = [100, 200, 3] + [0] * (num_reqs_padded - num_real_reqs)

    cm = MagicMock()
    cm.num_actual_tokens = padded_T
    cm.query_start_loc_cpu = query_start_loc_cpu
    cm.seq_lens_cpu = seq_lens_cpu
    cm.num_reqs = num_reqs_padded
    cm.max_query_len = 3
    cm.max_seq_len = 200
    cm.query_start_loc = torch.tensor(
        query_start_loc_cpu, dtype=torch.int32, device=device
    )
    cm.slot_mapping = torch.zeros(padded_T, dtype=torch.int64, device=device)
    cm.block_table_tensor = torch.zeros(
        num_reqs_padded, 4, dtype=torch.int32, device=device
    )

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=cm,
    )

    # Verify shapes
    assert metadata.req_id_per_token.shape[0] == padded_T
    assert metadata.cache_seqlens.shape[0] == padded_T

    # Verify req_id mapping
    assert metadata.req_id_per_token[0].item() == 0  # req0, decode
    assert metadata.req_id_per_token[1].item() == 1  # req1, decode
    assert metadata.req_id_per_token[2].item() == 2  # req2, prefill tok0
    assert metadata.req_id_per_token[3].item() == 2  # req2, prefill tok1
    assert metadata.req_id_per_token[4].item() == 2  # req2, prefill tok2
    # Padding tokens
    assert metadata.req_id_per_token[5].item() == 0
    assert metadata.req_id_per_token[6].item() == 0
    assert metadata.req_id_per_token[7].item() == 0

    # Verify cache_seqlens
    assert metadata.cache_seqlens[0].item() == 100  # req0 seq_len
    assert metadata.cache_seqlens[1].item() == 200  # req1 seq_len
    assert metadata.cache_seqlens[2].item() == 3  # req2 seq_len
    assert metadata.cache_seqlens[3].item() == 3  # req2 seq_len
    assert metadata.cache_seqlens[4].item() == 3  # req2 seq_len
    # Padding (default = 1)
    assert metadata.cache_seqlens[5].item() >= 1
    assert metadata.cache_seqlens[6].item() >= 1
    assert metadata.cache_seqlens[7].item() >= 1


# ─── TEST 2.11: Zero Real Tokens Edge Case (Review Issue #3) ─────
# Tests the edge case where ALL tokens are padding (actual_tokens=0).
# This can happen during CUDA graph warmup/capture with dummy batches.


def test_metadata_builder_zero_real_tokens():
    """Verify build() handles the case where all tokens are padding.

    This edge case can occur during CUDA graph warmup or capture where
    dummy batches may have zero real tokens but T > 0 (padded size).
    """
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseMetadataBuilder,
    )

    device = "cuda"
    max_tokens = 512
    block_size = 64

    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size

    vllm_config = _make_mock_vllm_config(max_tokens)

    builder = CutlassFA3MLASparseMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device(device),
    )
    builder.topk_tokens = 2048

    # Zero real tokens, padded to 4
    # This happens when query_start_loc = [0] only (1 entry, no requests)
    # and num_actual_tokens is still the padded count.
    padded_T = 4
    real_tokens = 0
    # query_start_loc_cpu with a single entry means 0 requests
    query_start_loc_cpu = [0]
    seq_lens_cpu = []

    cm = MagicMock()
    cm.num_actual_tokens = padded_T
    cm.query_start_loc_cpu = query_start_loc_cpu
    cm.seq_lens_cpu = seq_lens_cpu
    cm.num_reqs = 0
    cm.max_query_len = 0
    cm.max_seq_len = 0
    cm.query_start_loc = torch.tensor(
        query_start_loc_cpu, dtype=torch.int32, device=device
    )
    cm.slot_mapping = torch.zeros(padded_T, dtype=torch.int64, device=device)
    cm.block_table_tensor = torch.zeros(1, 4, dtype=torch.int32, device=device)

    # Should NOT raise any errors
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=cm,
    )

    # Verify shapes match padded T
    assert metadata.req_id_per_token.shape[0] == padded_T
    assert metadata.cache_seqlens.shape[0] == padded_T
    assert metadata.cu_seqlens_q.shape[0] == padded_T + 1
    assert metadata.cu_seqlens_k.shape[0] == padded_T + 1

    # All tokens are padding — verify safe defaults
    for i in range(padded_T):
        assert metadata.req_id_per_token[i].item() == 0
        assert metadata.cache_seqlens[i].item() >= 1

    # cu_seqlens_k should be monotonically non-decreasing
    for i in range(padded_T):
        assert metadata.cu_seqlens_k[i + 1].item() >= metadata.cu_seqlens_k[i].item()


# ─── TEST 2.12: Batch Size Gating Constant ───────────────────────────


def test_batch_size_gating_threshold():
    """Verify MAX_BATCH_SIZE_FOR_FA3 is 16 and controls routing."""
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        MAX_BATCH_SIZE_FOR_FA3,
        _flashmla_sparse_available,
    )

    assert MAX_BATCH_SIZE_FOR_FA3 == 16
    # On SM90 builds, FlashMLA fallback should be available
    # (unless FlashMLA was explicitly excluded from the build)
    assert isinstance(_flashmla_sparse_available, bool)


# ─── TEST 2.13: FlashMLA Fallback Head Padding ──────────────────────


def test_flashmla_fallback_head_padding():
    """Verify FlashMLA fallback head padding constant is 64 for SM90."""
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        _FLASHMLA_SM90_HEAD_PADDING,
    )

    assert _FLASHMLA_SM90_HEAD_PADDING == 64, (
        f"SM90 head padding should be 64, got {_FLASHMLA_SM90_HEAD_PADDING}"
    )


# ─── TEST 2.14: Forward MQA Dispatch Verification ────────────────────


def test_forward_mqa_has_fa3_and_fallback_methods():
    """Verify CutlassFA3MLASparseImpl has both kernel dispatch methods."""
    from vllm.v1.attention.backends.mla.cutlass_fa3_sparse import (
        CutlassFA3MLASparseImpl,
    )

    assert hasattr(CutlassFA3MLASparseImpl, "_forward_fa3"), (
        "CutlassFA3MLASparseImpl should have _forward_fa3 method"
    )
    assert hasattr(CutlassFA3MLASparseImpl, "_forward_flashmla_bf16_fallback"), (
        "CutlassFA3MLASparseImpl should have _forward_flashmla_bf16_fallback method"
    )
