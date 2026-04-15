# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HPC attention backend."""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.hpc_attn import (
    HpcAttentionBackend,
    HpcAttentionImpl,
    HpcAttentionMetadata,
    HpcAttentionMetadataBuilder,
    _hpc_decode_use_splitk,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def kv_cache_spec():
    spec = MagicMock()
    spec.block_size = 32
    spec.num_kv_heads = 8
    spec.head_size = 128
    spec.dtype = torch.bfloat16
    return spec


@pytest.fixture
def vllm_config():
    cfg = MagicMock()
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.speculative_config = None
    return cfg


def make_builder(vllm_config, kv_cache_spec, device):
    return HpcAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["attn"],
        vllm_config=vllm_config,
        device=device,
    )


def make_impl(block_size=32, num_heads=8, head_size=128, num_kv_heads=8):
    return HpcAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        block_size=block_size,
    )


def make_common_meta(
    num_reqs: int,
    seq_lens,  # list[int] or int (broadcast)
    query_lens,  # list[int] or int (broadcast)
    device: torch.device,
    max_blocks: int = 32,
) -> CommonAttentionMetadata:
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens] * num_reqs
    if isinstance(query_lens, int):
        query_lens = [query_lens] * num_reqs

    q_start = [0] + list(torch.tensor(query_lens).cumsum(0).tolist())
    q_locs_cpu = torch.tensor(q_start, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=q_locs_cpu.to(device),
        query_start_loc_cpu=q_locs_cpu,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros(
            (num_reqs, max_blocks), dtype=torch.int32, device=device
        ),
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.int64, device=device),
    )


# ---------------------------------------------------------------------------
# Backend static properties
# ---------------------------------------------------------------------------


def test_backend_properties():
    assert HpcAttentionBackend.get_name() == "HPC_ATTN"
    assert HpcAttentionBackend.accept_output_buffer is True
    assert HpcAttentionBackend.forward_includes_kv_cache_update is False
    assert HpcAttentionBackend.get_impl_cls() is HpcAttentionImpl
    assert HpcAttentionBackend.get_builder_cls() is HpcAttentionMetadataBuilder
    assert torch.bfloat16 in HpcAttentionBackend.supported_dtypes
    assert torch.float16 not in HpcAttentionBackend.supported_dtypes
    assert set(HpcAttentionBackend.get_supported_kernel_block_sizes()) == {32, 64}


def test_kv_cache_shape():
    assert HpcAttentionBackend.get_kv_cache_shape(10, 32, 8, 128) == (2, 10, 32, 8, 128)
    with pytest.raises(ValueError, match="block_size"):
        HpcAttentionBackend.get_kv_cache_shape(10, 16, 8, 128)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_builder_metadata_fields(device, kv_cache_spec, vllm_config):
    """build() returns correct metadata dtypes and shapes for decode/prefill/mixed."""
    cases = [
        # (seq_lens, query_lens, expected_tokens, expected_max_q)
        ([32, 64], [1, 1], 2, 1),  # pure decode
        ([16, 32], [16, 32], 48, 32),  # pure prefill
        ([32, 64, 128], [1, 1, 16], 18, 16),  # mixed
    ]
    builder = make_builder(vllm_config, kv_cache_spec, device)
    for seq_lens, query_lens, exp_tokens, exp_max_q in cases:
        common = make_common_meta(len(seq_lens), seq_lens, query_lens, device)
        meta = builder.build(common_prefix_len=0, common_attn_metadata=common)
        assert isinstance(meta, HpcAttentionMetadata)
        assert meta.num_actual_tokens == exp_tokens
        assert meta.max_query_len == exp_max_q
        assert meta.seq_lens.dtype == torch.int32
        assert meta.block_table.dtype == torch.int32


def test_builder_rejects_cascade(device, kv_cache_spec, vllm_config):
    builder = make_builder(vllm_config, kv_cache_spec, device)
    common = make_common_meta(2, [64, 64], [1, 1], device)
    with pytest.raises(NotImplementedError, match="cascade"):
        builder.build(common_prefix_len=32, common_attn_metadata=common)


def test_builder_reorder_threshold(device, kv_cache_spec, vllm_config):
    builder = make_builder(vllm_config, kv_cache_spec, device)
    assert builder.reorder_batch_threshold == 1


def test_builder_split_fields(device, kv_cache_spec, vllm_config):
    """num_decodes / num_decode_tokens are correctly set for pure-decode and mixed."""
    builder = make_builder(vllm_config, kv_cache_spec, device)

    # Pure decode
    common = make_common_meta(4, 512, 1, device)
    meta = builder.build(0, common)
    assert meta.num_decodes == 4
    assert meta.num_decode_tokens == 4

    # Mixed: 4 decode (q=1) + 2 prefill (q=256)
    num_decodes, num_prefills, pref_q = 4, 2, 256
    seq_lens = [512] * num_decodes + [pref_q] * num_prefills
    query_lens = [1] * num_decodes + [pref_q] * num_prefills
    common = make_common_meta(num_decodes + num_prefills, seq_lens, query_lens, device)
    meta = builder.build(0, common)
    assert meta.num_decodes == num_decodes
    assert meta.num_decode_tokens == num_decodes


# ---------------------------------------------------------------------------
# CUDA graph capture
# ---------------------------------------------------------------------------


def test_cudagraph_support_level():
    assert (
        HpcAttentionMetadataBuilder._cudagraph_support
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


def test_cudagraph_capture_overwrites_seq_lens(device, kv_cache_spec, vllm_config):
    """build_for_cudagraph_capture fills seq_lens=1 and block_table=0."""
    builder = make_builder(vllm_config, kv_cache_spec, device)
    common = make_common_meta(4, 1024, 1, device)
    meta = builder.build_for_cudagraph_capture(common)
    assert torch.all(meta.seq_lens == 1)
    assert torch.all(meta.block_table == 0)
    assert meta.num_actual_tokens == 4


def test_cudagraph_normal_build_unchanged(device, kv_cache_spec, vllm_config):
    """Regular build() must NOT overwrite seq_lens."""
    builder = make_builder(vllm_config, kv_cache_spec, device)
    common = make_common_meta(2, 64, 1, device)
    meta = builder.build(0, common)
    assert torch.all(meta.seq_lens == 64)


# ---------------------------------------------------------------------------
# Impl: dtype guard & KV cache update
# ---------------------------------------------------------------------------


def test_impl_rejects_wrong_dtype(device):
    with pytest.raises((AssertionError, ValueError)):
        HpcAttentionImpl(
            num_heads=8,
            head_size=128,
            scale=0.1,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            block_size=32,
        )


def test_do_kv_cache_update(device):
    impl = make_impl()
    bs, num_kv_heads, head_size, block_size = 4, 8, 128, 32
    kv_cache = torch.randn(
        2, 100, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device=device
    )
    key = torch.randn(bs, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)
    value = torch.randn(
        bs, num_kv_heads, head_size, dtype=torch.bfloat16, device=device
    )
    slot_mapping = torch.zeros(bs, dtype=torch.int64, device=device)
    layer = MagicMock()
    layer._k_scale = torch.tensor(1.0)
    layer._v_scale = torch.tensor(1.0)

    mock_fn = MagicMock()
    with patch(
        "vllm.v1.attention.backends.hpc_attn._get_reshape_and_cache_flash",
        return_value=mock_fn,
    ):
        impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
    mock_fn.assert_called_once()
    args = mock_fn.call_args[0]
    key_cache, value_cache = kv_cache.unbind(0)
    assert torch.equal(args[0], key)
    assert torch.equal(args[1], value)
    assert torch.equal(args[2], key_cache)
    assert torch.equal(args[3], value_cache)


# ---------------------------------------------------------------------------
# Impl: forward dispatch (mocked hpc module)
# ---------------------------------------------------------------------------


def _make_decode_meta(num_reqs, device):
    seq_lens = torch.randint(32, 512, (num_reqs,), dtype=torch.int32, device=device)
    max_decode_seq_len = int(seq_lens.max().item())
    return HpcAttentionMetadata(
        num_actual_tokens=num_reqs,
        max_query_len=1,
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32, device=device),
        max_seq_len=int(seq_lens.max()),
        seq_lens=seq_lens,
        block_table=torch.zeros(num_reqs, 8, dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(num_reqs, dtype=torch.int64, device=device),
        num_decodes=num_reqs,
        num_decode_tokens=num_reqs,
        max_decode_seq_len=max_decode_seq_len,
    )


def _make_prefill_meta(query_lens, device):
    num_reqs = len(query_lens)
    total_q = sum(query_lens)
    q_start = [0] + list(torch.tensor(query_lens).cumsum(0).tolist())
    return HpcAttentionMetadata(
        num_actual_tokens=total_q,
        max_query_len=max(query_lens),
        query_start_loc=torch.tensor(q_start, dtype=torch.int32, device=device),
        max_seq_len=max(query_lens),
        seq_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
        block_table=torch.zeros(num_reqs, 2, dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(total_q, dtype=torch.int64, device=device),
    )


def test_forward_profiling_run(device):
    """forward() returns zeros when attn_metadata is None (profiling pass)."""
    impl = make_impl()
    output = torch.ones(4, 8, 128, dtype=torch.bfloat16, device=device)
    result = impl.forward(
        MagicMock(),
        None,
        None,
        None,
        torch.zeros(1, dtype=torch.bfloat16, device=device),
        None,
        output,
    )
    assert torch.all(result == 0)


def test_forward_decode_dispatch(device):
    """forward() calls attention_decode_bf16 for a pure decode batch."""
    impl = make_impl()
    num_reqs, num_heads, head_size = 4, 8, 128
    meta = _make_decode_meta(num_reqs, device)
    query = torch.randn(
        num_reqs, num_heads, head_size, dtype=torch.bfloat16, device=device
    )
    output = torch.zeros_like(query)
    kv_cache = torch.zeros(
        2, 100, 32, 8, head_size, dtype=torch.bfloat16, device=device
    )
    layer = MagicMock()
    layer._k_scale = torch.tensor(1.0)
    layer._v_scale = torch.tensor(1.0)

    fake_hpc = MagicMock()
    fake_hpc.attention_decode_bf16.return_value = torch.zeros_like(query)
    with patch.dict(sys.modules, {"hpc": fake_hpc}):
        impl.forward(layer, query, None, None, kv_cache, meta, output)

    fake_hpc.attention_decode_bf16.assert_called_once()
    fake_hpc.attention_with_kvcache_prefill_bf16.assert_not_called()
    kw = fake_hpc.attention_decode_bf16.call_args[1]
    assert torch.equal(kw["num_seq_kvcache"], meta.seq_lens)
    assert kw["new_kv_included"] is True
    assert kw["splitk"] is False  # even batch, MHA, max decode KV < 2048


@pytest.mark.parametrize(
    ("max_sl", "n_dec_tok", "nq", "nkv", "expect"),
    [
        # default branch (not 32h/4kv): split-K off only for short KV
        (100, 4, 8, 8, False),
        (1024, 4, 8, 8, True),
        # 32h / 4kv table
        (100, 3, 32, 4, True),  # t < 6
        (500, 8, 32, 4, False),  # 6 <= t < 12 and sl < 1024
        (2000, 8, 32, 4, True),  # 6 <= t < 12 but sl >= 1024
        (2000, 12, 32, 4, False),  # 12 <= t < 14 and sl < 3072
        (4000, 12, 32, 4, True),  # 12 <= t < 14 but sl >= 3072
        (4095, 15, 32, 4, False),  # 14 <= t < 16 and sl < 4k
        (5000, 15, 32, 4, True),  # 14 <= t < 16 but sl >= 4k
        (8191, 20, 32, 4, False),  # 16 <= t < 24 and sl < 8k
        (9000, 20, 32, 4, True),
        (24575, 28, 32, 4, False),  # 24 <= t < 32 and sl < 24k
        (30000, 28, 32, 4, True),
        (20000, 35, 32, 4, False),  # t >= 32 and sl < 24k
        (30000, 35, 32, 4, True),  # t >= 32 and sl >= 24k
    ],
)
def test_hpc_decode_use_splitk_heuristic(max_sl, n_dec_tok, nq, nkv, expect):
    assert _hpc_decode_use_splitk(max_sl, n_dec_tok, nq, nkv) is expect


def test_forward_prefill_dispatch(device):
    """forward() calls attention_with_kvcache_prefill_bf16 for a pure prefill batch."""
    impl = make_impl()
    query_lens = [16, 32]
    num_heads, head_size = 8, 128
    total_q = sum(query_lens)
    meta = _make_prefill_meta(query_lens, device)
    query = torch.randn(
        total_q, num_heads, head_size, dtype=torch.bfloat16, device=device
    )
    output = torch.zeros_like(query)
    kv_cache = torch.zeros(
        2, 100, 32, 8, head_size, dtype=torch.bfloat16, device=device
    )
    layer = MagicMock()
    layer._k_scale = torch.tensor(1.0)
    layer._v_scale = torch.tensor(1.0)

    fake_hpc = MagicMock()
    fake_hpc.attention_with_kvcache_prefill_bf16.return_value = torch.zeros_like(query)
    with patch.dict(sys.modules, {"hpc": fake_hpc}):
        impl.forward(layer, query, None, None, kv_cache, meta, output)

    fake_hpc.attention_with_kvcache_prefill_bf16.assert_called_once()
    fake_hpc.attention_decode_bf16.assert_not_called()
    kw = fake_hpc.attention_with_kvcache_prefill_bf16.call_args[1]
    assert torch.equal(kw["seqlens_kvcache"], meta.seq_lens)
    assert kw["max_seqlens_q"] == 32


def test_forward_mixed_batch_output_slices(device):
    """Mixed batch: decode/prefill kernels receive the right query slices and
    their outputs are written to the correct positions in the output buffer."""
    num_heads, num_kv_heads, head_size = 4, 2, 64
    num_decodes, pref_q, num_prefills = 4, 128, 2
    num_prefill_tokens = num_prefills * pref_q
    num_actual_tokens = num_decodes + num_prefill_tokens
    block_size = 32

    q_start = [0, 1, 2, 3, 4, 4 + pref_q, 4 + 2 * pref_q]
    meta = HpcAttentionMetadata(
        num_actual_tokens=num_actual_tokens,
        max_query_len=pref_q,
        query_start_loc=torch.tensor(q_start, dtype=torch.int32, device=device),
        max_seq_len=512,
        seq_lens=torch.tensor(
            [512] * num_decodes + [pref_q] * num_prefills,
            dtype=torch.int32,
            device=device,
        ),
        block_table=torch.zeros(
            num_decodes + num_prefills, 16, dtype=torch.int32, device=device
        ),
        slot_mapping=torch.zeros(num_actual_tokens, dtype=torch.int64, device=device),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        max_decode_seq_len=512,
    )

    query = torch.randn(
        num_actual_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device
    )
    output = torch.zeros_like(query)
    kv_cache = torch.zeros(
        2, 64, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device=device
    )
    layer = MagicMock()
    layer._k_scale = torch.tensor(1.0, device=device)
    layer._v_scale = torch.tensor(1.0, device=device)

    decode_out = torch.ones(
        num_decodes, num_heads, head_size, dtype=torch.bfloat16, device=device
    )
    prefill_out = torch.full(
        (num_prefill_tokens, num_heads, head_size),
        2.0,
        dtype=torch.bfloat16,
        device=device,
    )
    mock_hpc = MagicMock()
    mock_hpc.attention_decode_bf16.return_value = decode_out
    mock_hpc.attention_with_kvcache_prefill_bf16.return_value = prefill_out

    impl = make_impl(
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
    )
    with patch.dict(sys.modules, {"hpc": mock_hpc}):
        result = impl.forward(
            layer=layer,
            query=query,
            key=None,
            value=None,
            kv_cache=kv_cache,
            attn_metadata=meta,
            output=output,
        )

    # Verify query slices passed to each kernel
    dec_q = mock_hpc.attention_decode_bf16.call_args[1].get(
        "q", mock_hpc.attention_decode_bf16.call_args[0][0]
    )
    assert dec_q.shape == (num_decodes, num_heads, head_size)

    pref_q_arg = mock_hpc.attention_with_kvcache_prefill_bf16.call_args[1].get(
        "q", mock_hpc.attention_with_kvcache_prefill_bf16.call_args[0][0]
    )
    assert pref_q_arg.shape == (num_prefill_tokens, num_heads, head_size)

    # Verify output slices
    assert torch.all(result[:num_decodes] == 1.0)
    assert torch.all(result[num_decodes:num_actual_tokens] == 2.0)


# ---------------------------------------------------------------------------
# Numerical accuracy: HPC_ATTN vs PyTorch SDPA
# ---------------------------------------------------------------------------


def _sdpa_reference(
    query, key_cache, value_cache, block_table, seq_lens, query_start_loc, scale
):
    import torch.nn.functional as F

    batch = seq_lens.shape[0]
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    heads_per_group = num_heads // num_kv_heads
    outputs = []
    for i in range(batch):
        q_s, q_e = int(query_start_loc[i]), int(query_start_loc[i + 1])
        q_i = query[q_s:q_e].float()
        kv_len = int(seq_lens[i])
        n_blk = (kv_len + block_size - 1) // block_size
        k_flat = (
            key_cache[block_table[i, :n_blk]]
            .reshape(-1, num_kv_heads, head_dim)
            .float()
        )
        v_flat = (
            value_cache[block_table[i, :n_blk]]
            .reshape(-1, num_kv_heads, head_dim)
            .float()
        )
        k_i = k_flat[:kv_len].repeat_interleave(heads_per_group, dim=1)
        v_i = v_flat[:kv_len].repeat_interleave(heads_per_group, dim=1)
        out = (
            F.scaled_dot_product_attention(
                q_i.transpose(0, 1).unsqueeze(0),
                k_i.transpose(0, 1).unsqueeze(0),
                v_i.transpose(0, 1).unsqueeze(0),
                scale=scale,
                is_causal=(q_i.shape[0] > 1),
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        outputs.append(out)
    return torch.cat(outputs, dim=0).to(torch.bfloat16)


def _build_kvcache_and_meta(
    seq_lens, query_lens, num_kv_heads, head_dim, block_size, device
):
    batch = len(seq_lens)
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    total_blocks = batch * max_blocks
    key_cache = torch.randn(
        total_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn(
        total_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(
        batch, max_blocks
    )
    total_q = sum(query_lens)
    q_start = torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    # Decode requests (query_len == 1) are placed at the front of the batch.
    # num_decodes and num_decode_tokens must be set correctly so that
    # forward() dispatches to the right kernel (decode vs. prefill).
    num_decodes = sum(1 for q in query_lens if q == 1)
    num_decode_tokens = sum(q for q in query_lens if q == 1)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    max_decode_seq_len = (
        int(seq_lens_t[:num_decodes].max().item()) if num_decodes > 0 else 0
    )
    meta = HpcAttentionMetadata(
        num_actual_tokens=total_q,
        max_query_len=max(query_lens),
        query_start_loc=q_start,
        max_seq_len=max(seq_lens),
        seq_lens=seq_lens_t,
        block_table=block_table,
        slot_mapping=torch.zeros(total_q, dtype=torch.int64, device=device),
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        max_decode_seq_len=max_decode_seq_len,
    )
    return (
        key_cache,
        value_cache,
        torch.stack([key_cache, value_cache]),
        block_table,
        meta,
    )


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (4, 1)])
@pytest.mark.parametrize(
    "seq_lens,query_lens",
    [
        ([128, 256, 512], [1, 1, 1]),  # pure decode
        ([64, 128], [1, 1]),
        ([64, 64], [64, 64]),  # pure prefill
        ([128, 64], [128, 64]),
        ([128, 64], [1, 64]),  # mixed
    ],
)
def test_accuracy_vs_sdpa(device, num_q_heads, num_kv_heads, seq_lens, query_lens):
    """HPC_ATTN output must match PyTorch SDPA within bfloat16 tolerance."""
    block_size, head_dim = 32, 128
    scale = head_dim**-0.5
    total_q = sum(query_lens)

    key_cache, value_cache, full_kv_cache, block_table, meta = _build_kvcache_and_meta(
        seq_lens, query_lens, num_kv_heads, head_dim, block_size, device
    )
    query = torch.randn(
        total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    impl = make_impl(
        block_size=block_size,
        num_heads=num_q_heads,
        head_size=head_dim,
        num_kv_heads=num_kv_heads,
    )
    layer = MagicMock()
    layer._k_scale = torch.tensor(1.0)
    layer._v_scale = torch.tensor(1.0)

    hpc_out = torch.zeros(
        total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    impl.forward(layer, query, None, None, full_kv_cache, meta, hpc_out)

    ref_out = _sdpa_reference(
        query,
        key_cache,
        value_cache,
        block_table,
        meta.seq_lens,
        meta.query_start_loc,
        scale,
    )

    max_err = (hpc_out.float() - ref_out.float()).abs().max().item()
    assert torch.allclose(hpc_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2), (
        f"max abs error {max_err:.4f} exceeds tolerance. "
        f"seq_lens={seq_lens} query_lens={query_lens} "
        f"num_q_heads={num_q_heads} num_kv_heads={num_kv_heads}"
    )
