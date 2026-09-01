# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER MLA tests", allow_module_level=True)

from vllm.v1.attention.backends.mla import rocm_aiter_mla  # noqa: E402
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (  # noqa: E402
    AiterMLAHelper,
    AiterMLAImpl,
    AiterMLAMetadataBuilder,
)
from vllm.v1.attention.ops.rocm_aiter_mla_merge import (  # noqa: E402
    merge_mla_segments_triton,
)


class _NoOpTritonKernel:
    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        pass


def _lse_combine_natural(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalizer = torch.logaddexp(lse_a, lse_b)
    out = out_a * torch.exp(lse_a - normalizer).unsqueeze(-1)
    out += out_b * torch.exp(lse_b - normalizer).unsqueeze(-1)
    return out.to(out_dtype), normalizer


class _ExpandPageIndicesKernel:
    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(
        self,
        page_indices,
        block_table_tensor,
        stride,
        paged_kv_indptr,
        *,
        KERNEL_BLOCK_SIZE,
        BLOCK_SIZE,
    ):
        self.kernel_block_size = KERNEL_BLOCK_SIZE
        for req_idx in range(self.grid[0]):
            out_start = int(paged_kv_indptr[req_idx].item())
            seq_len = int(
                (paged_kv_indptr[req_idx + 1] - paged_kv_indptr[req_idx]).item()
            )
            for token_idx in range(seq_len):
                block_id = int(
                    block_table_tensor[req_idx, token_idx // KERNEL_BLOCK_SIZE].item()
                )
                page_indices[out_start + token_idx] = (
                    block_id * KERNEL_BLOCK_SIZE + token_idx % KERNEL_BLOCK_SIZE
                )


def _builder(
    *,
    mtp_decode_qlen: int,
    has_full_cudagraphs: bool = False,
    kernel_block_size: int = 1,
    max_decode_rows: int = 32,
    num_heads: int = 16,
    kv_cache_dtype: str = "auto",
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
):
    stub = SimpleNamespace(
        device=torch.device("cpu"),
        num_heads=num_heads,
        # DCP gathers every rank's head shard before decode, and the routing
        # predicates read the gathered count.
        _decode_num_heads=num_heads * dcp_world_size,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=1,
        # Mirrors the production constructor: configuration decides whether the
        # segmented route is available, query length decides per batch.
        _supports_segmented_dcp_verify=rocm_aiter_mla._segmented_dcp_verify_supported(
            dcp_world_size, 1
        ),
        # Derived once in the real constructor, so derive it once here too.
        _segmented_page_size=rocm_aiter_mla._segmented_mla_page_size(kernel_block_size),
        _dcp_verify_buffers=None,
        _graph_seq_lens=None,
        _kv_cache_dtype_str=kv_cache_dtype,
        paged_kv_last_page_len=torch.ones(max_decode_rows, dtype=torch.int32),
        paged_kv_indices=torch.empty(1024, dtype=torch.int32),
        paged_kv_indptr=torch.empty(max_decode_rows + 1, dtype=torch.int32),
        qo_indptr=torch.empty(max_decode_rows + 1, dtype=torch.int32),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                has_full_cudagraphs=lambda: has_full_cudagraphs
            )
        ),
        _mtp_decode_qlen=mtp_decode_qlen,
        _uniform_padded_mtp_qo_len=(AiterMLAMetadataBuilder._uniform_padded_mtp_qo_len),
        _use_persistent_metadata=False,
        kernel_block_size=kernel_block_size,
        _num_attention_heads=AiterMLAHelper.get_actual_mla_num_heads(num_heads),
        _mla_work_meta_data=torch.empty(1, dtype=torch.int32),
        _mla_work_info_set=torch.empty(1, dtype=torch.int32),
        _mla_work_indptr=torch.empty(1, dtype=torch.int32),
        _mla_reduce_indptr=torch.empty(1, dtype=torch.int32),
        _mla_reduce_final_map=torch.empty(1, dtype=torch.int32),
        _mla_reduce_partial_map=torch.empty(1, dtype=torch.int32),
        _mla_q_dtype=torch.bfloat16,
        _mla_kv_dtype=torch.bfloat16,
        decode_attn_out_dtype=torch.bfloat16,
    )
    # Bound to the stub rather than faked: the verify flatten's per-row view is
    # part of what _build_decode is being tested for.
    stub._build_dcp_verify_row_view = (
        AiterMLAMetadataBuilder._build_dcp_verify_row_view.__get__(stub)
    )
    stub._fill_dcp_verify_page_table = (
        AiterMLAMetadataBuilder._fill_dcp_verify_page_table.__get__(stub)
    )
    return stub


def test_backend_declares_uniform_batch_support():
    # UNIFORM/UNIFORM_BATCH is unconditional: MTP yields uniform qlen>1 and
    # non-MTP yields qlen==1, both uniform batches.
    assert (
        AiterMLAMetadataBuilder.query_len_support
        == rocm_aiter_mla.QueryLenSupport.UNIFORM
    )
    assert (
        AiterMLAMetadataBuilder._cudagraph_support
        == rocm_aiter_mla.AttentionCGSupport.UNIFORM_BATCH
    )


def test_dcp_verify_row_view_is_causal_per_row():
    """Row t must cover this rank's share of positions [0, seq_len - qlen + t].

    Regression guard: giving every row of a request the committed prefix only
    drops the whole verify block, including each row's own token, since nothing
    else adds it back on the target path.
    """
    qlen = 4
    builder = _builder(mtp_decode_qlen=qlen, dcp_world_size=2, kernel_block_size=2)
    block_table = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32)

    view = builder._build_dcp_verify_row_view(
        qlen,
        block_table,
        torch.tensor([10, 12], dtype=torch.int32),
    )

    # seq_len 10 -> rows see 7, 8, 9, 10 global positions, of which rank 0 of 2
    # holds ceil(n / 2); seq_len 12 -> 9, 10, 11, 12.
    assert view.row_lens.tolist() == [4, 4, 5, 5, 5, 5, 6, 6]
    assert view.qo_indptr.tolist() == list(range(2 * qlen + 1))
    # One query per row, so the page table is the request's ascending shard and
    # only the row length selects the causal prefix.
    assert view.block_table.tolist() == [[7, 8, 9]] * qlen + [[10, 11, 12]] * qlen
    assert view.max_kv_seq_len == 6


def test_dcp_verify_rows_sum_to_the_global_window_across_ranks():
    """Every rank's rows together must cover each row's window exactly once."""
    qlen = 4
    dcp_world_size = 2
    seq_lens = torch.tensor([10, 12], dtype=torch.int32)
    per_rank = []
    for dcp_rank in range(dcp_world_size):
        builder = _builder(
            mtp_decode_qlen=qlen,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            kernel_block_size=2,
        )
        view = builder._build_dcp_verify_row_view(
            qlen,
            torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32),
            seq_lens,
        )
        per_rank.append(view.row_lens.tolist())

    expected = [
        int(seq_len) - qlen + t + 1
        for seq_len in seq_lens.tolist()
        for t in range(qlen)
    ]
    assert [sum(lens) for lens in zip(*per_rank)] == expected


def test_dcp_verify_row_view_uses_static_graph_bound():
    """Under full graphs the bound comes from the buffers, not the batch."""
    qlen = 3
    builder = _builder(
        mtp_decode_qlen=qlen,
        dcp_world_size=8,
        has_full_cudagraphs=True,
        kernel_block_size=1536,
    )
    builder._dcp_verify_buffers = rocm_aiter_mla.AiterMLADCPVerifyMetadata(
        row_lens=torch.zeros(qlen, dtype=torch.int32),
        block_table=torch.zeros((qlen, 24), dtype=torch.int32),
        qo_indptr=torch.zeros(qlen + 1, dtype=torch.int32),
        page_size=128,
        max_kv_seq_len=3072,
    )

    view = builder._build_dcp_verify_row_view(
        qlen,
        torch.arange(2, dtype=torch.int32).view(1, 2),
        torch.tensor([1000], dtype=torch.int32),
    )

    buffers = builder._dcp_verify_buffers
    assert view.row_lens.tolist() == [125] * qlen
    assert view.block_table.data_ptr() == buffers.block_table.data_ptr()
    assert view.qo_indptr.data_ptr() == buffers.qo_indptr.data_ptr()
    assert view.block_table[0].tolist() == list(range(24))
    assert view.max_kv_seq_len == 3072


def test_dcp_fp8_verify_build_uses_segmented(monkeypatch):
    qlen = 4
    monkeypatch.setattr(rocm_aiter_mla, "_segmented_mla_decode_supported", lambda: True)

    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(
            mtp_decode_qlen=qlen,
            dcp_world_size=2,
            kv_cache_dtype="fp8",
            kernel_block_size=2,
        ),
        block_table_tensor=torch.tensor([[0, 1, 2], [10, 11, 12]], dtype=torch.int32),
        seq_lens_device=torch.tensor([5, 6], dtype=torch.int32),
        max_seq_len=6,
        query_start_loc_cpu=torch.tensor([0, qlen, 2 * qlen], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, qlen, 2 * qlen], dtype=torch.int32),
        num_decode_tokens=2 * qlen,
        dcp_tot_seq_lens_device=torch.tensor([10, 12], dtype=torch.int32),
    )

    assert metadata.dcp_verify is not None


def test_single_token_dcp_decode_returns_unpadded_lse(monkeypatch):
    """Single-token DCP decode must come back with an LSE the merge can use.

    This is the path every DCP step takes outside verification. The vLLM custom
    op drops aiter's LSE, so the impl calls aiter directly; without an LSE the
    MLA layer's cross-rank combine asserts. The head count is deliberately a
    non-divisor of 16 so both the output and the LSE have to be unpadded back
    to the gathered head count.
    """
    num_heads, dcp_world_size = 6, 2
    decode_heads = num_heads * dcp_world_size  # 12, padded to 16 by aiter
    num_tokens, head_dim = 3, 576
    captured = {}

    def fake_aiter_decode(q, kv_buffer, out, *args, **kwargs):
        captured["q_heads"] = q.shape[1]
        captured["return_lse"] = kwargs.get("return_lse")
        return None, torch.zeros(num_tokens, q.shape[1])

    monkeypatch.setattr(
        rocm_aiter_mla, "_get_aiter_mla_decode", lambda: fake_aiter_decode
    )

    impl = object.__new__(AiterMLAImpl)
    impl.num_heads = num_heads
    impl.dcp_world_size = dcp_world_size
    impl.kv_cache_dtype = "auto"
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.scale = head_dim**-0.5
    decode = SimpleNamespace(
        max_qo_len=1,
        qo_indptr=torch.arange(num_tokens + 1, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(num_tokens + 1, dtype=torch.int32),
        paged_kv_indices=torch.zeros(1, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(num_tokens, dtype=torch.int32),
        use_gluon_decode=False,
        use_gluon_verify=False,
        dcp_verify=None,
        has_persistent_metadata=False,
        attn_out_dtype=torch.bfloat16,
    )
    attn_metadata = SimpleNamespace(decode=decode, causal=True, work_meta_data=None)
    layer = SimpleNamespace(_q_scale=torch.tensor(1.0), _k_scale=torch.tensor(1.0))

    q = torch.zeros(num_tokens, decode_heads, head_dim, dtype=torch.bfloat16)
    output, lse = impl.forward_mqa(q, torch.zeros(1, 1, head_dim), attn_metadata, layer)

    assert captured["return_lse"] is True
    # aiter is handed the padded head count, but both results come back at the
    # gathered count the cross-rank merge expects.
    assert captured["q_heads"] == 16
    assert lse is not None
    assert output.shape[1] == decode_heads
    assert lse.shape == (num_tokens, decode_heads)


def test_verify_partial_attention_merge():
    device = torch.device("cuda")
    out_a = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], device=device)
    out_b = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]], device=device)
    lse_a = torch.tensor([[0.0], [1.0]], device=device)
    lse_b = torch.tensor([[1.0], [0.0]], device=device)

    out, lse = _lse_combine_natural(out_a, lse_a, out_b, lse_b, torch.float32)
    normalizer = torch.logaddexp(lse_a, lse_b)
    expected = out_a * torch.exp(lse_a - normalizer).unsqueeze(-1) + out_b * torch.exp(
        lse_b - normalizer
    ).unsqueeze(-1)

    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(lse, normalizer)


def test_segmented_verify_reduce_returns_natural_lse_and_masks_empty_rows():
    segment_output = torch.tensor(
        [1.0, 3.0, 99.0, 99.0],
        device="cuda",
    ).reshape(2, 1, 2, 1)
    segment_max = torch.tensor(
        [[[0.0, 1.0]], [[99.0, 99.0]]],
        device="cuda",
    )
    segment_expsum = torch.ones_like(segment_max)

    output, lse = merge_mla_segments_triton(
        segment_output,
        segment_max,
        segment_expsum,
        torch.tensor([2, 0], dtype=torch.int32, device="cuda"),
        tile_size=1,
        out_dtype=torch.float32,
    )

    torch.testing.assert_close(output[0], torch.tensor([[7.0 / 3]], device="cuda"))
    assert output[1].item() == 0
    torch.testing.assert_close(lse[0], torch.tensor([math.log(3.0)], device="cuda"))
    assert lse[1].item() == float("-inf")


def test_segmented_dcp_verify_matches_causal_attention(monkeypatch):
    """Exercise the target qlen>1 path and merge two rank-local results."""
    monkeypatch.setattr(rocm_aiter_mla, "_segmented_mla_decode_supported", lambda: True)

    torch.manual_seed(0)
    device = torch.device("cuda")
    dcp_world_size = 2
    qlen = 3
    num_heads = 64
    kv_lora_rank = 512
    rope_dim = 64
    head_dim = kv_lora_rank + rope_dim
    global_seq_len = 259
    block_size = 128
    kv_scale = 0.02
    sm_scale = head_dim**-0.5

    q_nope = torch.randn(
        qlen, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(qlen, num_heads, rope_dim, dtype=torch.bfloat16, device=device)
    kv_source = torch.randn(
        global_seq_len, head_dim, dtype=torch.float32, device=device
    )
    kv_fp8 = (kv_source / kv_scale).to(torch.float8_e4m3fn)
    kv_dequant = kv_fp8.float() * kv_scale

    partials = []
    for dcp_rank in range(dcp_world_size):
        local_kv = kv_fp8[dcp_rank::dcp_world_size]
        num_blocks = math.ceil(local_kv.shape[0] / block_size)
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        kv_cache.view(-1, head_dim)[: local_kv.shape[0]].copy_(local_kv)

        builder = _builder(
            mtp_decode_qlen=qlen,
            kernel_block_size=block_size,
            num_heads=num_heads // dcp_world_size,
            kv_cache_dtype="fp8",
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
        )
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)
        view = builder._build_dcp_verify_row_view(
            qlen,
            block_table,
            torch.tensor([global_seq_len], dtype=torch.int32, device=device),
        )
        decode = SimpleNamespace(
            max_qo_len=qlen,
            paged_kv_indptr=torch.tensor(
                [0, local_kv.shape[0]], dtype=torch.int32, device=device
            ),
            paged_kv_indices=torch.arange(
                local_kv.shape[0], dtype=torch.int32, device=device
            ),
            use_gluon_decode=False,
            use_gluon_verify=False,
            dcp_verify=view,
            attn_out_dtype=torch.bfloat16,
        )
        attn_metadata = SimpleNamespace(decode=decode, causal=True)
        impl = object.__new__(AiterMLAImpl)
        impl.num_heads = num_heads // dcp_world_size
        impl.dcp_world_size = dcp_world_size
        impl.kv_cache_dtype = "fp8"
        impl.kv_lora_rank = kv_lora_rank
        impl.qk_rope_head_dim = rope_dim
        impl.scale = sm_scale
        layer = SimpleNamespace(_k_scale=torch.tensor(kv_scale, device=device))

        partials.append(
            impl.forward_mqa((q_nope, q_pe), kv_cache, attn_metadata, layer)
        )

    output, lse = partials[0]
    assert lse is not None
    for rank_output, rank_lse in partials[1:]:
        assert rank_lse is not None
        output, lse = _lse_combine_natural(
            output.float(), lse, rank_output.float(), rank_lse, torch.float32
        )

    reference = torch.empty_like(output)
    q_nope_fp32 = q_nope.float()
    q_pe_fp32 = q_pe.float()
    for query_pos in range(qlen):
        visible = global_seq_len - qlen + query_pos + 1
        keys = kv_dequant[:visible]
        scores = torch.einsum(
            "hd,nd->hn", q_nope_fp32[query_pos], keys[:, :kv_lora_rank]
        )
        scores += torch.einsum(
            "hd,nd->hn", q_pe_fp32[query_pos], keys[:, kv_lora_rank:]
        )
        probs = torch.softmax(scores * sm_scale, dim=-1)
        reference[query_pos] = probs @ keys[:, :kv_lora_rank]

    torch.testing.assert_close(output, reference, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("num_heads", [8, 16, 24, 32, 64, 128])
@pytest.mark.parametrize(
    "spec_method, parallel_drafting",
    [
        ("deepseek_mtp", False),
        # A drafter that is not one of the historically recognized MTP methods,
        # and a parallel one, so the threshold is 1 + 2 * num_spec rather than
        # 1 + num_spec. Sizing the metadata off a method name instead leaves
        # these at qlen=1 while the router still admits the full range.
        ("custom", True),
        ("eagle", False),
    ],
)
def test_mtp_builder_init_sizes_native_fp8_metadata(
    monkeypatch, num_heads, spec_method, parallel_drafting
):
    """Aiter init sizes the metadata for every query length decode can be handed.

    Sweeping num_heads asserts metadata is sized for the padded decode shape,
    covering Kimi-K3 TP4's 24 -> 32 head path and native fp8 nhead=32 folding.
    """

    dtypes = SimpleNamespace(fp8="fp8", fp16="fp16", bf16="bf16")
    info_calls = []

    def get_mla_metadata_info_v1(
        max_batch_size,
        max_qo_len,
        num_attention_heads,
        q_dtype,
        kv_dtype,
        *,
        is_sparse,
        fast_mode,
    ):
        info_calls.append(
            {
                "max_batch_size": max_batch_size,
                "max_qo_len": max_qo_len,
                "num_attention_heads": num_attention_heads,
                "q_dtype": q_dtype,
                "kv_dtype": kv_dtype,
                "is_sparse": is_sparse,
                "fast_mode": fast_mode,
            }
        )
        return tuple((1, torch.int32) for _ in range(6))

    def init_common_builder(self, *args, **kwargs):
        self.num_heads = num_heads
        self.dcp_world_size = 1
        self.cp_kv_cache_interleave_size = 1
        # Mirror what _init_reorder_batch_threshold would have left behind: the
        # metadata is sized from the routing threshold, so a stub that skips it
        # would size for qlen=1 and hide the very mismatch this test covers.
        spec = config.speculative_config
        self.reorder_batch_threshold = (
            1 + (2 if spec.parallel_drafting else 1) * spec.num_speculative_tokens
        )

    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(
            dtypes=dtypes,
            get_mla_metadata_info_v1=get_mla_metadata_info_v1,
        ),
    )
    monkeypatch.setattr(
        rocm_aiter_mla.MLACommonMetadataBuilder,
        "__init__",
        init_common_builder,
    )
    monkeypatch.setattr(rocm_aiter_mla, "_fp8_mla_prefill_supported", lambda: False)

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=spec_method,
            num_speculative_tokens=3,
            parallel_drafting=parallel_drafting,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        model_config=SimpleNamespace(
            max_model_len=16,
            dtype=torch.bfloat16,
            get_num_attention_heads=lambda parallel_config: num_heads,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            # Empty: the per-layer head-count probe finds no attention layer and
            # falls back to the model config above.
            static_forward_context={},
        ),
    )
    builder = AiterMLAMetadataBuilder(
        kv_cache_spec=SimpleNamespace(block_size=1, dtype=torch.bfloat16),
        layer_names=["layer.0"],
        vllm_config=config,
        device=torch.device("cpu"),
    )

    assert info_calls == [
        {
            "max_batch_size": config.scheduler_config.max_num_seqs,
            "max_qo_len": builder.reorder_batch_threshold,
            "num_attention_heads": AiterMLAHelper.get_actual_mla_num_heads(num_heads),
            "q_dtype": dtypes.fp8,
            "kv_dtype": dtypes.fp8,
            "is_sparse": False,
            "fast_mode": True,
        }
    ]
    assert builder._mla_q_dtype == dtypes.fp8
    assert builder._mla_kv_dtype == dtypes.fp8


def test_mtp_decode_qlen4_keeps_uniform_rows_with_metadata(monkeypatch):
    get_mla_metadata_v1 = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=get_mla_metadata_v1),
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel()
    )

    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(mtp_decode_qlen=4),
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=torch.tensor([7, 5], dtype=torch.int32),
        max_seq_len=7,
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_decode_tokens=8,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.max_qo_len == 4
    assert torch.equal(metadata.seq_lens, torch.tensor([7, 5], dtype=torch.int32))
    assert torch.equal(metadata.qo_indptr, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert metadata.has_persistent_metadata
    assert get_mla_metadata_v1.call_args.kwargs["max_seqlen_qo"] == 4
    assert get_mla_metadata_v1.call_args.kwargs["uni_seqlen_qo"] == 4


def test_min_kv_seq_len_ignores_cudagraph_padding_rows(monkeypatch):
    """min_kv_seq_len must not be driven by padded dummy requests.

    Full-CG verify pins zero-qo rows to max_qo_len for paged_kv metadata; taking
    per_req_len.min() over all rows then reports qlen instead of a real KV length
    and collapses Gluon KV-split parallelism on single-request decode.
    """
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=mock.MagicMock()),
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel()
    )
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: True)
    monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: "auto")

    mtp_qlen = 8
    num_reqs = 8
    active_seq_len = 1032
    seq_lens = torch.tensor([active_seq_len] + [0] * (num_reqs - 1), dtype=torch.int32)
    query_start_loc = torch.tensor(
        [0, mtp_qlen] + [mtp_qlen] * (num_reqs - 1), dtype=torch.int32
    )

    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(
            mtp_decode_qlen=mtp_qlen,
            has_full_cudagraphs=True,
            max_decode_rows=num_reqs,
            num_heads=12,
        ),
        block_table_tensor=torch.zeros(num_reqs, active_seq_len, dtype=torch.int32),
        seq_lens_device=seq_lens,
        max_seq_len=active_seq_len,
        query_start_loc_cpu=query_start_loc,
        query_start_loc_device=query_start_loc,
        num_decode_tokens=num_reqs * mtp_qlen,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.min_kv_seq_len == active_seq_len


def test_full_cudagraph_padded_uniform_mtp_synthesizes_decode_indptr(
    monkeypatch,
):
    """Full-CG zero-qo rows follow rocm_aiter_mla.py:608-657,717-759."""

    get_mla_metadata_v1 = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=get_mla_metadata_v1),
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel()
    )

    mtp_qlen = 4
    seq_lens = torch.tensor([7, 0], dtype=torch.int32)
    qo_lens = torch.tensor([mtp_qlen, 0], dtype=torch.int32)
    expected_seq_lens = torch.where(qo_lens > 0, seq_lens, mtp_qlen)
    expected_paged_kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            expected_seq_lens.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    expected_qo_indptr = torch.arange(
        0,
        (seq_lens.numel() + 1) * mtp_qlen,
        step=mtp_qlen,
        dtype=torch.int32,
    )

    builder = _builder(
        mtp_decode_qlen=mtp_qlen,
        has_full_cudagraphs=True,
        max_decode_rows=4,
    )
    metadata = AiterMLAMetadataBuilder._build_decode(
        builder,
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=seq_lens,
        max_seq_len=int(seq_lens.max().item()),
        query_start_loc_cpu=torch.tensor([0, mtp_qlen, mtp_qlen], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, mtp_qlen, mtp_qlen], dtype=torch.int32),
        num_decode_tokens=seq_lens.numel() * mtp_qlen,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.max_qo_len == mtp_qlen
    assert torch.equal(metadata.seq_lens, expected_seq_lens)
    assert torch.equal(metadata.paged_kv_indptr, expected_paged_kv_indptr)
    assert torch.equal(metadata.qo_indptr, expected_qo_indptr)
    assert torch.all(
        builder.paged_kv_indptr[expected_paged_kv_indptr.numel() :]
        == expected_paged_kv_indptr[-1]
    )
    assert torch.all(
        builder.qo_indptr[expected_qo_indptr.numel() :] == expected_qo_indptr[-1]
    )
    assert metadata.has_persistent_metadata
    assert get_mla_metadata_v1.call_args.kwargs["max_seqlen_qo"] == mtp_qlen
    assert get_mla_metadata_v1.call_args.kwargs["uni_seqlen_qo"] == mtp_qlen


def test_decode_expands_kernel_block_page_indices(monkeypatch):
    """kernel_block_size>1 expands b -> b*K+offset at rocm_aiter_mla.py:696-704."""

    expand_kernel = _ExpandPageIndicesKernel()
    monkeypatch.setattr(rocm_aiter_mla, "_expand_page_indices_kernel", expand_kernel)
    # qlen==1 now takes the persistent-metadata path, which imports
    # get_mla_metadata_v1 from aiter; mock it so this test needs no real kernel.
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=mock.MagicMock()),
    )

    kernel_block_size = 2
    block_table = torch.tensor(
        [
            [10, 11, 99],
            [20, 21, 22],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    expected_paged_kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lens.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    expected_indices = torch.tensor(
        [
            int(block_table[req_idx, token_idx // kernel_block_size].item())
            * kernel_block_size
            + token_idx % kernel_block_size
            for req_idx, seq_len in enumerate(seq_lens.tolist())
            for token_idx in range(seq_len)
        ],
        dtype=torch.int32,
    )

    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(
            mtp_decode_qlen=1,
            kernel_block_size=kernel_block_size,
        ),
        block_table_tensor=block_table,
        seq_lens_device=seq_lens,
        max_seq_len=int(seq_lens.max().item()),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_decode_tokens=seq_lens.numel(),
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.max_qo_len == 1
    assert torch.equal(metadata.paged_kv_indptr, expected_paged_kv_indptr)
    assert torch.equal(
        metadata.paged_kv_indices[: expected_indices.numel()],
        expected_indices,
    )
    assert expand_kernel.grid == (seq_lens.numel(),)
    assert expand_kernel.kernel_block_size == kernel_block_size


@pytest.mark.parametrize(
    "mtp_decode_qlen, qo_len, num_heads, kv_cache_dtype, expect_persistent",
    [
        (1, 1, 16, "auto", True),  # non-MTP decode
        (4, 2, 16, "auto", True),  # MTP deployment, in-range step
        (4, 4, 16, "auto", True),  # MTP deployment, full-qlen verification step
        (1, 1, 24, "auto", True),  # unaligned H24 pads to H32 persistent decode
        (2, 4, 16, "auto", False),  # step demand exceeds provisioned K -> fallback
        (1, 1, 8, "auto", False),  # divisor head count -> Gluon decode owns qlen==1
        (4, 4, 8, "auto", False),  # small head count -> Gluon flatten owns qlen>1
        # A non-divisor head count is padded to 16 and runs the asm decode, so it
        # needs the schedule even though num_heads < 16. Its verify still
        # flattens onto Gluon though -- divisibility only steers the qlen==1
        # decode, and bf16 has no small-head multi-token asm kernel either way.
        (1, 1, 12, "auto", True),
        (8, 8, 12, "auto", False),
        # An fp8 cache never reaches Gluon, at any head count or qlen, so it
        # always needs the schedule. (1, 1, 8) is the case the head-count gate
        # got wrong: divisor head count, so it read as Gluon-owned.
        (1, 1, 8, "fp8", True),
        (1, 1, 12, "fp8", True),
        (1, 1, 16, "fp8", True),
        # DSpark verify shape. The fp8 fold path rejects non-persistent outright
        # once qlen > 4, so these must not fall through.
        (8, 8, 8, "fp8", True),
        (8, 8, 12, "fp8", True),
    ],
)
def test_persistent_metadata_gate(
    monkeypatch, mtp_decode_qlen, qo_len, num_heads, kv_cache_dtype, expect_persistent
):
    """Persistent metadata is passed iff the asm decode runs and 1 <= qlen <= K.

    K = _mtp_decode_qlen sizes the metadata buffers at init; a decode step gets
    the pre-built schedule only when its qlen fits those buffers, otherwise it
    falls back to the kernel computing its own. qlen==1 (non-MTP) must stay
    in-range -- dropping it is the regression this guards. This includes
    unaligned H24, whose padded H32 decode uses persistent metadata.

    Only the Gluon paths ignore the schedule, so the gate follows the routing
    predicates rather than the raw head count. Reading `num_heads >= 16` instead
    denies the schedule to two sets of shapes that do run the asm kernels: a
    non-divisor head count padded up to 16, and any fp8 cache below 16 heads.
    """
    get_mla_metadata_v1 = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=get_mla_metadata_v1),
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel()
    )
    # The gate follows the routing now, so the small-head bf16 rows above only
    # hold where a Gluon build exists. Pin the arch they describe: on gfx942
    # those shapes run the asm decode and do get the schedule, which is a
    # different assertion, made arch-free in the fp8 routing test.
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: True)
    monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: "auto")

    # Uniform, non-CUDA-graph batch: every request has exactly qo_len tokens, so
    # num_decode_tokens == sum(qo_len) and no dummy-row padding kicks in.
    num_reqs = 2
    query_start_loc = torch.arange(
        0, (num_reqs + 1) * qo_len, step=qo_len, dtype=torch.int32
    )
    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(
            mtp_decode_qlen=mtp_decode_qlen,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
        ),
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=torch.tensor([8, 8], dtype=torch.int32),
        max_seq_len=8,
        query_start_loc_cpu=query_start_loc,
        query_start_loc_device=query_start_loc,
        num_decode_tokens=num_reqs * qo_len,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.max_qo_len == qo_len
    assert metadata.has_persistent_metadata is expect_persistent
    assert get_mla_metadata_v1.called is expect_persistent
    if expect_persistent:
        assert get_mla_metadata_v1.call_args.kwargs["max_seqlen_qo"] == qo_len
        assert get_mla_metadata_v1.call_args.kwargs["uni_seqlen_qo"] == qo_len


@pytest.mark.parametrize(
    "qo_len, num_heads, kv_cache_dtype, expect_persistent",
    [
        # A padded rank keeps the schedule wherever a persistent kernel exists:
        # bf16 up to qseqlen 4, fp8 at every qlen (its fold requires it).
        (1, 12, "auto", True),
        (4, 12, "auto", True),
        (1, 12, "fp8", True),
        (8, 12, "fp8", True),
        # Past qseqlen 4 bf16 has only the non-persistent entry, and the fold
        # that would reach a persistent one is gfx950-only, so the padded rank
        # must fall back rather than ask for a kernel that is not built.
        (8, 12, "auto", False),
        # A native 16-head rank is unchanged: it took the schedule before this
        # backend keyed the gate off the routing, and still does.
        (8, 16, "auto", True),
    ],
)
def test_persistent_metadata_gate_without_gluon_build(
    monkeypatch, qo_len, num_heads, kv_cache_dtype, expect_persistent
):
    """The gate on an arch with no Gluon build, i.e. gfx942.

    Both Gluon predicates read False there, so a small-head shape reaches the
    padded asm decode instead of being flattened. The schedule then has to
    follow what the asm dispatch actually ships: bf16 has no gqa=16 persistent
    kernel above qseqlen 4, and the q-row fold that reaches one on gfx950 is
    guarded on that arch, so asking for the schedule there selects a kernel
    that does not exist.
    """
    get_mla_metadata_v1 = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(get_mla_metadata_v1=get_mla_metadata_v1),
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_expand_page_indices_kernel", _NoOpTritonKernel()
    )
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: False)
    monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: "auto")

    num_reqs = 2
    query_start_loc = torch.arange(
        0, (num_reqs + 1) * qo_len, step=qo_len, dtype=torch.int32
    )
    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(
            mtp_decode_qlen=qo_len,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
        ),
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=torch.tensor([8, 8], dtype=torch.int32),
        max_seq_len=8,
        query_start_loc_cpu=query_start_loc,
        query_start_loc_device=query_start_loc,
        num_decode_tokens=num_reqs * qo_len,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.has_persistent_metadata is expect_persistent
    assert get_mla_metadata_v1.called is expect_persistent
