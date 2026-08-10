# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    AiterMLAMetadataBuilder,
)


class _NoOpTritonKernel:
    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        pass


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
        seq_lens_for_kernel,
        *,
        KERNEL_BLOCK_SIZE,
        BLOCK_SIZE,
    ):
        self.kernel_block_size = KERNEL_BLOCK_SIZE
        for req_idx in range(self.grid[0]):
            out_start = int(paged_kv_indptr[req_idx].item())
            seq_len = int(seq_lens_for_kernel[req_idx].item())
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
):
    return SimpleNamespace(
        device=torch.device("cpu"),
        num_heads=num_heads,
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
        _num_attention_heads=16,
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


@pytest.mark.parametrize("num_heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize(
    "spec_method, parallel_drafting",
    [
        ("deepseek_mtp", False),
        # A drafter that is not one of the historically recognized MTP methods,
        # and a parallel one, so the threshold is 1 + 2 * num_spec rather than
        # 1 + num_spec. Sizing the metadata off a method name instead leaves
        # these at qlen=1 while the router still admits the full range.
        ("dspark", True),
        ("eagle", False),
    ],
)
def test_mtp_builder_init_sizes_native_fp8_metadata(
    monkeypatch, num_heads, spec_method, parallel_drafting
):
    """Aiter init sizes the metadata for every query length decode can be handed.

    Sweeping num_heads asserts the max(16, num_heads) clamp is what sizes the
    metadata, covering the fp8 nhead=32 (TP4) fold path.
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
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        model_config=SimpleNamespace(max_model_len=16, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
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
            "num_attention_heads": max(16, num_heads),
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
    in-range -- dropping it is the regression this guards.

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
