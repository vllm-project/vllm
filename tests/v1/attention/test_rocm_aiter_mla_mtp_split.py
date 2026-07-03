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
                    block_table_tensor[
                        req_idx, token_idx // KERNEL_BLOCK_SIZE
                    ].item()
                )
                page_indices[out_start + token_idx] = (
                    block_id * KERNEL_BLOCK_SIZE
                    + token_idx % KERNEL_BLOCK_SIZE
                )


def _builder(
    *,
    split_mtp_decode: bool,
    mtp_decode_qlen: int,
    has_full_cudagraphs: bool = False,
    kernel_block_size: int = 1,
    max_decode_rows: int = 32,
):
    return SimpleNamespace(
        device=torch.device("cpu"),
        paged_kv_last_page_len=torch.ones(max_decode_rows, dtype=torch.int32),
        paged_kv_indices=torch.empty(1024, dtype=torch.int32),
        paged_kv_indptr=torch.empty(max_decode_rows + 1, dtype=torch.int32),
        qo_indptr=torch.empty(max_decode_rows + 1, dtype=torch.int32),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                has_full_cudagraphs=lambda: has_full_cudagraphs
            )
        ),
        _split_mtp_decode=split_mtp_decode,
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
        _mla_metadata_q_dtype=torch.bfloat16,
        _mla_metadata_kv_dtype=torch.bfloat16,
        decode_attn_out_dtype=torch.bfloat16,
        _max_split_per_batch=lambda num_reqs: 32,
    )


def _config(*, method: str | None, num_speculative_tokens: int | None, tp_size: int):
    speculative_config = None
    if method is not None:
        speculative_config = SimpleNamespace(
            method=method,
            num_speculative_tokens=num_speculative_tokens,
        )
    return SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize(
    ("num_speculative_tokens", "expected_query_len", "expected_split"),
    [
        (3, 4, False),
        (4, 5, True),
    ],
)
def test_mtp_config_selects_uniform_decode_policy_without_gpu_allocation(
    tp_size, num_speculative_tokens, expected_query_len, expected_split
):
    config = _config(
        method="deepseek_mtp",
        num_speculative_tokens=num_speculative_tokens,
        tp_size=tp_size,
    )

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == expected_query_len
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert AiterMLAMetadataBuilder._split_uniform_mtp_decode(config) is expected_split
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == rocm_aiter_mla.AttentionCGSupport.UNIFORM_BATCH
    )


def test_non_mtp_config_keeps_single_token_decode_policy():
    config = _config(method=None, num_speculative_tokens=None, tp_size=1)

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) is None
    assert not AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert not AiterMLAMetadataBuilder._split_uniform_mtp_decode(config)
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == rocm_aiter_mla.AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


def test_mtp_builder_init_sizes_native_fp8_metadata(monkeypatch):
    """Aiter init passes real MTP qlen/dtypes to rocm_aiter_mla.py:349-355."""

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
        self.num_heads = 8

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
            method="deepseek_mtp",
            num_speculative_tokens=3,
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
            "max_qo_len": config.speculative_config.num_speculative_tokens + 1,
            "num_attention_heads": 16,
            "q_dtype": dtypes.fp8,
            "kv_dtype": dtypes.fp8,
            "is_sparse": False,
            "fast_mode": True,
        }
    ]
    assert builder._mla_metadata_q_dtype == dtypes.fp8
    assert builder._mla_metadata_kv_dtype == dtypes.fp8


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
        _builder(split_mtp_decode=False, mtp_decode_qlen=4),
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
        split_mtp_decode=False,
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
    monkeypatch.setattr(
        rocm_aiter_mla, "_expand_page_indices_kernel", expand_kernel
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
            split_mtp_decode=False,
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


def test_mtp_decode_qlen5_splits_to_causal_single_token_rows(monkeypatch):
    expand_mtp_kernel = _NoOpTritonKernel()
    monkeypatch.setattr(
        rocm_aiter_mla, "_expand_mtp_decode_page_indices_kernel", expand_mtp_kernel
    )

    metadata = AiterMLAMetadataBuilder._build_decode(
        _builder(split_mtp_decode=True, mtp_decode_qlen=5),
        block_table_tensor=torch.arange(16, dtype=torch.int32).view(2, 8),
        seq_lens_device=torch.tensor([7, 5], dtype=torch.int32),
        max_seq_len=7,
        query_start_loc_cpu=torch.tensor([0, 5, 10], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, 5, 10], dtype=torch.int32),
        num_decode_tokens=10,
        dcp_tot_seq_lens_device=None,
    )

    assert metadata.max_qo_len == 1
    assert torch.equal(
        metadata.seq_lens,
        torch.tensor([3, 4, 5, 6, 7, 1, 2, 3, 4, 5], dtype=torch.int32),
    )
    assert torch.equal(metadata.qo_indptr, torch.arange(11, dtype=torch.int32))
    assert not metadata.has_persistent_metadata
    assert expand_mtp_kernel.grid == (10,)
