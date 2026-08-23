# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 CUTLASS sparse decode."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import AttentionConfig, CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparseTritonImpl,
    select_main_backend_and_impl_cls,
)
from vllm.models.minimax_m3.nvidia import (
    sparse_attention_msa as sparse_attention_msa_module,
)
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    MSACutlassDecodePlanCache,
    msa_cutlass_sparse_decode,
    prepare_decode_metadata,
    should_prepare_decode_metadata,
)
from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
    MiniMaxM3SparseMSABackend,
    MiniMaxM3SparseMSADecodeMetadata,
    MiniMaxM3SparseMSAImpl,
    MiniMaxM3SparseMSAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "fmha_sm100 sparse decode requires SM100 (Blackwell).",
        allow_module_level=True,
    )


HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 16
DEFAULT_QUERY_LEN = 4
SM_SCALE = HEAD_DIM**-0.5


@pytest.mark.parametrize(
    (
        "batch_size",
        "decode_query_len",
        "num_q_heads",
        "num_kv_heads",
        "expected",
    ),
    [
        pytest.param(8, 4, 64, 4, False, id="tp1-below-min-batch"),
        pytest.param(16, 4, 64, 4, True, id="tp1-supported"),
        pytest.param(16, 4, 16, 1, True, id="tp4-min-batch"),
        pytest.param(24, 4, 16, 1, True, id="tp4-intermediate-batch"),
        pytest.param(32, 4, 16, 1, True, id="tp4-supported"),
        pytest.param(16, 1, 64, 4, True, id="tp1-query-len-1"),
        pytest.param(16, 1, 16, 1, True, id="tp4-query-len-1"),
        pytest.param(16, 2, 64, 4, True, id="tp1-query-len-2"),
        pytest.param(16, 2, 16, 1, True, id="tp4-query-len-2"),
        pytest.param(16, 32, 64, 4, True, id="query-len-upper-bound"),
        pytest.param(16, 0, 64, 4, False, id="query-len-zero"),
        pytest.param(16, 33, 64, 4, False, id="query-len-above-bound"),
    ],
)
def test_msa_cutlass_decode_static_dispatch(
    batch_size: int,
    decode_query_len: int,
    expected: bool,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    assert (
        should_prepare_decode_metadata(
            batch_size,
            decode_query_len,
            decode_backend="cutlass",
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype="fp8_e4m3",
            page_size=BLOCK_SIZE,
            topk_blocks=TOPK,
        )
        is expected
    )


def test_msa_cutlass_decode_static_dispatch_requires_opt_in() -> None:
    assert not should_prepare_decode_metadata(
        32,
        DEFAULT_QUERY_LEN,
        decode_backend="triton",
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype="fp8_e4m3",
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


def test_msa_cutlass_decode_static_dispatch_accepts_fp8_alias() -> None:
    assert should_prepare_decode_metadata(
        32,
        DEFAULT_QUERY_LEN,
        decode_backend="cutlass",
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype="fp8",
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


@pytest.mark.parametrize(
    "kv_cache_dtype",
    ["auto", "bfloat16", "float16", "fp8_e5m2"],
)
def test_msa_cutlass_decode_static_dispatch_requires_fp8_e4m3(
    kv_cache_dtype: str,
) -> None:
    assert not should_prepare_decode_metadata(
        32,
        DEFAULT_QUERY_LEN,
        decode_backend="cutlass",
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype=kv_cache_dtype,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


def test_msa_cutlass_decode_static_dispatch_requires_sm100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    assert not should_prepare_decode_metadata(
        32,
        DEFAULT_QUERY_LEN,
        decode_backend="cutlass",
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype="fp8",
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (AttentionBackendEnum.CUTLASS_MSA, "cutlass"),
        (AttentionBackendEnum.TRITON_MSA, "triton"),
    ],
)
def test_msa_attention_backend_alias(
    backend: AttentionBackendEnum,
    expected: str,
) -> None:
    config = AttentionConfig(backend=backend)
    assert config.backend is None
    assert config.minimax_m3_msa_decode_backend == expected


def test_msa_backend_owns_msa_metadata_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_cls, impl_cls = select_main_backend_and_impl_cls(
        topk_blocks=TOPK,
        kv_cache_dtype="fp8_e4m3",
        num_kv_heads=1,
    )
    assert backend_cls is MiniMaxM3SparseMSABackend
    assert impl_cls is MiniMaxM3SparseMSAImpl

    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    backend_cls, impl_cls = select_main_backend_and_impl_cls(
        topk_blocks=TOPK,
        kv_cache_dtype="fp8_e4m3",
        num_kv_heads=1,
    )
    assert backend_cls is MiniMaxM3SparseBackend
    assert impl_cls is MiniMaxM3SparseTritonImpl


def test_msa_metadata_builder_prepares_cutlass_for_regular_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = 16
    block_table = torch.zeros(batch, 3, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((batch,), 257, dtype=torch.int32, device="cuda")
    base_metadata = SimpleNamespace(
        num_decodes=batch,
        decode=SimpleNamespace(
            block_table=block_table,
            seq_lens=seq_lens,
            decode_query_len=1,
        ),
    )
    monkeypatch.setattr(
        MiniMaxM3SparseMetadataBuilder,
        "build",
        lambda *args, **kwargs: base_metadata,
    )
    expected_metadata = object()
    monkeypatch.setattr(
        sparse_attention_msa_module,
        "prepare_decode_metadata",
        lambda *args, **kwargs: expected_metadata,
    )

    builder = object.__new__(MiniMaxM3SparseMSAMetadataBuilder)
    builder.num_q_heads = 64
    builder.num_kv_heads = 4
    builder.topk_blocks = TOPK
    builder.kv_cache_spec = SimpleNamespace(num_kv_heads=4)
    builder.kv_cache_dtype = "fp8_e4m3"
    builder.decode_backend = "cutlass"
    builder.msa_cutlass_plan_cache = object()

    metadata = builder.build(
        0,
        SimpleNamespace(
            seq_lens_cpu_upper_bound=torch.full((batch,), 257, dtype=torch.int32)
        ),
    )

    assert isinstance(metadata.decode, MiniMaxM3SparseMSADecodeMetadata)
    assert metadata.decode.decode_query_len == 1
    assert metadata.decode.msa_cutlass is expected_metadata


def test_msa_cutlass_plan_cache_keys_query_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = 16
    block_table = torch.zeros(batch, 3, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((batch,), 257, dtype=torch.int32, device="cuda")
    seq_lens_cpu = torch.full((batch,), 257, dtype=torch.int32)
    plan_cache = MSACutlassDecodePlanCache()
    built_query_lens: list[int] = []

    def fake_build_plan(**kwargs):
        query_len = kwargs["decode_query_len"]
        built_query_lens.append(query_len)
        num_rows = batch * query_len
        return (
            None,
            None,
            None,
            {
                "kv_segment_lens": torch.empty(
                    num_rows, dtype=torch.int32, device="cuda"
                ),
                "qo_offset": torch.empty(num_rows, dtype=torch.int32, device="cuda"),
            },
        )

    monkeypatch.setattr(plan_cache, "_build_plan", fake_build_plan)
    first = prepare_decode_metadata(
        block_table,
        seq_lens,
        seq_lens_cpu,
        1,
        num_q_heads=64,
        num_kv_heads=4,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    repeated = prepare_decode_metadata(
        block_table,
        seq_lens,
        seq_lens_cpu,
        1,
        num_q_heads=64,
        num_kv_heads=4,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    different = prepare_decode_metadata(
        block_table,
        seq_lens,
        seq_lens_cpu,
        2,
        num_q_heads=64,
        num_kv_heads=4,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    current_platform.synchronize()

    assert first.plan is repeated.plan
    assert different.plan is not first.plan
    assert built_query_lens == [1, 2]


def test_query_fp8_stays_valid_when_cutlass_plan_appears_on_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.envs as envs
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    envs.disable_envs_cache()

    num_tokens = 16
    layer_name = "model.layers.0.self_attn.attn"
    seq_lens = torch.full((num_tokens,), 257, dtype=torch.int32, device="cuda")
    block_table = torch.zeros(num_tokens, 3, dtype=torch.int32, device="cuda")
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    decode = MiniMaxM3SparseMSADecodeMetadata(
        seq_lens=seq_lens,
        block_table=block_table,
        decode_query_len=1,
        msa_cutlass=None,
    )
    metadata = MiniMaxM3SparseMetadata(
        seq_lens=seq_lens,
        max_seq_len=257,
        slot_mapping=slot_mapping,
        num_actual_tokens=num_tokens,
        num_decodes=num_tokens,
        num_decode_tokens=num_tokens,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={layer_name: metadata},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
    )

    impl = object.__new__(MiniMaxM3SparseMSAImpl)
    impl.use_cutlass_decode = True
    attention = SimpleNamespace(impl=impl, q_size=HEAD_DIM)
    qkv = torch.empty(num_tokens, HEAD_DIM, device="cuda")
    observed_ptrs: list[int] = []

    @eager_break_during_capture
    def run_attention(query_fp8: torch.Tensor | None) -> None:
        if impl.should_use_msa_decode(layer_name):
            assert query_fp8 is not None
            observed_ptrs.append(query_fp8.data_ptr())

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream), override_forward_context(forward_context):
        capture = BreakableCUDAGraphCapture()
        with capture:
            query_fp8 = MiniMaxM3SparseAttention._allocate_query_fp8(attention, qkv)
            if query_fp8 is not None:
                query_fp8.zero_()
            run_attention(query_fp8)
            qkv.zero_()

        assert capture.num_graphs == 2
        assert capture.num_eager_breaks == 1
        assert observed_ptrs == []

        decode.msa_cutlass = object()  # type: ignore[assignment]
        for _ in range(3):
            capture.replay()
        stream.synchronize()

    assert len(observed_ptrs) == 3
    assert len(set(observed_ptrs)) == 1


def _make_topk(
    seq_lens: list[int],
    num_kv_heads: int,
    query_len: int,
) -> torch.Tensor:
    topk = torch.full(
        (len(seq_lens) * query_len, num_kv_heads, TOPK),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for request, seq_len in enumerate(seq_lens):
        for local_query in range(query_len):
            token = request * query_len + local_query
            visible_tokens = seq_len - query_len + local_query + 1
            visible_pages = math.ceil(visible_tokens / BLOCK_SIZE)
            topk[token, :, :visible_pages] = torch.arange(
                visible_pages, dtype=torch.int32, device="cuda"
            )
    return topk


@pytest.mark.parametrize(
    (
        "num_q_heads",
        "num_kv_heads",
        "num_request_pairs",
        "query_len",
        "capture_graph",
    ),
    [
        pytest.param(64, 4, 8, 1, True, id="tp1-query-len-1"),
        pytest.param(64, 4, 8, 2, False, id="tp1-query-len-2"),
        pytest.param(64, 4, 8, 3, False, id="tp1-query-len-3"),
        pytest.param(64, 4, 8, 4, True, id="tp1-query-len-4"),
        pytest.param(64, 4, 8, 8, False, id="tp1-query-len-8"),
        pytest.param(16, 1, 8, 1, True, id="tp4-query-len-1"),
        pytest.param(16, 1, 16, 4, True, id="tp4-query-len-4"),
    ],
)
def test_msa_cutlass_decode_matches_triton_with_interleaved_cache(
    num_q_heads: int,
    num_kv_heads: int,
    num_request_pairs: int,
    query_len: int,
    capture_graph: bool,
) -> None:
    torch.manual_seed(0)
    seq_lens_list = [257, 513] * num_request_pairs
    seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
    seq_lens = seq_lens_cpu.cuda()
    pages_per_request = [math.ceil(seq_len / BLOCK_SIZE) for seq_len in seq_lens_list]
    num_pages = sum(pages_per_request)
    max_pages = max(pages_per_request)

    block_table = torch.zeros(
        len(seq_lens_list), max_pages, dtype=torch.int32, device="cuda"
    )
    physical_pages = torch.randperm(num_pages, dtype=torch.int32, device="cuda")
    offset = 0
    for request, request_pages in enumerate(pages_per_request):
        block_table[request, :request_pages] = physical_pages[
            offset : offset + request_pages
        ]
        offset += request_pages

    key = (
        torch.randn(
            num_pages,
            num_kv_heads,
            BLOCK_SIZE,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    value = (torch.randn_like(key, dtype=torch.bfloat16) * 0.25).to(torch.float8_e4m3fn)
    kv_cache = torch.cat((key, value), dim=-1)
    assert kv_cache.stride() == (
        num_kv_heads * BLOCK_SIZE * 2 * HEAD_DIM,
        BLOCK_SIZE * 2 * HEAD_DIM,
        2 * HEAD_DIM,
        1,
    )

    num_query_tokens = len(seq_lens_list) * query_len
    query = torch.randn(
        num_query_tokens,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    query_fp8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    ops.scaled_fp8_quant(
        query.view(num_query_tokens, -1),
        scale=q_scale,
        output=query_fp8.view(num_query_tokens, -1),
    )
    query_dequantized = query_fp8.to(torch.bfloat16) * q_scale

    topk_token_major = _make_topk(seq_lens_list, num_kv_heads, query_len)
    expected = torch.empty_like(query)
    minimax_m3_sparse_attn_decode(
        query_dequantized,
        kv_cache,
        topk_token_major.transpose(0, 1),
        block_table,
        seq_lens,
        num_kv_heads,
        SM_SCALE,
        expected,
        query_len,
        k_scale=None,
        v_scale=None,
    )

    plan_cache = MSACutlassDecodePlanCache()
    metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        seq_lens_cpu,
        query_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    assert metadata.page_table.data_ptr() == block_table.data_ptr()
    actual = torch.empty_like(query)
    msa_cutlass_sparse_decode(
        query_fp8,
        kv_cache,
        topk_token_major,
        actual,
        metadata,
        scale=SM_SCALE,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )

    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    # The same captured plan must remain correct as ragged lengths change.
    updated_seq_lens_list = [129, 385] * num_request_pairs
    seq_lens.copy_(
        torch.tensor(updated_seq_lens_list, dtype=torch.int32, device="cuda")
    )
    updated_seq_lens_cpu = torch.tensor(updated_seq_lens_list, dtype=torch.int32)
    topk_token_major.copy_(_make_topk(updated_seq_lens_list, num_kv_heads, query_len))
    updated_metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        updated_seq_lens_cpu,
        query_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    assert updated_metadata.plan is metadata.plan
    assert updated_metadata.page_table.data_ptr() == metadata.page_table.data_ptr()

    minimax_m3_sparse_attn_decode(
        query_dequantized,
        kv_cache,
        topk_token_major.transpose(0, 1),
        block_table,
        seq_lens,
        num_kv_heads,
        SM_SCALE,
        expected,
        query_len,
        k_scale=None,
        v_scale=None,
    )
    msa_cutlass_sparse_decode(
        query_fp8,
        kv_cache,
        topk_token_major,
        actual,
        updated_metadata,
        scale=SM_SCALE,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    if capture_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            msa_cutlass_sparse_decode(
                query_fp8,
                kv_cache,
                topk_token_major,
                actual,
                updated_metadata,
                scale=SM_SCALE,
                q_scale_float=1.0,
                k_scale_float=1.0,
                v_scale_float=1.0,
            )

        replay_seq_lens_list = [257, 513] * num_request_pairs
        seq_lens.copy_(
            torch.tensor(replay_seq_lens_list, dtype=torch.int32, device="cuda")
        )
        topk_token_major.copy_(
            _make_topk(replay_seq_lens_list, num_kv_heads, query_len)
        )
        prepare_decode_metadata(
            block_table,
            seq_lens,
            torch.tensor(replay_seq_lens_list, dtype=torch.int32),
            query_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            page_size=BLOCK_SIZE,
            topk_blocks=TOPK,
            plan_cache=plan_cache,
        )
        minimax_m3_sparse_attn_decode(
            query_dequantized,
            kv_cache,
            topk_token_major.transpose(0, 1),
            block_table,
            seq_lens,
            num_kv_heads,
            SM_SCALE,
            expected,
            query_len,
            k_scale=None,
            v_scale=None,
        )
        graph.replay()
        current_platform.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
