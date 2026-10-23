# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4's ROCm FlyDSL MXFP4 indexer: selection, metadata planning and
dispatch. aiter kernels are mocked, so apart from the windows kernel test
these are host contract tests, not GPU validation."""

import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import AttentionConfig
from vllm.models.deepseek_v4.amd.ops import sparse_attn_indexer as sparse
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla import rocm_flydsl_mxfp4_indexer as flydsl

DECODE_MODULE = "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4"
PREFILL_MODULE = "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"


def _aiter_modules(monkeypatch, **functions):
    """Stand-in FlyDSL modules; ``*prefill*`` names go to the prefill one."""
    decode = ModuleType(DECODE_MODULE)
    prefill = ModuleType(PREFILL_MODULE)
    for name, function in functions.items():
        setattr(prefill if "prefill" in name else decode, name, function)
    monkeypatch.setitem(sys.modules, DECODE_MODULE, decode)
    monkeypatch.setitem(sys.modules, PREFILL_MODULE, prefill)


def _metadata(**kwargs):
    defaults = dict(
        seq_lens=torch.zeros(1, dtype=torch.int32),
        max_seq_len=0,
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=0,
        num_prefill_tokens=0,
    )
    return flydsl.RocmFlyDslIndexerMetadata(**(defaults | kwargs))


def _score(monkeypatch, metadata, *, rows, cache_pages, max_model_len, output):
    monkeypatch.setattr(
        sparse,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"indexer": metadata}),
    )
    q = torch.zeros((rows, 64, 64), dtype=torch.uint8)
    return sparse.rocm_flydsl_mxfp4_sparse_attn_indexer(
        q,
        "indexer",
        torch.zeros((cache_pages, 64, 68), dtype=torch.uint8),
        q,
        torch.zeros((rows, 1, 4, 16, 4), dtype=torch.uint8),
        torch.zeros((rows, 64), dtype=torch.bfloat16),
        128,
        max_model_len,
        8,
        output,
    )


@pytest.mark.parametrize("max_num_batched_tokens", [32, 4096])
@pytest.mark.parametrize("rows,total_ctas", [(2, 96), (8, 1024)])
def test_decode_uses_temporary_logits_and_builder_schedule(
    monkeypatch, max_num_batched_tokens, rows, total_ctas
):
    """Decode logits follow the batch rows, not the caller's capacity."""
    scorer = Mock(side_effect=lambda *args, out, **kwargs: out)
    _aiter_modules(
        monkeypatch,
        flydsl_pa_mqa_logits_fp4=scorer,
        flydsl_pa_mqa_logits_fp4_prefill=Mock(),
    )
    topk = Mock()
    monkeypatch.setattr(sparse.ops, "top_k_per_row_decode", topk)
    cta_info = torch.zeros((total_ctas, 4), dtype=torch.int32)
    decode = SimpleNamespace(
        requires_padding=False,
        seq_lens=torch.full((rows, 1), 64, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
    )
    metadata = _metadata(
        num_decodes=2,
        num_decode_tokens=rows,
        decode=decode,
        decode_cta_info=cta_info,
        decode_total_ctas=total_ctas,
    )
    output = torch.empty((max_num_batched_tokens, 8), dtype=torch.int32)

    result = _score(
        monkeypatch,
        metadata,
        rows=rows,
        cache_pages=1,
        max_model_len=64,
        output=output,
    )

    assert scorer.call_args.kwargs["cta_info"] is cta_info
    assert scorer.call_args.kwargs["total_ctas"] == total_ctas
    assert "parallel_unit_num" not in scorer.call_args.kwargs
    logits = scorer.call_args.kwargs["out"]
    assert logits.shape == (rows, 64)
    assert logits.dtype == torch.float32
    assert topk.call_args.args[0] is logits
    assert topk.call_args.args[2].data_ptr() == decode.seq_lens.data_ptr()
    assert topk.call_args.args[3].data_ptr() == output.data_ptr()
    assert result is output


@pytest.mark.parametrize("total_ctas", [64, 1280])
def test_prefill_forwards_planned_width_and_schedule(monkeypatch, total_ctas):
    scorer = Mock(return_value=torch.zeros((3, 192), dtype=torch.float32))
    _aiter_modules(
        monkeypatch,
        flydsl_pa_mqa_logits_fp4=Mock(),
        flydsl_pa_mqa_logits_fp4_prefill=scorer,
    )
    topk = Mock()
    monkeypatch.setattr(sparse.ops, "top_k_per_row_prefill", topk)
    windows = (
        torch.tensor([0, 1, 1], dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([128, 160, 192], dtype=torch.int32),
    )
    cta_info = torch.zeros((total_ctas, 6), dtype=torch.int32)
    chunk = SimpleNamespace(
        local_total_seq_lens=320,
        token_start=0,
        token_end=3,
        block_table=torch.zeros((2, 4), dtype=torch.int32),
    )
    plan = flydsl.RocmFlyDslPrefillPlan(192, windows, cta_info, total_ctas)
    logits_storage = torch.empty((3 * 192,), dtype=torch.float32)
    metadata = _metadata(
        num_prefills=2,
        prefill=SimpleNamespace(chunks=[chunk]),
        prefill_plans=[plan],
        prefill_logits=logits_storage,
    )

    _score(
        monkeypatch,
        metadata,
        rows=3,
        cache_pages=4,
        max_model_len=4096,
        output=torch.empty((3, 8), dtype=torch.int32),
    )

    for position, window in enumerate(windows, start=6):
        assert scorer.call_args.args[position] is window
    assert scorer.call_args.args[9] == 192
    assert scorer.call_args.kwargs["cta_info"] is cta_info
    assert scorer.call_args.kwargs["n_ctas"] == total_ctas
    assert scorer.call_args.kwargs["out"].shape == (3, 192)
    assert (
        scorer.call_args.kwargs["out"].untyped_storage().data_ptr()
        == logits_storage.untyped_storage().data_ptr()
    )
    assert topk.call_args.args[2] is windows[2]


def test_prefill_skips_scoring_when_topk_covers_context(monkeypatch):
    scorer = Mock()
    _aiter_modules(
        monkeypatch,
        flydsl_pa_mqa_logits_fp4=Mock(),
        flydsl_pa_mqa_logits_fp4_prefill=scorer,
    )
    topk = Mock()
    monkeypatch.setattr(sparse.ops, "top_k_per_row_prefill", topk)
    fill = Mock()

    class FillKernel:
        def __getitem__(self, grid):
            assert grid == (3,)
            return fill

    monkeypatch.setattr(
        sparse, "_rocm_fp4_fill_all_prefill_indices_kernel", FillKernel()
    )
    ends = torch.tensor([4, 6, 8], dtype=torch.int32)
    chunk = SimpleNamespace(
        local_total_seq_lens=18,
        token_start=0,
        token_end=3,
        block_table=torch.zeros((2, 1), dtype=torch.int32),
    )
    plan = flydsl.RocmFlyDslPrefillPlan(
        8, (torch.tensor([0, 1, 1]), torch.zeros(3), ends)
    )
    metadata = _metadata(
        num_prefills=2, prefill=SimpleNamespace(chunks=[chunk]), prefill_plans=[plan]
    )
    output = torch.empty((3, 8), dtype=torch.int32)

    _score(
        monkeypatch,
        metadata,
        rows=3,
        cache_pages=2,
        max_model_len=4096,
        output=output,
    )

    scorer.assert_not_called()
    topk.assert_not_called()
    assert fill.call_args.args[0] is ends
    assert fill.call_args.args[1].untyped_storage().data_ptr() == output.data_ptr()


def test_profiling_run_reserves_without_aiter(monkeypatch):
    for module in (DECODE_MODULE, PREFILL_MODULE):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(
        sparse, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    output = torch.empty((4, 8), dtype=torch.int32)
    q = torch.zeros((4, 64, 64), dtype=torch.uint8)

    assert (
        sparse.rocm_flydsl_mxfp4_sparse_attn_indexer(
            q,
            "indexer",
            torch.zeros((1, 64, 68), dtype=torch.uint8),
            q,
            torch.zeros((4, 1, 4, 16, 4), dtype=torch.uint8),
            torch.zeros((4, 64), dtype=torch.bfloat16),
            128,
            64,
            8,
            output,
        )
        is output
    )


def _config(dtype="mxfp4", *, model_type="deepseek_v4"):
    return SimpleNamespace(
        attention_config=AttentionConfig(indexer_kv_dtype=dtype),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type, index_topk=8),
            max_model_len=1024,
        ),
    )


@pytest.mark.parametrize(
    "next_n,max_capture,capture_sizes,max_decode_tokens",
    [(3, None, None, 12), (1, 24, [16], 24), (1, 16, [32], 32)],
)
def test_builder_refreshes_one_graph_stable_decode_schedule(
    monkeypatch, next_n, max_capture, capture_sizes, max_decode_tokens
):
    """Host allocation and build, with GPU work and aiter's selector mocked."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    monkeypatch.setattr(
        indexer,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda=lambda: False,
            is_device_capability_family=lambda family: False,
        ),
    )
    monkeypatch.setattr(indexer, "dsa_indexer_uses_fp4", lambda config: True)
    monkeypatch.setattr(indexer, "num_compute_units", lambda _: 256)
    selector = Mock(side_effect=lambda batch_size, max_seq_len: 8 * batch_size)
    schedule = Mock(
        side_effect=lambda seq_lens, *, cta_info_out, parallel_unit_num, **kwargs: (
            None,
            cta_info_out,
            parallel_unit_num,
        )
    )
    _aiter_modules(
        monkeypatch,
        default_varctx_parallel_unit_num=selector,
        compute_varctx_schedule=schedule,
    )
    config = _config()
    config.attention_config.sparse_mla_prefill_workspace_factor = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.cp_kv_cache_interleave_size = 1
    config.scheduler_config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64)
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=next_n - 1, enable_adaptive_verification=False
    )
    config.num_speculative_tokens = next_n - 1
    config.compilation_config = SimpleNamespace(
        max_cudagraph_capture_size=max_capture, cudagraph_capture_sizes=capture_sizes
    )
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=68,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    allocate = Mock(wraps=torch.empty)
    with monkeypatch.context() as allocation_patch:
        allocation_patch.setattr(torch, "empty", allocate)
        builder = flydsl.RocmFlyDslIndexerMetadataBuilder(
            spec, ["indexer"], config, torch.device("cpu"), block_table_width=1
        )
    assert builder.use_flattening
    selector.assert_called_once_with(max_decode_tokens, 256)
    capacity = 8 * max_decode_tokens
    buffer = builder.decode_cta_info_buffer
    assert buffer.shape == (capacity, 4) and buffer.dtype == torch.int32
    # No logits are reserved up front.
    assert all(
        call.kwargs.get("dtype") != torch.float32 for call in allocate.call_args_list
    )

    monkeypatch.setattr(
        indexer, "get_compressed_slot_mapping", lambda *args, out, **kwargs: out
    )

    def prepare_decode_tensors(*, num_decode_tokens, **kwargs):
        seq_lens = builder.decode_seq_lens_buffer[:num_decode_tokens]
        seq_lens.fill_(256)
        return (
            seq_lens,
            builder.expanded_block_table_buffer[:num_decode_tokens],
            builder.decode_lens_buffer[:num_decode_tokens],
            num_decode_tokens,
            False,
        )

    monkeypatch.setattr(builder, "_prepare_decode_tensors", prepare_decode_tensors)
    query_start_loc = torch.tensor([0, next_n, 2 * next_n], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.full((2,), 256, dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2 * next_n,
        max_query_len=next_n,
        max_seq_len=256,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(2 * next_n + 2, dtype=torch.int64),
    )
    for capturing in (False, True):
        # Graph-padded rows count toward the schedule, not just live requests.
        rows = 2 * next_n + (2 if capturing else 0)
        common.num_actual_tokens = rows
        metadata = (
            builder.build_for_cudagraph_capture(common)
            if capturing
            else builder.build(0, common)
        )
        assert isinstance(metadata, flydsl.RocmFlyDslIndexerMetadata)
        selector.assert_called_with(rows, 256)
        assert metadata.decode_total_ctas == 8 * rows
        assert metadata.decode_cta_info is not None
        assert metadata.decode_cta_info.shape == (8 * rows, 4)
        assert metadata.decode_cta_info.data_ptr() == buffer.data_ptr()
        assert schedule.call_args.kwargs == {
            "block_k": 256,
            "parallel_unit_num": 8 * rows,
            "max_seq_len": 256,
            "next_n": 1,
            "cta_info_out": metadata.decode_cta_info,
        }
        torch.testing.assert_close(
            schedule.call_args.args[0], torch.full((rows,), 64, dtype=torch.int32)
        )

    selector.side_effect = None
    selector.return_value = capacity + 1
    schedule.reset_mock()
    with pytest.raises(AssertionError):
        builder.build(0, common)
    schedule.assert_not_called()


def _planner(topk_tokens=8):
    builder = object.__new__(flydsl.RocmFlyDslIndexerMetadataBuilder)
    builder.topk_tokens = topk_tokens
    builder.device = torch.device("cpu")
    builder.prefill_logits_buffer = None
    return builder


def test_prefill_plans_windows_schedules_and_shared_logits(monkeypatch):
    """Whole-request and query-sliced chunks, scored and top-k-covered."""
    selector = Mock(side_effect=lambda rows, width: 4 * rows)
    cta_info = torch.zeros((20, 6), dtype=torch.int32)
    schedule = Mock(return_value=(None, cta_info, 20))
    _aiter_modules(
        monkeypatch,
        default_varctx_parallel_unit_num=selector,
        compute_prefill_schedule=schedule,
    )
    windows = [tuple(Mock(name=f"window{i}.{j}") for j in range(3)) for i in range(3)]
    build_windows = Mock(side_effect=list(windows))
    monkeypatch.setattr(flydsl, "build_prefill_windows", build_windows)
    query_start_loc = torch.tensor([0, 2, 5, 9], dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc=query_start_loc, query_start_loc_cpu=query_start_loc
    )

    def chunk(start, end, num_reqs, width):
        return SimpleNamespace(
            token_start=start,
            token_end=end,
            num_reqs=num_reqs,
            max_seqlen_k=width,
            cu_seqlen_ks=Mock(name=f"ks{start}"),
            cu_seqlen_ke=Mock(name=f"ke{start}"),
        )

    # Requests 0-1 whole; request 2 cut on its query rows at 2 and at 3.
    chunks = [chunk(0, 5, 2, 300), chunk(7, 8, 1, 4), chunk(8, 9, 1, 40)]
    metadata = _metadata(num_prefills=3, prefill=SimpleNamespace(chunks=chunks))
    builder = _planner()

    builder._plan_prefill(metadata, common)

    calls = build_windows.call_args_list
    for call, expected in zip(
        calls, [(torch.tensor([0, 2, 5]), 5, 2, 0), (torch.tensor([0, 4]), 1, 1, 2)]
    ):
        starts, rows, num_reqs, offset = expected
        assert torch.equal(call.args[0], starts)
        assert call.args[3:] == (rows, num_reqs, offset)
    assert calls[2].args[3:] == (1, 1, 3)
    assert [plan.width for plan in metadata.prefill_plans] == [300, 4, 40]
    assert [plan.windows for plan in metadata.prefill_plans] == windows
    assert metadata.prefill_plans[1].cta_info is None
    assert [call.args for call in selector.call_args_list] == [(5, 300), (1, 40)]
    assert schedule.call_args_list[0].args == windows[0]
    assert schedule.call_args_list[0].kwargs == {
        "block_k": 256,
        "parallel_unit_num": 20,
        "max_seq_len": 300,
    }
    assert metadata.prefill_plans[0].cta_info is cta_info
    assert metadata.prefill_logits.numel() == 5 * 300
    assert metadata.prefill_logits.dtype == torch.float32

    smaller = _metadata(num_prefills=1, prefill=SimpleNamespace(chunks=chunks[2:]))
    build_windows.reset_mock(side_effect=True)
    build_windows.return_value = windows[0]
    builder._plan_prefill(smaller, common)
    assert smaller.prefill_logits is metadata.prefill_logits


def test_prefill_chunks_bound_rows_by_widest_context_not_gathered_sum():
    seq_lens = torch.tensor([300, 200, 100])
    query_lens = torch.tensor([10, 20, 30])
    max_logits_bytes = 300 * 60 * 4
    split = flydsl.RocmFlyDslIndexerMetadataBuilder._split_indexer_prefill_chunks

    assert split(_planner(), seq_lens, query_lens, 300, max_logits_bytes) == [
        (slice(0, 3), slice(0, 60))
    ]
    assert split(
        _planner(), torch.tensor([1024]), torch.tensor([1000]), 0, 1024 * 100 * 4
    ) == [(slice(0, 1), slice(start, start + 100)) for start in range(0, 1000, 100)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_prefill_windows_kernel_matches_reference():
    device = torch.device("cuda")
    query_start_loc = torch.tensor([0, 3, 3, 7, 12], dtype=torch.int32)
    rows, offset = 9, 2
    ks = torch.arange(rows, dtype=torch.int32) * 5
    ke = ks + torch.tensor([4, 0, 7, 1, 9, 3, 2, 6, 8], dtype=torch.int32)
    ke[1] = ks[1] - 3

    row_to_batch, starts, ends = flydsl.build_prefill_windows(
        query_start_loc.to(device), ks.to(device), ke.to(device), rows, 4, offset
    )

    queries = torch.arange(rows) + offset
    expected_batch = torch.searchsorted(query_start_loc[1:], queries, right=True)
    torch.testing.assert_close(row_to_batch.cpu(), expected_batch.to(torch.int32))
    assert not starts.any()
    torch.testing.assert_close(ends.cpu(), (ke - ks).clamp(min=0))


def _selection(monkeypatch, *, rocm=True, flag=True, fp4=True, available=True):
    monkeypatch.setattr(
        flydsl, "current_platform", SimpleNamespace(is_rocm=lambda: rocm)
    )
    monkeypatch.setattr(
        flydsl,
        "envs",
        SimpleNamespace(VLLM_ROCM_USE_AITER_FLYDSL_MXFP4_INDEXER=flag),
    )
    uses_fp4 = Mock(return_value=fp4)
    monkeypatch.setattr(flydsl, "dsa_indexer_uses_fp4", uses_fp4)
    probe = Mock(return_value=available)
    monkeypatch.setattr(flydsl, "aiter_flydsl_mxfp4_available", probe)
    logger = Mock()
    monkeypatch.setattr(flydsl, "logger", logger)
    return uses_fp4, probe, logger


def test_selection_is_opt_in_on_deepseek_v4_mxfp4(monkeypatch):
    uses_fp4, probe, logger = _selection(monkeypatch)

    assert flydsl.rocm_flydsl_mxfp4_indexer_enabled(_config()) is True
    uses_fp4.assert_called_once()
    probe.assert_called_once_with()
    logger.warning_once.assert_not_called()


@pytest.mark.parametrize(
    "rocm,flag,model_type",
    [
        (False, True, "deepseek_v4"),
        (True, False, "deepseek_v4"),
        (True, True, "deepseek_v41"),
        (True, True, "deepseek_v32"),
    ],
)
def test_selection_never_probes_outside_its_scope(monkeypatch, rocm, flag, model_type):
    uses_fp4, probe, _ = _selection(monkeypatch, rocm=rocm, flag=flag)

    assert (
        flydsl.rocm_flydsl_mxfp4_indexer_enabled(_config(model_type=model_type))
        is False
    )
    uses_fp4.assert_not_called()
    probe.assert_not_called()


def test_selection_keeps_fp8_without_probing_aiter(monkeypatch):
    _, probe, _ = _selection(monkeypatch, fp4=False)

    assert flydsl.rocm_flydsl_mxfp4_indexer_enabled(_config("fp8")) is False
    probe.assert_not_called()


def test_missing_flydsl_abi_keeps_the_triton_mxfp4_indexer(monkeypatch):
    _, probe, logger = _selection(monkeypatch, available=False)

    assert flydsl.rocm_flydsl_mxfp4_indexer_enabled(_config()) is False
    probe.assert_called_once_with()
    logger.warning_once.assert_called_once()
    assert "Triton paged MXFP4 indexer" in logger.warning_once.call_args.args[0]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("sparse_logits", [False, True])
def test_q_warmup_adds_bf16_weights_only_for_its_consumers(
    monkeypatch, enabled, sparse_logits
):
    from vllm.models.deepseek_v4.common.ops import fused_indexer_q

    monkeypatch.setattr(
        flydsl, "rocm_flydsl_mxfp4_indexer_enabled", lambda config: enabled
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(indexer_sparse_logits=sparse_logits)
    )

    dtypes = fused_indexer_q._indexer_weights_out_dtypes(config)

    assert dtypes == (
        (torch.float32, torch.bfloat16)
        if enabled or sparse_logits
        else (torch.float32,)
    )


@pytest.mark.parametrize(
    "missing",
    [
        None,
        ("decode", "default_varctx_parallel_unit_num", "batch_size"),
        ("decode", "default_varctx_parallel_unit_num", "max_seq_len"),
        ("decode", "default_varctx_parallel_unit_num", None),
        ("decode", "flydsl_pa_mqa_logits_fp4", "cta_info"),
        ("decode", "flydsl_pa_mqa_logits_fp4", "total_ctas"),
        ("decode", "compute_varctx_schedule", "cta_info_out"),
        ("decode", "compute_varctx_schedule", "next_n"),
        ("decode", "compute_varctx_schedule", None),
        ("prefill", "flydsl_pa_mqa_logits_fp4_prefill", "cta_info"),
        ("prefill", "flydsl_pa_mqa_logits_fp4_prefill", "n_ctas"),
        ("prefill", "compute_prefill_schedule", "max_seq_len"),
        ("prefill", "compute_prefill_schedule", None),
        ("common", "_i32_buffer", "byte_offset"),
    ],
)
def test_aiter_probe_checks_stride_and_schedule_abi(monkeypatch, missing):
    modules = {name: SimpleNamespace() for name in ("decode", "prefill", "common")}
    signatures: dict[str, dict[str, tuple[str, ...]]] = {
        "decode": {
            "default_varctx_parallel_unit_num": ("batch_size", "max_seq_len"),
            "flydsl_pa_mqa_logits_fp4": ("cta_info", "total_ctas"),
            "compute_varctx_schedule": (
                "block_k",
                "parallel_unit_num",
                "max_seq_len",
                "next_n",
                "cta_info_out",
            ),
        },
        "prefill": {
            "flydsl_pa_mqa_logits_fp4_prefill": ("cta_info", "n_ctas"),
            "compute_prefill_schedule": ("block_k", "parallel_unit_num", "max_seq_len"),
        },
        "common": {"_i32_buffer": ("byte_offset",)},
    }
    for name in ("decode", "prefill"):
        suffix = "_prefill" if name == "prefill" else ""
        for function in (
            f"build_pa_mqa_logits_fp4{suffix}_module",
            f"compile_pa_mqa_logits_fp4{suffix}",
        ):
            signatures[name][function] = (
                "kv_page_stride",
                "kv_scale_page_stride",
                "block_table_stride",
            )
    for name, functions in signatures.items():
        for function, parameters in functions.items():
            if missing == (name, function, None):
                continue
            signature = inspect.Signature(
                [
                    inspect.Parameter(parameter, inspect.Parameter.KEYWORD_ONLY)
                    for parameter in parameters
                    if missing != (name, function, parameter)
                ]
            )
            setattr(modules[name], function, Mock(__signature__=signature))
    package = ModuleType("aiter.ops.flydsl.kernels.mqa_logits")
    for name in ("decode", "prefill", "common"):
        suffix = "" if name == "decode" else f"_{name}"
        setattr(package, f"pa_mqa_logits_fp4{suffix}", modules[name])
    monkeypatch.setitem(sys.modules, package.__name__, package)
    flydsl.aiter_flydsl_mxfp4_available.cache_clear()
    try:
        assert flydsl.aiter_flydsl_mxfp4_available() is (missing is None)
    finally:
        flydsl.aiter_flydsl_mxfp4_available.cache_clear()


def test_aiter_probe_reports_missing_flydsl(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.kernels.mqa_logits", None)
    flydsl.aiter_flydsl_mxfp4_available.cache_clear()
    try:
        assert flydsl.aiter_flydsl_mxfp4_available() is False
    finally:
        flydsl.aiter_flydsl_mxfp4_available.cache_clear()
