# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only cache-format gating; mocked architectures are not GPU validation."""

import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import AttentionConfig
from vllm.v1.attention.backends.mla import indexer


@pytest.mark.parametrize("max_num_batched_tokens", [32, 4096])
@pytest.mark.parametrize("rows,total_ctas", [(2, 96), (8, 1024)])
def test_rocm_fp4_decode_uses_temporary_logits_and_precomputed_schedule(
    monkeypatch, max_num_batched_tokens, rows, total_ctas
):
    """Decode logits follow batch rows, independent of the caller's capacity."""
    import torch

    from vllm.models.deepseek_v4.amd.ops import sparse_attn_indexer as sparse

    scorer = Mock(side_effect=lambda *args, out, **kwargs: out)
    decode_module = ModuleType("aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4")
    monkeypatch.setattr(
        decode_module, "flydsl_pa_mqa_logits_fp4", scorer, raising=False
    )
    prefill_module = ModuleType(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    monkeypatch.setattr(
        prefill_module, "flydsl_pa_mqa_logits_fp4_prefill", Mock(), raising=False
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4",
        decode_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill",
        prefill_module,
    )
    topk = Mock()
    monkeypatch.setattr(sparse.ops, "top_k_per_row_decode", topk)

    cta_info = torch.zeros((total_ctas, 4), dtype=torch.int32)
    decode = SimpleNamespace(
        requires_padding=False,
        seq_lens=torch.full((rows, 1), 64, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        fp4_cta_info=cta_info,
        fp4_total_ctas=total_ctas,
    )
    metadata = SimpleNamespace(
        num_prefills=0, num_decodes=2, num_decode_tokens=rows, decode=decode
    )
    cache = torch.zeros((1, 64, 68), dtype=torch.uint8)
    q_quant = torch.zeros((rows, 64, 64), dtype=torch.uint8)
    q_scale = torch.zeros((rows, 1, 4, 16, 4), dtype=torch.uint8)
    weights = torch.zeros((rows, 64), dtype=torch.bfloat16)
    output = torch.empty((max_num_batched_tokens, 8), dtype=torch.int32)

    result = sparse._rocm_fp4_sparse_attn_indexer(
        cache,
        q_quant,
        q_scale,
        weights,
        128,
        64,
        8,
        output,
        metadata,
    )

    assert scorer.call_args.kwargs["cta_info"] is cta_info
    assert scorer.call_args.kwargs["total_ctas"] == total_ctas
    assert "parallel_unit_num" not in scorer.call_args.kwargs
    logits = scorer.call_args.kwargs["out"]
    assert logits.shape == (rows, 64)
    assert logits.dtype == torch.float32 and logits.device == q_quant.device
    assert topk.call_args.args[0] is logits
    assert topk.call_args.args[2].data_ptr() == decode.seq_lens.data_ptr()
    assert topk.call_args.args[3].data_ptr() == output.data_ptr()
    assert result is output


@pytest.mark.parametrize(
    "next_n,max_capture,capture_sizes,max_decode_tokens",
    [(3, None, None, 12), (1, 24, [16], 24), (1, 16, [32], 32)],
)
def test_rocm_fp4_builder_selects_bounded_graph_stable_schedule(
    monkeypatch, next_n, max_capture, capture_sizes, max_decode_tokens
):
    """Exercise host allocation/build with GPU work and AITER selection mocked."""
    import torch

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    _platform(monkeypatch, rocm=True)
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", lambda: True)
    monkeypatch.setattr(indexer, "num_compute_units", lambda _: 256)
    selector = Mock(side_effect=lambda batch_size, max_seq_len: 8 * batch_size)
    schedule = Mock(
        side_effect=lambda seq_lens, *, cta_info_out, parallel_unit_num, **kwargs: (
            None,
            cta_info_out,
            parallel_unit_num,
        )
    )
    decode_module = ModuleType("aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4")
    monkeypatch.setattr(
        decode_module, "default_varctx_parallel_unit_num", selector, raising=False
    )
    monkeypatch.setattr(
        decode_module, "compute_varctx_schedule", schedule, raising=False
    )
    monkeypatch.setitem(sys.modules, decode_module.__name__, decode_module)

    config = _config("mxfp4")
    config.model_config.max_model_len = 1024
    config.attention_config.sparse_mla_prefill_workspace_factor = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.cp_kv_cache_interleave_size = 1
    config.scheduler_config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64)
    config.speculative_config = SimpleNamespace(num_speculative_tokens=next_n - 1)
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
        builder = indexer.DeepseekV32IndexerMetadataBuilder(
            spec, ["indexer"], config, torch.device("cpu"), block_table_width=1
        )
    selector.assert_called_once_with(max_decode_tokens, 256)
    capacity = 8 * max_decode_tokens
    buffer = builder.fp4_cta_info_buffer
    assert buffer.shape == (capacity, 4)
    assert buffer.dtype == torch.int32 and buffer.device == torch.device("cpu")
    assert all(
        call.kwargs["dtype"] != torch.float32 for call in allocate.call_args_list
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
        # Include graph-padded rows in selection, not just live requests.
        rows = 2 * next_n + (2 if capturing else 0)
        common.num_actual_tokens = rows
        metadata = (
            builder.build_for_cudagraph_capture(common)
            if capturing
            else builder.build(0, common)
        )
        selector.assert_called_with(rows, 256)
        assert metadata.decode.fp4_total_ctas == 8 * rows
        assert metadata.decode.fp4_cta_info.shape == (8 * rows, 4)
        assert metadata.decode.fp4_cta_info.data_ptr() == buffer.data_ptr()
        assert schedule.call_args.kwargs == {
            "block_k": 256,
            "parallel_unit_num": 8 * rows,
            "max_seq_len": 256,
            "next_n": 1,
            "cta_info_out": metadata.decode.fp4_cta_info,
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


@pytest.mark.parametrize("total_ctas", [64, 1280])
def test_rocm_fp4_prefill_forwards_planned_width_and_schedule(monkeypatch, total_ctas):
    import torch

    from vllm.models.deepseek_v4.amd.ops import sparse_attn_indexer as sparse

    scorer = Mock(return_value=torch.zeros((3, 192), dtype=torch.float32))
    decode_module = ModuleType("aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4")
    monkeypatch.setattr(
        decode_module, "flydsl_pa_mqa_logits_fp4", Mock(), raising=False
    )
    prefill_module = ModuleType(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    monkeypatch.setattr(
        prefill_module, "flydsl_pa_mqa_logits_fp4_prefill", scorer, raising=False
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4",
        decode_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill",
        prefill_module,
    )
    topk = Mock()
    monkeypatch.setattr(sparse.ops, "top_k_per_row_prefill", topk)

    row_to_batch = torch.tensor([0, 1, 1], dtype=torch.int32)
    starts = torch.zeros(3, dtype=torch.int32)
    ends = torch.tensor([128, 160, 192], dtype=torch.int32)
    cta_info = torch.zeros((total_ctas, 6), dtype=torch.int32)
    chunk = SimpleNamespace(
        local_total_seq_lens=320,
        token_start=0,
        token_end=3,
        block_table=torch.zeros((2, 4), dtype=torch.int32),
        logits_width=192,
        fp4_windows=(row_to_batch, starts, ends),
        fp4_cta_info=cta_info,
        fp4_total_ctas=total_ctas,
    )
    metadata = SimpleNamespace(
        num_prefills=2,
        num_decodes=0,
        num_decode_tokens=0,
        prefill=SimpleNamespace(
            chunks=[chunk], fp4_logits=torch.empty((3 * 192,), dtype=torch.float32)
        ),
        decode=None,
    )

    sparse._rocm_fp4_sparse_attn_indexer(
        torch.zeros((4, 64, 68), dtype=torch.uint8),
        torch.zeros((3, 64, 64), dtype=torch.uint8),
        torch.zeros((3, 1, 4, 16, 4), dtype=torch.uint8),
        torch.zeros((3, 64), dtype=torch.float32),
        128,
        4096,
        8,
        torch.empty((3, 8), dtype=torch.int32),
        metadata,
    )

    assert scorer.call_args.args[6] is row_to_batch
    assert scorer.call_args.args[7] is starts
    assert scorer.call_args.args[8] is ends
    assert scorer.call_args.args[9] == 192
    assert scorer.call_args.kwargs["cta_info"] is cta_info
    assert scorer.call_args.kwargs["n_ctas"] == total_ctas
    assert "parallel_unit_num" not in scorer.call_args.kwargs
    assert scorer.call_args.kwargs["out"].shape == (3, 192)
    assert (
        scorer.call_args.kwargs["out"].untyped_storage().data_ptr()
        == metadata.prefill.fp4_logits.untyped_storage().data_ptr()
    )
    assert topk.call_args.args[2] is ends


def test_rocm_fp4_prefill_skips_scoring_when_topk_covers_context(monkeypatch):
    import torch

    from vllm.models.deepseek_v4.amd.ops import sparse_attn_indexer as sparse

    scorer = Mock()
    decode_module = ModuleType("aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4")
    monkeypatch.setattr(
        decode_module, "flydsl_pa_mqa_logits_fp4", Mock(), raising=False
    )
    prefill_module = ModuleType(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    monkeypatch.setattr(
        prefill_module, "flydsl_pa_mqa_logits_fp4_prefill", scorer, raising=False
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4",
        decode_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill",
        prefill_module,
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
        logits_width=8,
        fp4_windows=(torch.tensor([0, 1, 1]), torch.zeros(3), ends),
        fp4_cta_info=None,
        fp4_total_ctas=None,
    )
    metadata = SimpleNamespace(
        num_prefills=2,
        num_decodes=0,
        num_decode_tokens=0,
        prefill=SimpleNamespace(chunks=[chunk], fp4_logits=None),
        decode=None,
    )
    output = torch.empty((3, 8), dtype=torch.int32)

    sparse._rocm_fp4_sparse_attn_indexer(
        torch.zeros((2, 64, 68), dtype=torch.uint8),
        torch.zeros((3, 64, 64), dtype=torch.uint8),
        torch.zeros((3, 1, 4, 16, 4), dtype=torch.uint8),
        torch.zeros((3, 64), dtype=torch.bfloat16),
        128,
        4096,
        8,
        output,
        metadata,
    )

    scorer.assert_not_called()
    topk.assert_not_called()
    assert fill.call_args.args[0] is ends
    assert fill.call_args.args[1].untyped_storage().data_ptr() == output.data_ptr()


@pytest.mark.parametrize(
    "rows,width,query_slice,total_ctas",
    [(4, 2048, None, 64), (16, 32768, slice(4, 8), 1280), (640, 8192, None, 1024)],
)
def test_fp4_prefill_schedule_uses_chunk_shape_selector(
    monkeypatch, rows, width, query_slice, total_ctas
):
    """Chunk scheduling must honor the selector, including grids beyond 512."""
    import torch

    selected_rows = (
        rows if query_slice is None else query_slice.stop - query_slice.start
    )
    windows = (
        torch.zeros(selected_rows, dtype=torch.int32),
        torch.zeros(selected_rows, dtype=torch.int32),
        torch.full((selected_rows,), width, dtype=torch.int32),
    )
    cta_info = torch.empty((total_ctas, 6), dtype=torch.int32)
    selector = Mock(return_value=total_ctas)
    schedule = Mock(return_value=(None, cta_info, total_ctas))
    decode_module = ModuleType("aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4")
    monkeypatch.setattr(
        decode_module, "default_varctx_parallel_unit_num", selector, raising=False
    )
    prefill_module = ModuleType(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    monkeypatch.setattr(
        prefill_module, "compute_prefill_schedule", schedule, raising=False
    )
    monkeypatch.setitem(sys.modules, decode_module.__name__, decode_module)
    monkeypatch.setitem(sys.modules, prefill_module.__name__, prefill_module)
    monkeypatch.setattr(indexer, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", Mock())
    monkeypatch.setattr(
        indexer, "_build_rocm_fp4_prefill_windows", lambda *args: windows
    )
    query_start_loc = torch.tensor([0, rows], dtype=torch.int32)
    compressed_seq_lens = torch.tensor([width], dtype=torch.int32)

    chunk = indexer.build_prefill_chunk_metadata(
        0,
        1,
        query_start_loc,
        query_start_loc,
        compressed_seq_lens * 4,
        compressed_seq_lens,
        compressed_seq_lens,
        torch.zeros((1, (width + 63) // 64), dtype=torch.int32),
        compress_ratio=4,
        query_slice=query_slice,
        logits_width=width,
        build_fp4_schedule=True,
        fp4_topk_tokens=8,
    )

    selector.assert_called_once_with(selected_rows, width)
    schedule.assert_called_once_with(
        *windows, block_k=256, parallel_unit_num=total_ctas, max_seq_len=width
    )
    assert chunk.token_end - chunk.token_start == selected_rows
    assert chunk.fp4_cta_info is cta_info
    assert chunk.fp4_total_ctas == total_ctas


def test_fp4_prefill_planner_uses_max_width_and_ignores_gather_workspace():
    import torch

    seq_lens = torch.tensor([300, 200, 100])
    query_lens = torch.tensor([10, 20, 30])
    max_logits_bytes = 300 * 60 * 4

    fp4_chunks = (
        indexer.DeepseekV32IndexerMetadataBuilder._split_fp4_indexer_prefill_chunks(
            seq_lens, query_lens, max_logits_bytes
        )
    )
    gathered_chunks = (
        indexer.DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
            seq_lens,
            query_lens,
            workspace_size=300,
            max_logits_bytes=max_logits_bytes,
        )
    )

    assert fp4_chunks == [(slice(0, 3), slice(0, 60))]
    assert gathered_chunks == [
        (slice(0, 1), slice(0, 10)),
        (slice(1, 3), slice(0, 50)),
    ]


def test_fp4_prefill_planner_splits_large_request_only_by_logits_budget():
    import torch

    split_fp4 = (
        indexer.DeepseekV32IndexerMetadataBuilder._split_fp4_indexer_prefill_chunks
    )
    chunks = split_fp4(
        torch.tensor([1024]),
        torch.tensor([1000]),
        max_logits_bytes=1024 * 100 * 4,
    )

    assert chunks == [
        (slice(0, 1), slice(start, start + 100)) for start in range(0, 1000, 100)
    ]


def _config(dtype=None, *, dcp=1, pcp=1, model_type="deepseek_v4"):
    attention = (
        AttentionConfig() if dtype is None else AttentionConfig(indexer_kv_dtype=dtype)
    )
    return SimpleNamespace(
        attention_config=attention,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
    )


def _platform(monkeypatch, *, rocm, capability=100, gfx950=True):
    platform = SimpleNamespace(
        is_rocm=lambda: rocm,
        is_cuda=lambda: not rocm,
        is_device_capability_family=Mock(
            side_effect=lambda family: capability // 10 == family // 10
        ),
    )
    monkeypatch.setattr(indexer, "current_platform", platform)
    # Avoid importing ROCm driver libraries on CUDA/CPU hosts.
    rocm_module = ModuleType("vllm.platforms.rocm")
    gfx_probe = Mock(return_value=gfx950)
    monkeypatch.setattr(rocm_module, "on_gfx950", gfx_probe, raising=False)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm_module)
    return platform, gfx_probe


@pytest.mark.parametrize("rocm", [False, True])
@pytest.mark.parametrize("dtype", [None, "auto", "fp8"])
def test_fp8_default_never_probes_fp4_dependencies(monkeypatch, rocm, dtype):
    platform, gfx_probe = _platform(monkeypatch, rocm=rocm, gfx950=False)
    aiter_probe = Mock(side_effect=AssertionError("FP8 must not probe AITER"))
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", aiter_probe)

    # FP4's context-parallel restriction must not leak into the FP8 default.
    assert indexer.dsa_indexer_uses_fp4(_config(dtype, dcp=2, pcp=2)) is False
    aiter_probe.assert_not_called()
    gfx_probe.assert_not_called()
    platform.is_device_capability_family.assert_not_called()


def test_rocm_gfx950_fp4_is_explicit_opt_in(monkeypatch):
    platform, gfx_probe = _platform(monkeypatch, rocm=True)
    aiter_probe = Mock(return_value=True)
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", aiter_probe)

    assert indexer.dsa_indexer_uses_fp4(_config("mxfp4")) is True
    gfx_probe.assert_called_once_with()
    aiter_probe.assert_called_once_with()
    assert indexer.rocm_flydsl_indexer_uses_fp4(_config("mxfp4")) is True
    platform.is_device_capability_family.assert_not_called()


@pytest.mark.parametrize("cdna", [3, 4])
def test_rocm_fp4_deepseek_v41_keeps_its_own_gate(monkeypatch, cdna):
    _, gfx_probe = _platform(monkeypatch, rocm=True)
    rocm_module = sys.modules["vllm.platforms.rocm"]
    monkeypatch.setattr(rocm_module, "get_cdna_version", lambda: cdna, raising=False)
    aiter_ops = ModuleType("vllm._aiter_ops")
    monkeypatch.setattr(
        aiter_ops,
        "rocm_aiter_ops",
        SimpleNamespace(is_enabled=lambda: True),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_ops)
    aiter_probe = Mock(side_effect=AssertionError("V4.1 must not probe FlyDSL"))
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", aiter_probe)
    config = _config("mxfp4", dcp=2, model_type="deepseek_v41")

    if cdna == 4:
        assert indexer.dsa_indexer_uses_fp4(config) is True
        # Uses MXFP4, but never DeepSeek-V4's FlyDSL buffers, schedules or checks.
        assert indexer.rocm_flydsl_indexer_uses_fp4(config) is False
    else:
        with pytest.raises(ValueError, match="requires CDNA4"):
            indexer.dsa_indexer_uses_fp4(config)
    gfx_probe.assert_not_called()
    aiter_probe.assert_not_called()


@pytest.mark.parametrize("model_type", ["deepseek_v32", "glm_moe_dsa"])
def test_rocm_fp4_rejects_models_other_than_deepseek_v4(monkeypatch, model_type):
    _, gfx_probe = _platform(monkeypatch, rocm=True)
    aiter_probe = Mock(side_effect=AssertionError("unsupported model must fail first"))
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", aiter_probe)
    config = _config("mxfp4", model_type=model_type)

    with pytest.raises(ValueError, match="only supported for DeepSeek-V4"):
        indexer.dsa_indexer_uses_fp4(config)
    assert indexer.rocm_flydsl_indexer_uses_fp4(config) is False
    gfx_probe.assert_not_called()
    aiter_probe.assert_not_called()


def test_rocm_fp4_rejects_other_architectures_before_aiter(monkeypatch):
    _platform(monkeypatch, rocm=True, gfx950=False)
    aiter_probe = Mock(side_effect=AssertionError("unsupported GPU must fail first"))
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", aiter_probe)

    with pytest.raises(ValueError, match="requires gfx950"):
        indexer.dsa_indexer_uses_fp4(_config("mxfp4"))
    aiter_probe.assert_not_called()


@pytest.mark.parametrize("dcp,pcp", [(2, 1), (1, 2), (2, 2)])
def test_rocm_fp4_rejects_context_parallelism_before_aiter(monkeypatch, dcp, pcp):
    _platform(monkeypatch, rocm=True)
    aiter_probe = Mock(side_effect=AssertionError("parallel gate must fail first"))
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", aiter_probe)

    with pytest.raises(ValueError, match="DCP=PCP=1"):
        indexer.dsa_indexer_uses_fp4(_config("mxfp4", dcp=dcp, pcp=pcp))
    aiter_probe.assert_not_called()


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
    ],
)
def test_aiter_fp4_probe_checks_schedule_abi(monkeypatch, missing):
    _platform(monkeypatch, rocm=True)
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
    monkeypatch.setattr(package, "pa_mqa_logits_fp4", modules["decode"], raising=False)
    monkeypatch.setattr(
        package, "pa_mqa_logits_fp4_prefill", modules["prefill"], raising=False
    )
    monkeypatch.setattr(
        package, "pa_mqa_logits_fp4_common", modules["common"], raising=False
    )
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(indexer.logger, "warning", Mock())
    indexer.aiter_mxfp4_available.cache_clear()
    try:
        assert indexer.dsa_indexer_uses_fp4(_config("mxfp4")) is (missing is None)
        assert indexer.logger.warning.call_count == (missing is not None)
    finally:
        indexer.aiter_mxfp4_available.cache_clear()


def test_missing_aiter_falls_back_to_fp8_with_warning(monkeypatch):
    _platform(monkeypatch, rocm=True)
    # Exercise the real import probe, rather than mocking its Boolean result.
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.kernels.mqa_logits", None)
    warning = Mock()
    monkeypatch.setattr(indexer.logger, "warning", warning)
    indexer.aiter_mxfp4_available.cache_clear()
    try:
        config = _config("mxfp4")
        assert indexer.dsa_indexer_uses_fp4(config) is False
        assert config.attention_config.indexer_kv_dtype == "mxfp4"
        warning.assert_called_once()
        assert "falling back to the fp8 indexer" in warning.call_args.args[0]
    finally:
        indexer.aiter_mxfp4_available.cache_clear()


@pytest.mark.parametrize("capability", [100, 103, 90, 89, 120])
def test_nvidia_fp4_architecture_gate_never_probes_aiter(monkeypatch, capability):
    platform, gfx_probe = _platform(monkeypatch, rocm=False, capability=capability)
    aiter_probe = Mock(side_effect=AssertionError("CUDA must not probe AITER"))
    monkeypatch.setattr(indexer, "aiter_mxfp4_available", aiter_probe)
    config = _config("mxfp4", dcp=2, pcp=2)

    if capability // 10 == 10:
        assert indexer.dsa_indexer_uses_fp4(config) is True
    else:
        with pytest.raises(ValueError, match="requires Blackwell datacenter GPUs"):
            indexer.dsa_indexer_uses_fp4(config)
    platform.is_device_capability_family.assert_called_once_with(100)
    aiter_probe.assert_not_called()
    gfx_probe.assert_not_called()
