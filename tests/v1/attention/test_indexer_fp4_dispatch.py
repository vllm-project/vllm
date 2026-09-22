# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only cache-format gating; mocked architectures are not GPU validation."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import AttentionConfig
from vllm.v1.attention.backends.mla import indexer


def test_rocm_fp4_decode_forwards_precomputed_schedule(monkeypatch):
    import torch

    from vllm.model_executor.layers import sparse_attn_indexer as sparse

    scorer = Mock(return_value=torch.zeros((2, 64), dtype=torch.float32))
    decode_module = ModuleType("aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4")
    decode_module.flydsl_pa_mqa_logits_fp4 = scorer
    prefill_module = ModuleType(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    prefill_module.flydsl_pa_mqa_logits_fp4_prefill = Mock()
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
    monkeypatch.setattr(sparse.ops, "top_k_per_row_decode", Mock())

    cta_info = torch.zeros((512, 4), dtype=torch.int32)
    decode = SimpleNamespace(
        requires_padding=False,
        seq_lens=torch.tensor([[64], [64]], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        fp4_cta_info=cta_info,
        fp4_total_ctas=512,
    )
    metadata = SimpleNamespace(
        num_prefills=0, num_decodes=2, num_decode_tokens=2, decode=decode
    )
    cache = torch.zeros((1, 64, 68), dtype=torch.uint8)

    sparse._rocm_fp4_sparse_attn_indexer(
        cache,
        torch.zeros((2, 64, 64), dtype=torch.uint8),
        torch.zeros((2, 1, 4, 16, 4), dtype=torch.uint8),
        torch.zeros((2, 64), dtype=torch.float32),
        128,
        64,
        8,
        torch.empty((2, 8), dtype=torch.int32),
        metadata,
    )

    assert scorer.call_args.kwargs["cta_info"] is cta_info
    assert scorer.call_args.kwargs["total_ctas"] == 512

def test_rocm_fp4_prefill_forwards_planned_width_and_schedule(monkeypatch):
    import torch

    from vllm.model_executor.layers import sparse_attn_indexer as sparse

    scorer = Mock(return_value=torch.zeros((3, 192), dtype=torch.float32))
    decode_module = ModuleType("aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4")
    decode_module.flydsl_pa_mqa_logits_fp4 = Mock()
    prefill_module = ModuleType(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    prefill_module.flydsl_pa_mqa_logits_fp4_prefill = scorer
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
    cta_info = torch.zeros((512, 6), dtype=torch.int32)
    chunk = SimpleNamespace(
        local_total_seq_lens=320,
        token_start=0,
        token_end=3,
        block_table=torch.zeros((2, 4), dtype=torch.int32),
        logits_width=192,
        fp4_windows=(row_to_batch, starts, ends),
        fp4_cta_info=cta_info,
        fp4_total_ctas=512,
    )
    metadata = SimpleNamespace(
        num_prefills=2,
        num_decodes=0,
        num_decode_tokens=0,
        prefill=SimpleNamespace(chunks=[chunk]),
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
    assert scorer.call_args.kwargs["n_ctas"] == 512
    assert topk.call_args.args[2] is ends


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


def _config(dtype=None, *, dcp=1, pcp=1):
    attention = (
        AttentionConfig() if dtype is None else AttentionConfig(indexer_kv_dtype=dtype)
    )
    return SimpleNamespace(
        attention_config=attention,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
    )


def _platform(monkeypatch, *, rocm, capability=100, gfx950=True):
    platform = SimpleNamespace(
        is_rocm=lambda: rocm,
        is_device_capability_family=Mock(
            side_effect=lambda family: capability // 10 == family // 10
        ),
    )
    monkeypatch.setattr(indexer, "current_platform", platform)
    # Avoid importing ROCm driver libraries on CUDA/CPU hosts.
    rocm_module = ModuleType("vllm.platforms.rocm")
    rocm_module.on_gfx950 = Mock(return_value=gfx950)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm_module)
    return platform, rocm_module.on_gfx950


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
    platform.is_device_capability_family.assert_not_called()


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
