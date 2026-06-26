# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


def _attention_spec(head_size: int, head_size_v: int | None = None):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=head_size,
        head_size_v=head_size_v,
        dtype=torch.float16,
    )


def test_flashinfer_separate_cudagraph_memory_profile_gate():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    assert not FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, _attention_spec(256)
    )
    assert FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, _attention_spec(512)
    )
    assert FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, _attention_spec(256, head_size_v=512)
    )

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer.0": _attention_spec(256),
            "layer.1": _attention_spec(512),
        },
    )
    assert FlashInferMetadataBuilder.requires_separate_cudagraph_memory_profiling(
        None, uniform_spec
    )


def test_flashinfer_workspace_buffer_uses_workspace_manager():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        first_builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
        first_builder._workspace_buffer = None
        first_builder.device = torch.device("cpu")
        first = first_builder._get_workspace_buffer()

        second_builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
        second_builder._workspace_buffer = None
        second_builder.device = torch.device("cpu")
        second = second_builder._get_workspace_buffer()

        assert first.device.type == "cpu"
        assert first.dtype == torch.uint8
        assert first.data_ptr() == second.data_ptr()
    finally:
        reset_workspace_manager()


def test_flashinfer_nvfp4_slot_mapping_symbol_available():
    flashinfer = pytest.importorskip("flashinfer")
    assert hasattr(
        flashinfer,
        "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
    )


def test_separate_profile_accounts_persistent_and_graph_pool(monkeypatch):
    from vllm.v1.worker import gpu_model_runner
    from vllm.v1.worker.gpu_model_runner import CUDAGraphMode, GPUModelRunner

    class FakeWrapper:
        _all_instances = []

        @staticmethod
        def clear_all_graphs():
            pass

    @contextlib.contextmanager
    def null_context(*args, **kwargs):
        yield

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.lora_config = None
    runner.cudagraph_dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [
            (
                CUDAGraphMode.PIECEWISE,
                [
                    SimpleNamespace(
                        num_tokens=128,
                        uniform=False,
                        num_active_loras=0,
                    )
                ],
            )
        ],
        cudagraph_keys={},
        keys_initialized=True,
    )

    warmup_calls = []
    capture_calls = []
    cleanup_calls = []

    runner._init_minimal_kv_cache_for_profiling = lambda: None
    runner._requires_separate_cudagraph_memory_profiling = lambda: True
    runner._create_encoder_cudagraph_manager = lambda: None
    runner._freeze_gc = null_context
    runner._cleanup_profiling_kv_cache = lambda: cleanup_calls.append("cleanup")
    runner.maybe_remove_all_loras = lambda lora_config: None
    runner._warmup_before_cudagraph_capture = (
        lambda *args, **kwargs: warmup_calls.append(kwargs)
    )
    runner._warmup_and_capture = lambda *args, **kwargs: capture_calls.append(kwargs)

    memory_reserved_values = iter([1_000, 1_600])
    mem_get_info_values = iter([(5_000, 0), (4_300, 0)])

    monkeypatch.setattr(gpu_model_runner, "CUDAGraphWrapper", FakeWrapper)
    monkeypatch.setattr(gpu_model_runner, "BreakableCUDAGraphWrapper", FakeWrapper)
    monkeypatch.setattr(
        gpu_model_runner,
        "set_current_vllm_config",
        lambda *args, **kwargs: null_context(),
    )
    monkeypatch.setattr(
        gpu_model_runner, "graph_capture", lambda *args, **kwargs: null_context()
    )
    monkeypatch.setattr(
        gpu_model_runner,
        "set_cudagraph_capturing_enabled",
        lambda enabled: None,
    )
    monkeypatch.setattr(
        gpu_model_runner.current_platform,
        "graph_pool_handle",
        lambda: object(),
    )
    monkeypatch.setattr(gpu_model_runner.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(gpu_model_runner.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        gpu_model_runner.torch.accelerator,
        "memory_reserved",
        lambda device: next(memory_reserved_values),
    )
    monkeypatch.setattr(
        gpu_model_runner.torch.cuda,
        "mem_get_info",
        lambda: next(mem_get_info_values),
    )

    estimate = runner.profile_cudagraph_memory()

    assert estimate == 1_300
    assert runner.cudagraph_memory_persistent_estimate == 600
    assert runner.cudagraph_memory_graph_pool_estimate == 700
    assert len(warmup_calls) == 1
    assert len(capture_calls) == 1
    assert capture_calls[0]["num_warmups"] == 0
    assert cleanup_calls == ["cleanup"]
