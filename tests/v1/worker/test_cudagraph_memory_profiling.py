# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import gc
import weakref
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
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder

    def make_builder():
        builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
        builder._workspace_buffer = None
        builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
        builder.device = torch.device("cpu")
        builder.use_dcp = False
        return builder

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        first_builder = make_builder()
        first_state = first_builder.get_workspace_buffer_state()
        first = first_builder._get_workspace_buffer(
            first_builder._native_initial_workspace_buffer_size()
        )

        second_builder = make_builder()
        second_builder.set_workspace_buffer_state(first_state)
        second = second_builder._get_workspace_buffer(
            second_builder._native_initial_workspace_buffer_size()
        )

        assert first.device.type == "cpu"
        assert first.dtype == torch.uint8
        assert first.numel() == 1
        assert first.data_ptr() == second.data_ptr()
    finally:
        reset_workspace_manager()


def test_flashinfer_workspace_buffer_growth_resets_registered_wrappers():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    class FakeWrapper:
        def __init__(self, float_workspace_buffer):
            self._float_workspace_buffer = float_workspace_buffer
            self._int_workspace_buffer = torch.empty(1, dtype=torch.uint8)
            self.reset_calls = 0

        def reset_workspace_buffer(self, float_workspace_buffer, int_workspace_buffer):
            self._float_workspace_buffer = float_workspace_buffer
            self._int_workspace_buffer = int_workspace_buffer
            self.reset_calls += 1

    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._workspace_buffer = None
    builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
    builder.device = torch.device("cpu")
    builder.use_dcp = False

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        wrapper = FakeWrapper(
            builder._get_workspace_buffer(
                builder._native_initial_workspace_buffer_size()
            )
        )
        builder._register_workspace_wrapper(wrapper)
        builder._ensure_flashinfer_wrapper_workspace(wrapper, 1024)

        assert builder._workspace_buffer.numel() == 1024
        assert wrapper._float_workspace_buffer.data_ptr() == (
            builder._workspace_buffer.data_ptr()
        )
        assert wrapper._float_workspace_buffer.numel() == 1024
        assert wrapper.reset_calls >= 1

        wrapper_ref = weakref.ref(wrapper)
        del wrapper
        gc.collect()

        builder._workspace_state.set_buffer(torch.empty(2048, dtype=torch.uint8))
        assert wrapper_ref() is None
        assert builder._workspace_state.wrappers == []
    finally:
        reset_workspace_manager()


def test_flashinfer_reserves_prefill_tail_workspace(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._workspace_buffer = None
    builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    builder.model_config = SimpleNamespace(max_model_len=1024, dtype=torch.float16)
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_seqs=4,
        )
    )
    builder.q_data_type = torch.float16
    builder.kv_cache_dtype = torch.uint8
    builder.page_size = 16
    builder.window_left = -1
    builder.prefill_fixed_split_size = -1
    builder.disable_split_kv = False

    class FakeWrapper:
        pass

    ensured = []
    observed_query_lens = []

    def fake_workspace_size(**kwargs):
        qo_indptr = kwargs["qo_indptr_cpu"]
        query_lens = torch.diff(qo_indptr).tolist()
        observed_query_lens.extend(query_lens)
        return 4096 if query_lens == [3] else 0

    monkeypatch.setattr(
        builder,
        "_get_prefill_workspace_size_func",
        lambda **kwargs: ("fa2", object()),
    )
    monkeypatch.setattr(builder, "_call_prefill_workspace_size", fake_workspace_size)
    monkeypatch.setattr(
        builder, "_get_prefill_wrapper", lambda causal=True: FakeWrapper()
    )
    monkeypatch.setattr(
        builder,
        "_ensure_flashinfer_wrapper_workspace",
        lambda wrapper, size: ensured.append(size),
    )

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        assert builder.reserve_workspace_for_cudagraph_capture() == 4096
    finally:
        reset_workspace_manager()

    assert ensured == [4096]
    assert 3 in observed_query_lens
    assert 8 in observed_query_lens


def test_flashinfer_workspace_query_len_candidates():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    candidates = (
        flashinfer_backend.FlashInferMetadataBuilder._get_workspace_query_len_candidates
    )

    assert candidates(8) == list(range(1, 9))

    large_candidates = candidates(1024)
    assert 1 in large_candidates
    assert 256 in large_candidates
    assert 512 in large_candidates
    assert 1024 in large_candidates
    assert 257 not in large_candidates


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
                    ),
                    SimpleNamespace(
                        num_tokens=64,
                        uniform=False,
                        num_active_loras=0,
                    ),
                    SimpleNamespace(
                        num_tokens=32,
                        uniform=False,
                        num_active_loras=0,
                    ),
                ],
            ),
            (
                CUDAGraphMode.FULL,
                [
                    SimpleNamespace(
                        num_tokens=80,
                        uniform=False,
                        num_active_loras=0,
                    ),
                    SimpleNamespace(
                        num_tokens=40,
                        uniform=False,
                        num_active_loras=0,
                    ),
                ],
            ),
        ],
        cudagraph_keys={},
        keys_initialized=True,
    )

    warmup_calls = []
    capture_calls = []
    cleanup_calls = []

    runner.max_model_len = 4096
    runner.max_num_tokens = 128
    runner._init_minimal_kv_cache_for_profiling = lambda: None
    runner._requires_separate_cudagraph_memory_profiling = lambda: True
    runner._create_encoder_cudagraph_manager = lambda: None
    runner._freeze_gc = null_context
    runner._cleanup_profiling_kv_cache = lambda: cleanup_calls.append("cleanup")
    runner.maybe_remove_all_loras = lambda lora_config: None
    runner._reserve_attention_workspace_for_cudagraph_capture = lambda: 200
    runner._warmup_before_cudagraph_capture = lambda *args, **kwargs: (
        warmup_calls.append((args[0], kwargs))
    )
    runner._warmup_and_capture = lambda *args, **kwargs: (
        capture_calls.append((args[0], kwargs))
    )

    memory_reserved_values = iter([1_000, 1_600])
    mem_get_info_values = iter(
        [
            (10_000_000, 0),
            (8_000_000, 0),
            (8_000_000, 0),
            (6_500_000, 0),
            (6_500_000, 0),
            (4_100_000, 0),
            (4_100_000, 0),
            (3_000_000, 0),
        ]
    )

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

    assert estimate == 6_500_800
    assert runner.cudagraph_memory_persistent_estimate == 800
    assert runner.cudagraph_memory_graph_pool_estimate == 6_500_000
    assert [call[0].num_tokens for call in warmup_calls] == [128, 64, 80, 40]
    assert [call[0].num_tokens for call in capture_calls] == [128, 64, 80, 40]
    assert all(call[1]["num_warmups"] == 0 for call in capture_calls)
    assert warmup_calls[2][1]["profile_seq_lens"] == 1
    assert warmup_calls[3][1]["profile_seq_lens"] is None
    assert cleanup_calls == ["cleanup"]
