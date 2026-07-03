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


class _FakeFlashInferWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor | None = None,
        int_workspace_bytes: int = 1,
    ) -> None:
        self._float_workspace_buffer = (
            float_workspace_buffer
            if float_workspace_buffer is not None
            else torch.empty(1, dtype=torch.uint8)
        )
        self._int_workspace_buffer = torch.empty(
            max(int_workspace_bytes, 1), dtype=torch.uint8
        )
        self._vllm_flashinfer_int_workspace_finalized = False
        self.is_cuda_graph_enabled = False
        self.reset_calls = 0

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self.reset_calls += 1


class _FakeWorkspaceSizeArray:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


def _make_flashinfer_builder(flashinfer_backend):
    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._workspace_buffer = None
    builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    return builder


def test_flashinfer_workspace_size_parses_sequence_like_array():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes

    assert flashinfer_backend._parse_workspace_sizes(
        _FakeWorkspaceSizeArray([1024, 64])
    ) == WorkspaceSizes(1024, 64, True)


def test_flashinfer_workspace_size_float_only_keeps_default_int_workspace():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    builder = _make_flashinfer_builder(flashinfer_backend)
    wrapper = _FakeFlashInferWrapper(int_workspace_bytes=64)
    sizes = flashinfer_backend._parse_workspace_sizes(_FakeWorkspaceSizeArray([1024]))

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        builder._ensure_flashinfer_wrapper_workspace(wrapper, sizes)
    finally:
        reset_workspace_manager()

    assert wrapper._float_workspace_buffer.numel() == 1024
    assert wrapper._int_workspace_buffer.numel() == 64


def test_flashinfer_workspace_size_explicit_int_shrinks_default_workspace():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    mib = 1 << 20
    builder = _make_flashinfer_builder(flashinfer_backend)
    wrapper = _FakeFlashInferWrapper(int_workspace_bytes=8 * mib)

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        builder._ensure_flashinfer_wrapper_workspace(
            wrapper, WorkspaceSizes(1024, 64 * 1024, True)
        )
    finally:
        reset_workspace_manager()

    assert wrapper._float_workspace_buffer.numel() == 1024
    assert wrapper._int_workspace_buffer.numel() == mib


def test_flashinfer_workspace_size_rejects_invalid_sequence_length():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    with pytest.raises(ValueError, match="workspace_size"):
        flashinfer_backend._parse_workspace_sizes(_FakeWorkspaceSizeArray([1, 2, 3]))


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

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        first_builder = _make_flashinfer_builder(flashinfer_backend)
        first_state = first_builder.get_workspace_buffer_state()
        first = first_builder._get_workspace_buffer(
            first_builder._native_initial_workspace_buffer_size()
        )

        second_builder = _make_flashinfer_builder(flashinfer_backend)
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

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        wrapper = _FakeFlashInferWrapper(
            builder._get_workspace_buffer(
                builder._native_initial_workspace_buffer_size()
            )
        )
        builder._register_workspace_wrapper(wrapper)
        builder._ensure_flashinfer_wrapper_workspace(
            wrapper, WorkspaceSizes(1024, 16, True)
        )

        assert builder._workspace_buffer.numel() == 1024
        assert wrapper._float_workspace_buffer.data_ptr() == (
            builder._workspace_buffer.data_ptr()
        )
        assert wrapper._float_workspace_buffer.numel() == 1024
        assert wrapper._int_workspace_buffer.numel() == 1 << 20
        reset_calls = wrapper.reset_calls
        assert reset_calls >= 1

        builder._ensure_flashinfer_wrapper_workspace(
            wrapper, WorkspaceSizes(1024, 16, True)
        )
        assert wrapper.reset_calls == reset_calls

        wrapper_ref = weakref.ref(wrapper)
        del wrapper
        gc.collect()

        builder._workspace_state.set_buffer(torch.empty(2048, dtype=torch.uint8))
        assert wrapper_ref() is None
        assert builder._workspace_state.wrappers == []
    finally:
        reset_workspace_manager()


def test_flashinfer_int_workspace_is_per_wrapper():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        first = _FakeFlashInferWrapper()
        second = _FakeFlashInferWrapper()

        builder._ensure_flashinfer_wrapper_workspace(
            first, WorkspaceSizes(1024, 32, True)
        )
        builder._ensure_flashinfer_wrapper_workspace(
            second, WorkspaceSizes(1024, 32, True)
        )

        assert first._float_workspace_buffer.data_ptr() == (
            second._float_workspace_buffer.data_ptr()
        )
        assert first._int_workspace_buffer.data_ptr() != (
            second._int_workspace_buffer.data_ptr()
        )
        assert first._int_workspace_buffer.numel() == 1 << 20
        assert second._int_workspace_buffer.numel() == 1 << 20
    finally:
        reset_workspace_manager()


def test_flashinfer_finalized_int_workspace_cannot_grow():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)
    wrapper = _FakeFlashInferWrapper(int_workspace_bytes=8)
    wrapper._vllm_flashinfer_int_workspace_finalized = True

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        with pytest.raises(AssertionError, match="int workspace is finalized"):
            builder._ensure_flashinfer_wrapper_workspace(
                wrapper, WorkspaceSizes(1024, 16, True)
            )
    finally:
        reset_workspace_manager()


def test_flashinfer_non_cudagraph_int_workspace_can_grow(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)
    wrapper = _FakeFlashInferWrapper(int_workspace_bytes=8)
    warnings = []

    monkeypatch.setattr(
        flashinfer_backend.logger,
        "warning",
        lambda msg, *args: warnings.append(msg),
    )

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        builder._ensure_flashinfer_wrapper_workspace(
            wrapper, WorkspaceSizes(1024, 8, True)
        )
        builder._ensure_flashinfer_wrapper_workspace(
            wrapper, WorkspaceSizes(1024, 2 << 20, True)
        )

        assert wrapper._int_workspace_buffer.numel() == 2 << 20
        assert wrapper.reset_calls == 2
        assert any("Growing FlashInfer int workspace" in msg for msg in warnings)
    finally:
        reset_workspace_manager()


def test_flashinfer_reserves_prefill_tail_workspace(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
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
        ),
        speculative_config=None,
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
        return (
            WorkspaceSizes(4096, 64, True)
            if query_lens == [3]
            else WorkspaceSizes(0, 0)
        )

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
    monkeypatch.setattr(
        builder,
        "_reserve_decode_wrapper_workspace",
        lambda **kwargs: WorkspaceSizes(0, 0),
    )
    builder.enable_cuda_graph = False

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        assert builder.reserve_workspace_for_cudagraph_capture() == 4160
    finally:
        reset_workspace_manager()

    assert ensured == [WorkspaceSizes(4096, 64, True)]
    assert 3 in observed_query_lens
    assert 8 in observed_query_lens


def test_flashinfer_reserves_decode_cudagraph_int_workspace(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)
    builder.decode_fixed_split_size = -1
    builder.disable_split_kv = False

    wrapper = _FakeFlashInferWrapper()
    wrapper.is_cuda_graph_enabled = True

    monkeypatch.setattr(builder, "_get_decode_wrapper", lambda *args: wrapper)
    monkeypatch.setattr(
        builder,
        "_get_decode_workspace_size",
        lambda **kwargs: WorkspaceSizes(128, 32, True),
    )

    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    try:
        sizes = builder._reserve_decode_wrapper_workspace(
            batch_size=4,
            num_pages=8,
            last_page_len=16,
            use_cudagraph=True,
        )
    finally:
        reset_workspace_manager()

    assert sizes == WorkspaceSizes(128, 32, True)
    assert wrapper._float_workspace_buffer.numel() == 128
    assert wrapper._int_workspace_buffer.numel() == 1 << 20
    assert wrapper.reset_calls == 1
    assert wrapper._vllm_flashinfer_int_workspace_finalized


def test_flashinfer_workspace_debug_info_reports_retained_default_int_workspace():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    WorkspaceSizes = flashinfer_backend.WorkspaceSizes
    builder = _make_flashinfer_builder(flashinfer_backend)
    mib = 1 << 20
    builder._prefill_wrapper = _FakeFlashInferWrapper(int_workspace_bytes=8 * mib)
    builder._decode_wrapper = _FakeFlashInferWrapper(int_workspace_bytes=4 * mib)
    builder._decode_wrappers_cudagraph = {
        1: _FakeFlashInferWrapper(int_workspace_bytes=8 * mib),
        2: _FakeFlashInferWrapper(int_workspace_bytes=8 * mib),
    }
    builder._last_reserved_workspace_sizes = WorkspaceSizes(
        float_bytes=128 * mib,
        int_bytes=256 * 1024,
    )

    info = builder.get_workspace_reserve_debug_info()

    assert info["workspace_wrapper_count"] == 4
    assert info["prefill_wrappers"] == 1
    assert info["decode_wrappers"] == 1
    assert info["decode_cudagraph_wrappers"] == 2
    assert info["actual_int_workspace_bytes"] == 28 * mib
    assert info["reserved_int_workspace_bytes"] == 256 * 1024
    assert info["int_workspace_over_reserved_bytes"] == 28 * mib - 256 * 1024
    assert info["default_int_workspace_wrappers"] == 3
    assert info["unique_int_workspace_buffers"] == 4


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
    get_memory_info_values = iter(
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
        gpu_model_runner.torch.accelerator,
        "get_memory_info",
        lambda: next(get_memory_info_values),
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


def test_v2_profile_accounts_attention_workspace(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    @contextlib.contextmanager
    def null_context(*args, **kwargs):
        yield

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()

    events = []

    monkeypatch.setattr(
        gpu_model_runner_v2,
        "set_current_vllm_config",
        lambda *args, **kwargs: null_context(),
    )

    def reserve_attention_workspace():
        events.append("reserve")
        return 4096

    runner._init_minimal_kv_cache_for_profiling = lambda: events.append("init")
    runner._reserve_attention_workspace_for_cudagraph_capture = (
        reserve_attention_workspace
    )
    runner._cleanup_profiling_kv_cache = lambda: events.append("cleanup")

    estimate = runner.profile_cudagraph_memory()

    assert estimate == 4096
    assert runner.cudagraph_memory_persistent_estimate == 4096
    assert runner.cudagraph_memory_graph_pool_estimate == 0
    assert events == ["init", "reserve", "cleanup"]


def test_v2_cleanup_profiling_kv_cache_releases_builder_refs(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    class Builder:
        pass

    builder = Builder()
    builder_ref = weakref.ref(builder)
    layer = SimpleNamespace(
        kv_cache=torch.empty(1),
        impl=SimpleNamespace(_k_scale_cache=object(), _v_scale_cache=object()),
    )

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=4)
    runner.kv_caches = [torch.empty(1)]
    runner.attn_groups = [[SimpleNamespace(metadata_builders=[builder])]]
    runner.kv_cache_config = object()
    runner.block_tables = object()
    runner.kernel_block_sizes = [16]
    runner.cudagraph_manager = object()
    runner.compilation_config = SimpleNamespace(static_forward_context={"layer": layer})
    del builder

    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "empty_cache", lambda: None
    )

    runner._cleanup_profiling_kv_cache()
    gc.collect()

    assert runner.kv_caches == []
    assert not hasattr(runner, "attn_groups")
    assert not hasattr(runner, "kv_cache_config")
    assert not hasattr(runner, "block_tables")
    assert not hasattr(runner, "kernel_block_sizes")
    assert not hasattr(runner, "cudagraph_manager")
    assert runner.cache_config.num_gpu_blocks is None
    assert isinstance(layer.kv_cache, torch.Tensor)
    assert layer.kv_cache.numel() == 0
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    assert builder_ref() is None


def test_v2_capture_reserves_workspace_before_measurement_and_locks(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    @contextlib.contextmanager
    def null_context(*args, **kwargs):
        yield

    class Builder:
        reserved = False

        def reserve_workspace_for_cudagraph_capture(self):
            events.append("builder_reserve")
            self.reserved = True
            return 128

    class FakeCudaGraphManager:
        def needs_capture(self):
            return True

        def capture(
            self,
            model,
            model_state,
            input_buffers,
            intermediate_tensors,
            block_tables,
            attn_groups,
            kv_cache_config,
            **kwargs,
        ):
            events.append("capture")
            assert attn_groups[0][0].metadata_builders[0].reserved
            return {}

    events = []
    builder = Builder()
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.cudagraph_manager = FakeCudaGraphManager()
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda lora_config: null_context()
    runner.model = object()
    runner.model_state = object()
    runner.input_buffers = object()
    runner.intermediate_tensors = None
    runner.block_tables = object()
    runner.attn_groups = [[SimpleNamespace(metadata_builders=[builder])]]
    runner.kv_cache_config = object()
    runner.use_aux_hidden_state_outputs = False
    runner.speculator = None

    memory_reserved_values = iter([1_000, 1_000, 1_128, 1_128])
    memory_allocated_values = iter([500, 500, 628, 628])
    get_memory_info_values = iter([(10_000, 0), (9_000, 0)])

    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "empty_cache", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "memory_reserved",
        lambda device: next(memory_reserved_values),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "memory_allocated",
        lambda device: next(memory_allocated_values),
    )

    def get_memory_info():
        events.append("memory_info")
        return next(get_memory_info_values)

    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "get_memory_info",
        get_memory_info,
    )
    monkeypatch.setattr(
        gpu_model_runner_v2,
        "lock_workspace",
        lambda: events.append("lock"),
    )

    assert runner.capture_model() == 1_000
    assert events == [
        "builder_reserve",
        "memory_info",
        "capture",
        "memory_info",
        "lock",
    ]


def test_v2_attention_workspace_reserve_logs_breakdown(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    mib = 1 << 20
    workspace_buffer = torch.empty(1)

    class Builder:
        def __init__(self, requested_bytes, debug_info):
            self.requested_bytes = requested_bytes
            self.debug_info = debug_info

        def reserve_workspace_for_cudagraph_capture(self):
            return self.requested_bytes

        def get_workspace_buffer_state(self):
            return SimpleNamespace(buffer=workspace_buffer)

        def get_workspace_reserve_debug_info(self):
            return self.debug_info

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.attn_groups = [
        [
            SimpleNamespace(
                metadata_builders=[
                    Builder(
                        64 * mib,
                        {
                            "workspace_wrapper_count": 4,
                            "decode_cudagraph_wrappers": 2,
                            "default_int_workspace_wrappers": 3,
                            "actual_int_workspace_bytes": 24 * mib,
                            "reserved_int_workspace_bytes": 1 * mib,
                            "int_workspace_over_reserved_bytes": 23 * mib,
                            "unique_int_workspace_buffers": 4,
                            "unique_float_workspace_buffers": 1,
                            "workspace_state_live_wrappers": 4,
                        },
                    ),
                    Builder(
                        32 * mib,
                        {
                            "workspace_wrapper_count": 2,
                            "decode_cudagraph_wrappers": 1,
                            "default_int_workspace_wrappers": 1,
                            "actual_int_workspace_bytes": 8 * mib,
                            "reserved_int_workspace_bytes": 1 * mib,
                            "int_workspace_over_reserved_bytes": 7 * mib,
                            "unique_int_workspace_buffers": 2,
                            "unique_float_workspace_buffers": 1,
                            "workspace_state_live_wrappers": 2,
                        },
                    ),
                ]
            )
        ]
    ]

    memory_reserved_values = iter(
        [1000 * mib, 1000 * mib, 1064 * mib, 1064 * mib, 1160 * mib, 1160 * mib]
    )
    memory_allocated_values = iter(
        [500 * mib, 500 * mib, 564 * mib, 564 * mib, 596 * mib, 596 * mib]
    )
    debug_logs = []

    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "memory_reserved",
        lambda device: next(memory_reserved_values),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator,
        "memory_allocated",
        lambda device: next(memory_allocated_values),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.torch.accelerator, "empty_cache", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_v2.logger,
        "debug",
        lambda msg, *args: debug_logs.append(msg % args if args else msg),
    )

    assert runner._reserve_attention_workspace_for_cudagraph_capture() == 160 * mib

    assert any("96.00 MiB requested by builders" in log for log in debug_logs)
    assert any("64.00 MiB unexplained" in log for log in debug_logs)
    assert any("1 unique workspace buffers" in log for log in debug_logs)
    assert any("6 wrappers" in log for log in debug_logs)
    assert any("32.00 MiB actual int workspace" in log for log in debug_logs)
    assert any("2.00 MiB requested int workspace" in log for log in debug_logs)
    assert any("30.00 MiB int workspace over request" in log for log in debug_logs)
    assert any("default_int_workspaces=3" in log for log in debug_logs)
    assert (
        sum(
            "Reserved attention workspace builder=Builder requested=" in log
            for log in debug_logs
        )
        == 2
    )
