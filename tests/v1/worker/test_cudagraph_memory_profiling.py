# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.worker.utils as worker_utils
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
    use_workspace_ubatch_id,
)


@contextlib.contextmanager
def _managed_workspace(num_ubatches=1):
    """A workspace manager that exists only for the duration of one test."""
    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"), num_ubatches=num_ubatches)
    try:
        yield
    finally:
        reset_workspace_manager()


@contextlib.contextmanager
def _null_context(*args, **kwargs):
    yield


def _attention_spec(
    head_size: int,
    head_size_v: int | None = None,
    *,
    non_causal: bool = False,
):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=head_size,
        head_size_v=head_size_v,
        dtype=torch.float16,
        non_causal=non_causal,
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


def _make_flashinfer_builder(flashinfer_backend):
    FlashInferMetadataBuilder = flashinfer_backend.FlashInferMetadataBuilder
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._workspace_buffer = None
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    builder.use_trtllm_decode_attention = False
    builder.max_num_batched_tokens = 1
    builder.num_qo_heads = 1
    builder.head_dim = 1
    return builder


def test_workspace_manager_reserves_and_locks_every_ubatch_slot():
    with _managed_workspace(num_ubatches=3):
        pointers = []
        for ubatch_id, size in enumerate((1024, 2048, 3072)):
            with use_workspace_ubatch_id(ubatch_id):
                (workspace,) = current_workspace_manager().get_simultaneous(
                    ((size,), torch.uint8)
                )
                pointers.append(workspace.data_ptr())

        assert current_workspace_manager().workspace_sizes_bytes() == (
            1024,
            2048,
            3072,
        )
        assert len(set(pointers)) == 3

        with use_workspace_ubatch_id(0):
            (reused,) = current_workspace_manager().get_simultaneous(
                ((512,), torch.uint8)
            )
        assert reused.data_ptr() == pointers[0]
        assert current_workspace_manager().workspace_sizes_bytes()[0] == 1024

        lock_workspace()
        with use_workspace_ubatch_id(1):
            current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
            with pytest.raises(AssertionError, match="Workspace is locked"):
                current_workspace_manager().get_simultaneous(((4096,), torch.uint8))


def test_attention_group_routes_builder_initialization_to_ubatch_slots():
    from vllm.v1.worker.utils import AttentionGroup

    class Builder:
        requires_block_table_width = False

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            size = (len(created_builders) + 1) * 1024
            (self.workspace,) = current_workspace_manager().get_simultaneous(
                ((size,), torch.uint8)
            )
            created_builders.append(self)

    class Backend:
        @staticmethod
        def get_builder_cls():
            return Builder

    created_builders: list[Builder] = []
    group = AttentionGroup(Backend, ["layer"], object(), 0)
    with _managed_workspace(num_ubatches=3):
        group.create_metadata_builders(
            None, torch.device("cpu"), num_metadata_builders=3
        )
        assert current_workspace_manager().workspace_sizes_bytes() == (
            1024,
            2048,
            3072,
        )
        assert len({builder.workspace.data_ptr() for builder in created_builders}) == 3


def test_flashinfer_default_workspace_covers_prefill_head_footprint(monkeypatch):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    builder = _make_flashinfer_builder(flashinfer_backend)
    builder.max_num_batched_tokens = 8
    builder.num_qo_heads = 4
    builder.head_dim = 16
    estimated_prefill_size = (
        builder.max_num_batched_tokens
        * builder.num_qo_heads
        * builder.head_dim
        * flashinfer_backend.FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM
    )

    monkeypatch.setattr(flashinfer_backend.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(
        flashinfer_backend.envs,
        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
        1,
    )
    assert builder._default_workspace_buffer_size() == estimated_prefill_size

    configured_size = estimated_prefill_size + 1
    monkeypatch.setattr(
        flashinfer_backend.envs,
        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
        configured_size,
    )
    assert builder._default_workspace_buffer_size() == configured_size


def _nbytes(buffer: torch.Tensor | None) -> int:
    return 0 if buffer is None else buffer.numel() * buffer.element_size()


def _reservation_builder(
    flashinfer_backend,
    *,
    default_float_bytes,
    builder_cls=None,
    is_mm_prefix_lm=False,
    use_trtllm_prefill_attention=False,
    # Decode defaults to trtllm so a reservation under test stays on the
    # native prefill leg unless a test asks for the decode one.
    use_trtllm_decode_attention=True,
    max_num_batched_tokens=8,
    max_num_seqs=4,
    max_model_len=1024,
    cudagraph_capture_sizes=None,
    decode_cudagraph_max_bs=0,
):
    """Builder wired for the reservation legs on CPU."""
    builder_cls = builder_cls or flashinfer_backend.FlashInferMetadataBuilder
    builder = builder_cls.__new__(builder_cls)
    builder._workspace_buffer = None
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    builder.use_xqa = False
    builder._reservation_trtllm_prefill = use_trtllm_prefill_attention
    builder.use_trtllm_decode_attention = use_trtllm_decode_attention
    builder.enable_cuda_graph = cudagraph_capture_sizes is not None
    builder.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=cudagraph_capture_sizes
    )
    builder._decode_cudagraph_max_bs = decode_cudagraph_max_bs
    builder.kv_cache_spec = _attention_spec(128)
    builder.model_config = SimpleNamespace(
        dtype=torch.float16,
        is_mm_prefix_lm=is_mm_prefix_lm,
        max_model_len=max_model_len,
    )
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
    )
    builder._prefill_wrapper = None
    builder._noncausal_prefill_wrapper = None
    builder._decode_wrapper = None
    builder._decode_wrappers_cudagraph = {}
    builder._cascade_wrapper = None
    builder._default_workspace_buffer_size = lambda: default_float_bytes
    return builder


def _install_prefill_factories(builder, monkeypatch):
    causal_wrapper = _FakeFlashInferWrapper(int_workspace_bytes=64)

    def get_prefill_wrapper(causal=True):
        assert causal
        if builder._prefill_wrapper is None:
            causal_wrapper._float_workspace_buffer = builder._get_workspace_buffer()
            builder._prefill_wrapper = causal_wrapper
        return builder._prefill_wrapper

    monkeypatch.setattr(builder, "_get_prefill_wrapper", get_prefill_wrapper)
    return causal_wrapper


@pytest.mark.parametrize(
    ("decode_context_parallel_size", "is_mm_prefix_lm", "expected"),
    [
        pytest.param(1, False, True, id="single-rank"),
        pytest.param(2, False, False, id="dcp-fallback"),
        pytest.param(1, True, True, id="mm-prefix-profiled"),
    ],
)
def test_flashinfer_persistent_workspace_profile_gate(
    decode_context_parallel_size, is_mm_prefix_lm, expected
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=is_mm_prefix_lm),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=decode_context_parallel_size
        ),
    )

    assert FlashInferMetadataBuilder.persistent_workspace_profiling_support(
        config, _attention_spec(128)
    ) is (True if expected else None)


def test_flashinfer_mm_prefix_keeps_the_conservative_arena(monkeypatch):
    """The model flag alone does not close the lifecycle.

    What the arena has to survive is a wrapper built after the lock asking
    for it with no argument. The reservation settles it at the default size,
    which is exactly that request, so such a wrapper finds the arena rather
    than growing a locked one.
    """
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    default_float_bytes = 4096
    builder = _reservation_builder(
        flashinfer_backend,
        default_float_bytes=default_float_bytes,
        is_mm_prefix_lm=True,
    )
    builder._reservation_trtllm_prefill = True
    assert builder._get_workspace_routes().native_prefill

    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=True),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    assert (
        flashinfer_backend.FlashInferMetadataBuilder.persistent_workspace_profiling_support(
            config, _attention_spec(128)
        )
        is True
    )

    _install_prefill_factories(builder, monkeypatch)
    monkeypatch.setattr(
        flashinfer_backend, "_get_trtllm_workspace_buffer", lambda: None
    )

    with _managed_workspace():
        builder.prepare_workspace_for_profiling(False)
        builder.prepare_workspace_for_profiling(True)
        arena = builder._workspace_buffer
        size_before, pointer_before = _nbytes(arena), arena.data_ptr()
        assert size_before == default_float_bytes

        lock_workspace()
        # What a wrapper built after the lock asks for, with no argument.
        late = builder._get_workspace_buffer()

        assert _nbytes(late) == size_before
        assert late.data_ptr() == pointer_before


@pytest.mark.parametrize(
    "kv_cache_spec",
    [
        pytest.param(_attention_spec(128, non_causal=True), id="single-spec"),
        pytest.param(
            UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={
                    "layer.0": _attention_spec(128),
                    "layer.1": _attention_spec(128, non_causal=True),
                },
            ),
            id="uniform-spec",
        ),
    ],
)
def test_flashinfer_persistent_workspace_profile_rejects_non_causal(
    kv_cache_spec,
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=False),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    assert (
        FlashInferMetadataBuilder.persistent_workspace_profiling_support(
            config, kv_cache_spec
        )
        is None
    )


@pytest.mark.parametrize(
    ("page_size", "configured_value", "expected_force"),
    [
        pytest.param(16, None, None, id="configured-dispatch"),
        pytest.param(128, False, True, id="large-page-forced"),
    ],
)
def test_flashinfer_prefill_reservation_uses_runtime_dispatch_contract(
    monkeypatch,
    page_size,
    configured_value,
    expected_force,
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    builder = flashinfer_backend.FlashInferMetadataBuilder.__new__(
        flashinfer_backend.FlashInferMetadataBuilder
    )
    builder.page_size = page_size
    builder.attention_config = SimpleNamespace(use_trtllm_attention=configured_value)
    builder.num_qo_heads = 32
    builder.num_kv_heads = 8
    builder.dcp_world_size = 1
    builder.cache_dtype = "auto"
    builder.q_data_type_prefill = torch.float16
    builder.has_sinks = True
    builder.reorder_batch_threshold = 2
    builder.max_num_batched_tokens = 17
    builder.model_config = SimpleNamespace(max_model_len=4096)

    calls = []

    def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(flashinfer_backend, "use_trtllm_attention", dispatch)

    assert builder._resolve_trtllm_prefill_attention()
    assert calls == [
        (
            (32, 8, 17, 4096, 1, "auto", torch.float16),
            {
                "is_prefill": True,
                "force_use_trtllm": expected_force,
                "has_sinks": True,
                "has_spec": True,
            },
        )
    ]


@pytest.mark.parametrize(
    (
        "trtllm_prefill",
        "trtllm_decode",
        "non_causal",
        "is_mm_prefix_lm",
        "use_xqa",
        "expected",
    ),
    [
        pytest.param(
            False,
            False,
            False,
            False,
            False,
            (True, False, True, False),
            id="all-native",
        ),
        pytest.param(
            True,
            True,
            False,
            False,
            False,
            (False, True, False, True),
            id="all-trtllm",
        ),
        pytest.param(
            True,
            True,
            False,
            True,
            False,
            (True, True, False, True),
            id="mm-prefix-native-and-trtllm",
        ),
        pytest.param(
            True,
            True,
            True,
            False,
            False,
            (True, False, False, False),
            id="non-causal-native-only",
        ),
        pytest.param(
            True,
            True,
            True,
            False,
            True,
            (True, False, False, True),
            id="non-causal-dedicated-xqa-decode",
        ),
    ],
)
def test_flashinfer_workspace_routes_match_reachable_dispatches(
    trtllm_prefill,
    trtllm_decode,
    non_causal,
    is_mm_prefix_lm,
    use_xqa,
    expected,
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._reservation_trtllm_prefill = trtllm_prefill
    builder.use_trtllm_decode_attention = trtllm_decode
    builder.use_xqa = use_xqa
    builder.kv_cache_spec = SimpleNamespace(non_causal=non_causal)
    builder.model_config = SimpleNamespace(is_mm_prefix_lm=is_mm_prefix_lm)

    assert builder._get_workspace_routes() == expected


def test_worker_persistent_workspace_gate_falls_back_for_flashinfer_dcp(
    monkeypatch,
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    from vllm.v1.worker.utils import (
        requires_persistent_attention_workspace_profiling,
    )

    class Backend:
        @staticmethod
        def get_builder_cls():
            return FlashInferMetadataBuilder

    class Layer:
        @staticmethod
        def get_kv_cache_spec(config):
            return _attention_spec(128)

        @staticmethod
        def get_attn_backend():
            return Backend

    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            enable_elastic_ep=False,
        ),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.utils.get_layers_from_vllm_config",
        lambda config, layer_type: {"layer": Layer()},
    )

    assert not requires_persistent_attention_workspace_profiling(config)


def test_worker_persistent_workspace_gate_allows_flashinfer_with_gdn(
    monkeypatch,
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.worker.utils import (
        requires_persistent_attention_workspace_profiling,
    )

    class Backend:
        def __init__(self, builder_cls):
            self.builder_cls = builder_cls

        def get_builder_cls(self):
            return self.builder_cls

    class Layer:
        def __init__(self, builder_cls):
            self.backend = Backend(builder_cls)

        @staticmethod
        def get_kv_cache_spec(config):
            return _attention_spec(128)

        def get_attn_backend(self):
            return self.backend

    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
        ),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.utils.get_layers_from_vllm_config",
        lambda config, layer_type: {
            "full-attention": Layer(FlashInferMetadataBuilder),
            "gdn": Layer(GDNAttentionMetadataBuilder),
        },
    )

    assert requires_persistent_attention_workspace_profiling(config)


def test_flashinfer_workspace_buffer_uses_workspace_manager():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    with _managed_workspace():
        first_builder = _make_flashinfer_builder(flashinfer_backend)
        first = first_builder._get_workspace_buffer()

        second_builder = _make_flashinfer_builder(flashinfer_backend)
        second = second_builder._get_workspace_buffer()

        assert first.device.type == "cpu"
        assert first.dtype == torch.uint8
        assert first.numel() == first_builder._default_workspace_buffer_size()
        assert first.data_ptr() == second.data_ptr()


@pytest.mark.parametrize("use_trtllm_prefill_attention", [False, True])
@pytest.mark.parametrize("use_trtllm_decode_attention", [False, True])
def test_flashinfer_memory_profile_materializes_active_route_wrappers(
    use_trtllm_prefill_attention,
    use_trtllm_decode_attention,
    monkeypatch,
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    class RouteBuilder(  # type: ignore[misc]
        flashinfer_backend.FlashInferMetadataBuilder
    ):
        _prefill_wrapper: _FakeFlashInferWrapper | None
        _decode_wrapper: _FakeFlashInferWrapper | None
        _decode_wrappers_cudagraph: dict[int, _FakeFlashInferWrapper]
        calls: list[tuple[str, int | None]]

        def _get_prefill_wrapper(self, causal=True):
            assert causal
            if self._prefill_wrapper is None:
                self.calls.append(("prefill", None))
                self._prefill_wrapper = _FakeFlashInferWrapper(
                    self._get_workspace_buffer(), int_workspace_bytes=64
                )
            return self._prefill_wrapper

        def _get_decode_wrapper(self, batch_size, use_cudagraph=False):
            self.calls.append(("decode_cg" if use_cudagraph else "decode", batch_size))
            if use_cudagraph:
                wrapper = self._decode_wrappers_cudagraph.get(batch_size)
            else:
                wrapper = self._decode_wrapper
            if wrapper is None:
                wrapper = _FakeFlashInferWrapper(
                    self._get_workspace_buffer(),
                    int_workspace_bytes=128 if use_cudagraph else 96,
                )
                if use_cudagraph:
                    self._decode_wrappers_cudagraph[batch_size] = wrapper
                else:
                    self._decode_wrapper = wrapper
            return wrapper

    builder = _reservation_builder(
        flashinfer_backend,
        default_float_bytes=4096,
        builder_cls=RouteBuilder,
        use_trtllm_prefill_attention=use_trtllm_prefill_attention,
        use_trtllm_decode_attention=use_trtllm_decode_attention,
        max_num_batched_tokens=4,
        max_num_seqs=3,
        max_model_len=16,
        cudagraph_capture_sizes=[0, 2, 4, 8],
        decode_cudagraph_max_bs=4,
    )
    builder.calls = []

    trtllm_workspace = torch.empty(512, dtype=torch.uint8)
    trtllm_workspace_calls: list[None] = []

    def get_trtllm_workspace_buffer():
        trtllm_workspace_calls.append(None)
        return trtllm_workspace

    monkeypatch.setattr(
        flashinfer_backend,
        "_get_trtllm_workspace_buffer",
        get_trtllm_workspace_buffer,
    )

    with _managed_workspace():
        builder.prepare_workspace_for_profiling(False)
        builder.prepare_workspace_for_profiling(True)

        wrappers: list[_FakeFlashInferWrapper] = []
        if not use_trtllm_prefill_attention:
            assert builder._prefill_wrapper is not None
            wrappers.append(builder._prefill_wrapper)
        if not use_trtllm_decode_attention:
            assert builder._decode_wrapper is not None
            wrappers.extend(
                [
                    builder._decode_wrapper,
                    *builder._decode_wrappers_cudagraph.values(),
                ]
            )
        arena_bytes = current_workspace_manager().workspace_sizes_bytes()
        use_native_prefill = not use_trtllm_prefill_attention
        use_native_decode = not use_trtllm_decode_attention
        if use_native_prefill or use_native_decode:
            assert arena_bytes[0] > 0
        else:
            assert arena_bytes[0] == 0

        expected_calls: list[tuple[str, int | None]] = []
        if use_native_prefill:
            expected_calls.append(("prefill", None))
        if not use_trtllm_decode_attention:
            expected_calls.extend(
                [
                    ("decode", 3),
                    ("decode_cg", 2),
                    ("decode_cg", 4),
                ]
            )
        assert builder.calls == expected_calls
        assert len(trtllm_workspace_calls) == int(
            use_trtllm_prefill_attention or use_trtllm_decode_attention
        )
        if wrappers:
            # Every wrapper was handed the one final arena, and none of them
            # can have been rebound afterwards.
            assert all(
                wrapper._float_workspace_buffer.data_ptr()
                == builder._workspace_buffer.data_ptr()
                for wrapper in wrappers
            )
        assert len(
            {wrapper._int_workspace_buffer.data_ptr() for wrapper in wrappers}
        ) == len(wrappers)

        wrapper_refs = [weakref.ref(wrapper) for wrapper in wrappers]
        lease = [builder]
        del wrappers
        del builder
        gc.collect()
        assert all(wrapper_ref() is not None for wrapper_ref in wrapper_refs)

        lease.clear()
        gc.collect()
        assert all(wrapper_ref() is None for wrapper_ref in wrapper_refs)


def _load_gpu_model_runner(version: str):
    if version == "v1":
        from vllm.v1.worker import gpu_model_runner as module
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    else:
        from vllm.v1.worker.gpu import model_runner as module
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    return module, GPUModelRunner


def _profiling_runner(monkeypatch, version):
    """A bare runner with the accelerator calls of the profiling path stubbed."""
    module, GPUModelRunner = _load_gpu_model_runner(version)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    if version == "v1":
        runner._attn_group_iterator = lambda: iter(runner.attn_groups[0])
    monkeypatch.setattr(worker_utils, "set_current_vllm_config", _null_context)
    for name, stub in (
        ("memory_allocated", lambda device: 0),
        ("memory_reserved", lambda device: 0),
        ("synchronize", lambda: None),
        ("empty_cache", lambda: None),
    ):
        monkeypatch.setattr(module.torch.accelerator, name, stub)
    return module, runner


def _patch_profiling_hooks(monkeypatch, module, version, runner, init_fn, cleanup_fn):
    """Redirect the minimal-KV-cache bootstrap/teardown used by persistent
    workspace profiling. V1 owns them as runner methods; V2 delegates to the
    module-level helpers shared with ``cudagraph_utils`` graph profiling."""
    if version == "v1":
        runner._init_minimal_kv_cache_for_profiling = init_fn
        runner._cleanup_profiling_kv_cache = cleanup_fn
    else:
        from vllm.v1.worker.gpu import cudagraph_utils

        monkeypatch.setattr(
            cudagraph_utils,
            "_init_minimal_kv_cache_for_profiling",
            lambda _runner: init_fn(),
        )
        monkeypatch.setattr(
            cudagraph_utils, "_teardown_profiling_state", lambda _runner: cleanup_fn()
        )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_persistent_workspace_lease_keeps_builder_allocations(monkeypatch, version):
    module, runner = _profiling_runner(monkeypatch, version)

    events = []
    references = {}

    class Builder:
        def __init__(self):
            references["builder"] = weakref.ref(self)

        def prepare_workspace_for_profiling(self, materialize):
            if not materialize:
                current_workspace_manager().get_simultaneous(((2048,), torch.uint8))
                events.append("reserve")
                return
            self.int_workspace = torch.empty(1536, dtype=torch.uint8)
            references["int_workspace"] = weakref.ref(self.int_workspace)
            events.append("materialize")

    def init_minimal_kv_cache():
        events.append("init")
        runner.attn_groups = [[SimpleNamespace(metadata_builders=[Builder()])]]

    def cleanup_profiling_kv_cache():
        events.append("cleanup")
        del runner.attn_groups
        gc.collect()

    _patch_profiling_hooks(
        monkeypatch,
        module,
        version,
        runner,
        init_minimal_kv_cache,
        cleanup_profiling_kv_cache,
    )

    def reset_peak_memory_stats(device):
        events.append("reset_peak")
        assert references["builder"]() is not None
        assert references["int_workspace"]() is not None
        assert current_workspace_manager().workspace_sizes_bytes() == (2048,)

    monkeypatch.setattr(
        module.torch.accelerator,
        "reset_peak_memory_stats",
        reset_peak_memory_stats,
    )

    with _managed_workspace():
        lease = worker_utils.prepare_profiling_workspace(runner)
        assert events == ["init", "reserve", "materialize", "cleanup", "reset_peak"]
        assert references["builder"]() is not None
        assert references["int_workspace"]() is not None

        lease.clear()
        gc.collect()

        assert references["builder"]() is None
        assert references["int_workspace"]() is None
        assert current_workspace_manager().workspace_sizes_bytes() == (2048,)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_persistent_workspace_preparation_preserves_primary_error(monkeypatch, version):
    module, runner = _profiling_runner(monkeypatch, version)

    class Builder:
        def prepare_workspace_for_profiling(self, materialize):
            raise ValueError("primary workspace error")

    cleanup_calls = []

    def failing_cleanup():
        cleanup_calls.append("cleanup")
        del runner.attn_groups
        raise RuntimeError("secondary cleanup error")

    _patch_profiling_hooks(
        monkeypatch,
        module,
        version,
        runner,
        lambda: setattr(
            runner,
            "attn_groups",
            [[SimpleNamespace(metadata_builders=[Builder()])]],
        ),
        failing_cleanup,
    )

    with (
        _managed_workspace(),
        pytest.raises(ValueError, match="primary workspace error"),
    ):
        worker_utils.prepare_profiling_workspace(runner)

    assert cleanup_calls == ["cleanup"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_final_persistent_workspace_reserve_sets_and_enforces_baseline(
    monkeypatch, version
):
    _, GPUModelRunner = _load_gpu_model_runner(version)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    requested_sizes = iter([2048, 1024, 4096])

    def fake_reserve(_runner):
        current_workspace_manager().get_simultaneous(
            ((next(requested_sizes),), torch.uint8)
        )

    monkeypatch.setattr(worker_utils, "reserve_attention_workspace", fake_reserve)

    with _managed_workspace():
        current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
        worker_utils.reserve_persistent_attention_workspace(runner)
        assert runner._profiled_persistent_workspace_sizes == (2048,)
        worker_utils.reserve_persistent_attention_workspace(runner)
        with pytest.raises(
            AssertionError,
            match="exceeded its profiled size during",
        ):
            worker_utils.reserve_persistent_attention_workspace(runner)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_final_persistent_workspace_reserve_rejects_preexisting_growth(
    monkeypatch, version
):
    _, GPUModelRunner = _load_gpu_model_runner(version)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    reserve_calls = []
    monkeypatch.setattr(
        worker_utils, "reserve_attention_workspace", lambda r: reserve_calls.append(r)
    )

    with _managed_workspace():
        current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
        runner._profiled_persistent_workspace_sizes = (
            current_workspace_manager().workspace_sizes_bytes()
        )
        current_workspace_manager().get_simultaneous(((2048,), torch.uint8))

        with pytest.raises(
            AssertionError,
            match="exceeded its profiled size before",
        ):
            worker_utils.reserve_persistent_attention_workspace(runner)
        assert reserve_calls == []


def test_v2_teardown_profiling_state_releases_builder_refs(monkeypatch):
    from vllm.v1.worker.gpu import cudagraph_utils

    class Builder:
        pass

    builder = Builder()
    builder_ref = weakref.ref(builder)
    layer = SimpleNamespace(
        kv_cache=torch.empty(1),
        impl=SimpleNamespace(_k_scale_cache=object(), _v_scale_cache=object()),
    )

    runner = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=4),
        kv_caches=[torch.empty(1)],
        attn_groups=[[SimpleNamespace(metadata_builders=[builder])]],
        kv_cache_config=object(),
        cudagraph_manager=object(),
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
        model_state=SimpleNamespace(supports_mm_inputs=False),
        lora_config=None,
        maybe_remove_all_loras=lambda lora_config: None,
    )
    del builder

    monkeypatch.setattr(cudagraph_utils.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(cudagraph_utils.torch.accelerator, "empty_cache", lambda: None)

    cudagraph_utils._teardown_profiling_state(runner)
    gc.collect()

    assert runner.kv_caches == []
    assert runner.attn_groups == []
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cudagraph_manager is None
    assert runner.cache_config.num_gpu_blocks is None
    assert isinstance(layer.kv_cache, torch.Tensor)
    assert layer.kv_cache.numel() == 0
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    assert builder_ref() is None


def test_v2_capture_reserves_workspace_before_measurement_and_locks(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    class Builder:
        reserved = False

        def prepare_workspace_for_profiling(self, materialize):
            if materialize:
                events.append("builder_reserve")
                self.reserved = True

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
    runner.is_encoder_only = False
    runner.cudagraph_manager = FakeCudaGraphManager()
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda lora_config: _null_context()
    runner.model = object()
    # capture_model() checks the encoder capture path before the decoder one.
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.input_buffers = object()
    runner.intermediate_tensors = None
    runner.block_tables = object()
    runner.attn_groups = [[SimpleNamespace(metadata_builders=[builder])]]
    runner.kv_cache_config = object()
    runner.use_aux_hidden_state_outputs = False
    runner.speculator = None
    runner.adaptive_verification = None
    runner.pcp_manager = None
    # capture_model() resets the connector's capture state at the end; the
    # real runner sets this in __init__, which this stub bypasses.
    runner.kv_connector = SimpleNamespace(reset_capture_state=lambda: None)

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


# --------------------------------------------------------------------------- #
# Composite backends
# --------------------------------------------------------------------------- #
# A composite routes one batch to two child builders, either of which can be
# chosen at runtime. The profiling lifecycle therefore has to reach both, and
# the pair can only promise what both children can honour.


@pytest.mark.parametrize("raises", [False, True])
def test_minimal_kv_cache_config_restores_block_override(monkeypatch, raises):
    """The profiling override must not leak into the real KV cache sizing."""
    cache_config = SimpleNamespace(num_gpu_blocks_override=7)
    runner = SimpleNamespace(
        vllm_config=object(),
        cache_config=cache_config,
        max_num_reqs=4,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=8),
    )
    seen = []

    def from_groups(vllm_config, groups, available_memory):
        seen.append(cache_config.num_gpu_blocks_override)
        if raises:
            raise RuntimeError("boom")
        return "minimal"

    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.get_kv_cache_groups", lambda cfg, spec: []
    )
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups", from_groups
    )

    if raises:
        with pytest.raises(RuntimeError, match="boom"):
            worker_utils.build_minimal_kv_cache_config(runner, object())
    else:
        assert worker_utils.build_minimal_kv_cache_config(runner, object()) == "minimal"

    assert seen == [4]
    assert cache_config.num_gpu_blocks_override == 7


def test_v1_minimal_kv_cache_init_keeps_spec_registry_check(monkeypatch):
    """V1 validates the KV cache spec registry before building the config."""
    _, GPUModelRunner = _load_gpu_model_runner("v1")
    runner = GPUModelRunner.__new__(GPUModelRunner)
    spec = object()
    checked = []
    runner.get_kv_cache_spec = lambda: spec
    runner.cache_config = SimpleNamespace(num_gpu_blocks=None)
    runner.initialize_kv_cache = lambda config, is_profiling: None

    module, _ = _load_gpu_model_runner("v1")
    monkeypatch.setattr(
        module.KVCacheSpecRegistry,
        "check_kv_cache_spec_registry",
        staticmethod(lambda s: checked.append(s)),
    )
    monkeypatch.setattr(
        module,
        "build_minimal_kv_cache_config",
        lambda r, s: SimpleNamespace(num_blocks=1),
    )

    runner._init_minimal_kv_cache_for_profiling()
    assert checked == [spec]


def test_v1_capture_model_reserves_workspace_and_locks(monkeypatch):
    """V1 capture_model() runs the shared reservation, then locks the arena."""
    module, GPUModelRunner = _load_gpu_model_runner("v1")
    events: list[str] = []

    class Builder:
        def prepare_workspace_for_profiling(self, materialize):
            events.append(f"prepare:{materialize}")

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    runner.vllm_config = SimpleNamespace(
        profiler_config=SimpleNamespace(capture_torch_profiler=False)
    )
    runner.attn_groups = [[SimpleNamespace(metadata_builders=[Builder()])]]
    runner.encoder_cudagraph_manager = None
    runner._maybe_init_encoder_cudagraph_manager = lambda: None
    runner._freeze_gc = _null_context
    runner.cudagraph_dispatcher = SimpleNamespace(get_capture_descs=lambda: [])

    free_memory = iter([10_000, 9_000])
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group",
        lambda: SimpleNamespace(local_rank=1),
    )
    monkeypatch.setattr(module, "graph_capture", lambda device: _null_context())
    monkeypatch.setattr(module, "set_cudagraph_capturing_enabled", lambda _: None)
    monkeypatch.setattr(module, "lock_workspace", lambda: events.append("lock"))
    monkeypatch.setattr(module.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(module.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        module.torch.accelerator, "get_memory_info", lambda: (next(free_memory), 0)
    )

    assert runner.capture_model() == 1_000
    assert events == ["prepare:False", "prepare:True", "lock"]


def test_v1_profile_cudagraph_memory_bootstraps_minimal_kv(monkeypatch):
    """V1 graph profiling installs its own minimal KV cache before capturing."""
    module, GPUModelRunner = _load_gpu_model_runner("v1")
    events: list[str] = []

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()
    runner._init_minimal_kv_cache_for_profiling = lambda: events.append("init")
    runner._cleanup_profiling_kv_cache = lambda: events.append("cleanup")
    runner.cudagraph_dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [], cudagraph_keys={}, keys_initialized=True
    )
    runner._create_encoder_cudagraph_manager = lambda: None
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: None

    monkeypatch.setattr(
        worker_utils, "set_current_vllm_config", lambda _: _null_context()
    )
    monkeypatch.setattr(
        module, "set_current_vllm_config", lambda _: _null_context(), raising=False
    )

    assert runner.profile_cudagraph_memory() == 0
    assert events == ["init", "cleanup"]


def test_arena_pass_never_leaves_a_builder_on_a_stale_arena(monkeypatch):
    """A small builder must not cache the arena a larger one later replaces."""
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    created: list[tuple[Any, int]] = []

    def fake_prefill_wrapper(self, causal=True):
        buffer = self._get_workspace_buffer()
        wrapper = SimpleNamespace(
            _float_workspace_buffer=buffer,
            _requested=self._default_workspace_buffer_size(),
        )
        created.append((wrapper, self._default_workspace_buffer_size()))
        return wrapper

    monkeypatch.setattr(
        flashinfer_backend.FlashInferMetadataBuilder,
        "_get_prefill_wrapper",
        fake_prefill_wrapper,
    )
    # Drop the 394 MiB floor so the per-builder estimate decides the size.
    monkeypatch.setattr(
        flashinfer_backend.envs, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", 1
    )
    monkeypatch.setattr(flashinfer_backend.envs, "VLLM_BATCH_INVARIANT", False)

    def make(tokens):
        builder = _make_flashinfer_builder(flashinfer_backend)
        builder.max_num_batched_tokens = tokens
        builder.num_qo_heads = 4
        builder.head_dim = 16
        builder.use_xqa = False
        builder._reservation_trtllm_prefill = False
        builder.kv_cache_spec = SimpleNamespace(non_causal=False)
        builder.model_config = SimpleNamespace(is_mm_prefix_lm=False, max_model_len=8)
        builder.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=0, max_num_batched_tokens=0)
        )
        builder.enable_cuda_graph = False
        return builder

    small, large = make(8), make(4096)
    assert small._default_workspace_buffer_size() < (
        large._default_workspace_buffer_size()
    )

    with _managed_workspace():
        # Pass 1 over every builder, smallest first so the arena grows after
        # the small builder has already asked for its size.
        for builder in (small, large):
            builder.prepare_workspace_for_profiling(False)
        arena_after_pass1 = current_workspace_manager().workspace_sizes_bytes()

        # Pass 2 builds the wrappers against the settled arena.
        for builder in (small, large):
            builder.prepare_workspace_for_profiling(True)

        assert len(created) == 2
        backing = current_workspace_manager()._current_workspaces[0]
        for wrapper, requested in created:
            # No wrapper points at a freed allocation ...
            assert wrapper._float_workspace_buffer.data_ptr() == backing.data_ptr()
            # ... and each one still got at least the size it asked for.
            assert _nbytes(backing) >= requested

        # Materialization must not have grown the arena any further.
        assert current_workspace_manager().workspace_sizes_bytes() == arena_after_pass1
