# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import PersistentWorkspaceProfilingSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.workspace import (
    PersistentWorkspaceLease,
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
    builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    builder.use_trtllm_decode_attention = False
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


def test_flashinfer_rebinds_all_builders_after_shared_arena_growth():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    with _managed_workspace():
        first = _make_flashinfer_builder(flashinfer_backend)
        first_wrapper = _FakeFlashInferWrapper(first._get_workspace_buffer(1024))
        first._register_workspace_wrapper(first_wrapper)
        first_int_pointer = first_wrapper._int_workspace_buffer.data_ptr()
        first_pointer_before_growth = first._workspace_buffer.data_ptr()

        second = _make_flashinfer_builder(flashinfer_backend)
        second_wrapper = _FakeFlashInferWrapper(second._get_workspace_buffer(2048))
        second._register_workspace_wrapper(second_wrapper)
        second_int_pointer = second_wrapper._int_workspace_buffer.data_ptr()
        final_pointer = current_workspace_manager().get_workspace().data_ptr()
        assert first_pointer_before_growth != final_pointer

        first.rebind_workspace_after_reservation()
        second.rebind_workspace_after_reservation()

        assert first._workspace_buffer.data_ptr() == final_pointer
        assert second._workspace_buffer.data_ptr() == final_pointer
        assert first_wrapper._float_workspace_buffer.data_ptr() == final_pointer
        assert second_wrapper._float_workspace_buffer.data_ptr() == final_pointer
        assert first_wrapper._int_workspace_buffer.data_ptr() == first_int_pointer
        assert second_wrapper._int_workspace_buffer.data_ptr() == second_int_pointer


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
    builder._workspace_state = flashinfer_backend._FlashInferWorkspaceState()
    builder.device = torch.device("cpu")
    builder.use_dcp = False
    builder.use_xqa = False
    builder.use_trtllm_prefill_attention = use_trtllm_prefill_attention
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
            builder._register_workspace_wrapper(causal_wrapper)
        return builder._prefill_wrapper

    monkeypatch.setattr(builder, "_get_prefill_wrapper", get_prefill_wrapper)
    return causal_wrapper


def test_persistent_reserve_grows_arena_before_runtime_wrappers(monkeypatch):
    """The runtime wrappers below the hoist never grow the arena themselves."""
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    default_float_bytes = 4096
    builder = _reservation_builder(
        flashinfer_backend,
        default_float_bytes=default_float_bytes,
        is_mm_prefix_lm=True,
    )
    arena_bytes_when_built = []

    causal_wrapper = _install_prefill_factories(builder, monkeypatch)
    inner = builder._get_prefill_wrapper

    def recording_get_prefill_wrapper(causal=True):
        if builder._prefill_wrapper is None:
            arena_bytes_when_built.append(
                _nbytes(current_workspace_manager().get_workspace())
            )
        return inner(causal=causal)

    monkeypatch.setattr(builder, "_get_prefill_wrapper", recording_get_prefill_wrapper)
    # Decode is routed to direct trtllm-gen, whose workspace lives outside the
    # arena this test is about; keep the reservation off the real allocator.
    monkeypatch.setattr(
        flashinfer_backend, "_get_trtllm_workspace_buffer", lambda: None
    )

    with _managed_workspace():
        builder.reserve_workspace_for_memory_profiling()
        assert arena_bytes_when_built == [default_float_bytes]
        assert causal_wrapper._float_workspace_buffer is not None


@pytest.mark.parametrize(
    ("decode_context_parallel_size", "is_mm_prefix_lm", "expected"),
    [
        pytest.param(1, False, True, id="single-rank"),
        pytest.param(2, False, False, id="dcp-fallback"),
        pytest.param(1, True, False, id="mm-prefix-fallback"),
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

    assert FlashInferMetadataBuilder.get_persistent_workspace_memory_profiling_support(
        config, _attention_spec(128)
    ) is (
        PersistentWorkspaceProfilingSupport.REQUIRED
        if expected
        else PersistentWorkspaceProfilingSupport.UNSUPPORTED
    )


def test_flashinfer_mm_prefix_route_does_not_open_the_profile_gate():
    """A native-prefill route is not on its own enough to opt in.

    mm-prefix reaches the native prefill leg, but through a wrapper the
    reservation does not own, so the gate has to stay closed until it does.
    """
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    builder = _reservation_builder(
        flashinfer_backend, default_float_bytes=4096, is_mm_prefix_lm=True
    )
    builder.use_trtllm_prefill_attention = True

    assert builder._get_workspace_routes().native_prefill

    config = SimpleNamespace(
        model_config=SimpleNamespace(is_mm_prefix_lm=True),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    support = flashinfer_backend.FlashInferMetadataBuilder
    assert (
        support.get_persistent_workspace_memory_profiling_support(
            config, _attention_spec(128)
        )
        is PersistentWorkspaceProfilingSupport.UNSUPPORTED
    )


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
        FlashInferMetadataBuilder.get_persistent_workspace_memory_profiling_support(
            config, kv_cache_spec
        )
        is PersistentWorkspaceProfilingSupport.UNSUPPORTED
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
    builder.use_trtllm_prefill_attention = trtllm_prefill
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
        first_state = first_builder.get_workspace_buffer_state()
        first = first_builder._get_workspace_buffer(1)

        second_builder = _make_flashinfer_builder(flashinfer_backend)
        second_builder.set_workspace_buffer_state(first_state)
        second = second_builder._get_workspace_buffer(1)

        assert first.device.type == "cpu"
        assert first.dtype == torch.uint8
        assert first.numel() == 1
        assert first.data_ptr() == second.data_ptr()


def test_flashinfer_workspace_buffer_growth_resets_registered_wrappers():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    builder = _make_flashinfer_builder(flashinfer_backend)

    with _managed_workspace():
        wrapper = _FakeFlashInferWrapper(builder._get_workspace_buffer(1))
        builder._register_workspace_wrapper(wrapper)
        reset_calls = wrapper.reset_calls

        # Growing the shared arena hands every registered wrapper the new
        # buffer, so none of them is left pointing at the freed one.
        builder._get_workspace_buffer(1024)
        assert builder._workspace_buffer.numel() == 1024
        assert wrapper._float_workspace_buffer.data_ptr() == (
            builder._workspace_buffer.data_ptr()
        )
        assert wrapper._float_workspace_buffer.numel() == 1024
        assert wrapper.reset_calls > reset_calls

        # Rebinding to the buffer a wrapper already holds changes nothing.
        reset_calls = wrapper.reset_calls
        builder._workspace_state.set_buffer(builder._workspace_buffer)
        assert wrapper.reset_calls == reset_calls

        wrapper_ref = weakref.ref(wrapper)
        del wrapper
        gc.collect()

        builder._workspace_state.set_buffer(torch.empty(2048, dtype=torch.uint8))
        assert wrapper_ref() is None
        assert builder._workspace_state.wrappers == []


def test_flashinfer_dcp_prefill_wrapper_rebinds_its_inner_wrappers():
    """The DCP prefill wrapper is a container with no workspace of its own.

    Registering the container would rebind nothing, so a later arena growth
    would leave its two inner wrappers on the workspace they captured at
    construction time.
    """
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    builder = _make_flashinfer_builder(flashinfer_backend)

    dcp_wrapper = flashinfer_backend.BatchDCPPrefillWrapper.__new__(
        flashinfer_backend.BatchDCPPrefillWrapper
    )
    original = torch.empty(128, dtype=torch.uint8)
    dcp_wrapper._context = _FakeFlashInferWrapper(original)
    dcp_wrapper._new_tokens = _FakeFlashInferWrapper(original)

    builder._register_workspace_wrapper(dcp_wrapper)
    grown = torch.empty(2048, dtype=torch.uint8)
    builder._workspace_state.set_buffer(grown)

    for inner in (dcp_wrapper._context, dcp_wrapper._new_tokens):
        assert inner._float_workspace_buffer.data_ptr() == grown.data_ptr()
        assert inner.reset_calls >= 1


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
                    self._get_workspace_buffer(1), int_workspace_bytes=64
                )
                self._register_workspace_wrapper(self._prefill_wrapper)
            return self._prefill_wrapper

        def _get_decode_wrapper(self, batch_size, use_cudagraph=False):
            self.calls.append(("decode_cg" if use_cudagraph else "decode", batch_size))
            if use_cudagraph:
                wrapper = self._decode_wrappers_cudagraph.get(batch_size)
            else:
                wrapper = self._decode_wrapper
            if wrapper is None:
                wrapper = _FakeFlashInferWrapper(
                    self._get_workspace_buffer(1),
                    int_workspace_bytes=128 if use_cudagraph else 96,
                )
                self._register_workspace_wrapper(wrapper)
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
        reserved = builder.reserve_workspace_for_memory_profiling()
        builder.rebind_workspace_after_reservation()

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
        final_workspace = current_workspace_manager().get_workspace()
        use_native_prefill = not use_trtllm_prefill_attention
        use_native_decode = not use_trtllm_decode_attention
        if use_native_prefill or use_native_decode:
            assert final_workspace is not None
        else:
            assert final_workspace is None

        expected_calls: list[tuple[str, int | None]] = []
        expected_reserved = 0
        if use_native_prefill or use_native_decode:
            expected_reserved += 4096
        if use_native_prefill:
            expected_calls.append(("prefill", None))
            expected_reserved += 64
        if not use_trtllm_decode_attention:
            expected_calls.extend(
                [
                    ("decode", 3),
                    ("decode_cg", 2),
                    ("decode_cg", 4),
                ]
            )
            expected_reserved += 96 + 128 + 128
        if use_trtllm_prefill_attention or use_trtllm_decode_attention:
            expected_reserved += trtllm_workspace.numel()
        assert reserved == expected_reserved
        assert builder.calls == expected_calls
        assert len(trtllm_workspace_calls) == int(
            use_trtllm_prefill_attention or use_trtllm_decode_attention
        )
        if final_workspace is not None:
            assert all(
                wrapper._float_workspace_buffer.data_ptr() == final_workspace.data_ptr()
                for wrapper in wrappers
            )
        assert len(
            {wrapper._int_workspace_buffer.data_ptr() for wrapper in wrappers}
        ) == len(wrappers)
        assert len(builder._workspace_state._live_wrappers()) == len(wrappers)

        wrapper_refs = [weakref.ref(wrapper) for wrapper in wrappers]
        lease = PersistentWorkspaceLease([builder])
        del wrappers
        del builder
        gc.collect()
        assert all(wrapper_ref() is not None for wrapper_ref in wrapper_refs)

        lease.release()
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
    monkeypatch.setattr(module, "set_current_vllm_config", _null_context)
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
        monkeypatch.setattr(
            module,
            "_init_minimal_kv_cache_for_profiling",
            lambda _runner: init_fn(),
        )
        monkeypatch.setattr(
            module, "_teardown_profiling_state", lambda _runner: cleanup_fn()
        )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_persistent_workspace_lease_keeps_builder_allocations(monkeypatch, version):
    module, runner = _profiling_runner(monkeypatch, version)

    events = []
    references = {}

    class Builder:
        def __init__(self):
            references["builder"] = weakref.ref(self)

        def reserve_workspace_for_memory_profiling(self):
            self.int_workspace = torch.empty(1536, dtype=torch.uint8)
            references["int_workspace"] = weakref.ref(self.int_workspace)
            current_workspace_manager().get_simultaneous(((2048,), torch.uint8))
            events.append("reserve")
            return 3584

        def rebind_workspace_after_reservation(self):
            self.float_workspace = current_workspace_manager().get_workspace()
            events.append("rebind")

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
        lease = runner.prepare_profiling_workspace()
        assert events == ["init", "reserve", "rebind", "cleanup", "reset_peak"]
        assert references["builder"]() is not None
        assert references["int_workspace"]() is not None

        lease.release()
        gc.collect()

        assert references["builder"]() is None
        assert references["int_workspace"]() is None
        assert current_workspace_manager().workspace_sizes_bytes() == (2048,)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_persistent_workspace_preparation_preserves_primary_error(monkeypatch, version):
    module, runner = _profiling_runner(monkeypatch, version)

    class Builder:
        def reserve_workspace_for_memory_profiling(self):
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
        runner.prepare_profiling_workspace()

    assert cleanup_calls == ["cleanup"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_final_persistent_workspace_reserve_sets_and_enforces_baseline(version):
    _, GPUModelRunner = _load_gpu_model_runner(version)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    requested_sizes = iter([2048, 1024, 4096])

    def reserve_attention_workspace(*, memory_profiling):
        assert memory_profiling
        current_workspace_manager().get_simultaneous(
            ((next(requested_sizes),), torch.uint8)
        )
        return 123

    runner._reserve_attention_workspace = reserve_attention_workspace

    with _managed_workspace():
        current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
        assert runner.reserve_persistent_attention_workspace() == 123
        assert runner._profiled_persistent_workspace_sizes == (2048,)
        assert runner.reserve_persistent_attention_workspace() == 123
        with pytest.raises(
            AssertionError,
            match="exceeded its profiled size during",
        ):
            runner.reserve_persistent_attention_workspace()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_final_persistent_workspace_reserve_rejects_preexisting_growth(version):
    _, GPUModelRunner = _load_gpu_model_runner(version)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    reserve_calls = []
    runner._reserve_attention_workspace = lambda **kwargs: reserve_calls.append(kwargs)

    with _managed_workspace():
        current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
        runner.record_persistent_attention_workspace_profile()
        current_workspace_manager().get_simultaneous(((2048,), torch.uint8))

        with pytest.raises(
            AssertionError,
            match="exceeded its profiled size before",
        ):
            runner.reserve_persistent_attention_workspace()
        assert reserve_calls == []


def test_cudagraph_profile_rejects_builder_init_workspace_growth(monkeypatch):
    module, GPUModelRunner = _load_gpu_model_runner("v1")

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()
    cleanup_calls = []
    runner._init_minimal_kv_cache_for_profiling = lambda: (
        current_workspace_manager().get_simultaneous(((2048,), torch.uint8))
    )
    runner._cleanup_profiling_kv_cache = lambda: cleanup_calls.append("cleanup")
    monkeypatch.setattr(module, "set_current_vllm_config", _null_context)

    with _managed_workspace():
        current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
        with pytest.raises(
            AssertionError,
            match="grew while rebuilding CUDA graph profiling metadata",
        ):
            runner.profile_cudagraph_memory(persistent_workspace_profiled=True)

    assert cleanup_calls == ["cleanup"]


def test_v2_cudagraph_profile_rejects_workspace_growth(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)

    def grow_arena(_runner):
        current_workspace_manager().get_simultaneous(((2048,), torch.uint8))
        return 4096

    monkeypatch.setattr(gpu_model_runner_v2, "_profile_cudagraph_memory", grow_arena)

    with _managed_workspace():
        current_workspace_manager().get_simultaneous(((1024,), torch.uint8))
        with pytest.raises(
            AssertionError,
            match="grew during CUDA graph profiling",
        ):
            runner.profile_cudagraph_memory(persistent_workspace_profiled=True)


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

        def reserve_workspace_for_cudagraph_capture(self):
            events.append("builder_reserve")
            self.reserved = True
            return 128

        def rebind_workspace_after_reservation(self):
            pass

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
