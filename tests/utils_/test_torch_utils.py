# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    LayerName,
    async_tensor_h2d,
    available_cpu_count,
    common_broadcastable_dtype,
    current_stream,
    get_kv_cache_torch_dtype,
    is_lossless_cast,
    lazy_torch_compile,
    set_default_torch_dtype,
    set_random_seed,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
    weak_ref_tensors,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_weak_ref_tensors_does_not_keep_nested_tensors_alive() -> None:
    # CUDA graph wrappers weak-ref model outputs such as
    # (hidden_states, [aux_hidden_states, ...]); no tensor may stay pinned.
    # Inductor graph partition outputs can also carry hoisted LayerNames.
    hidden, aux = (torch.empty(1 << 20, device="cuda") for _ in range(2))
    layer_name = LayerName("model.layers.0.self_attn.attn")
    out = weak_ref_tensors((hidden, [aux], 3, layer_name))
    assert isinstance(out, tuple)
    weak_hidden, weak_list, size, name = out
    assert isinstance(weak_list, list) and size == 3 and name is layer_name
    assert weak_hidden.data_ptr() == hidden.data_ptr()
    assert weak_list[0].data_ptr() == aux.data_ptr()
    nbytes = hidden.nbytes + aux.nbytes
    allocated = torch.accelerator.memory_allocated()
    del hidden, aux
    assert torch.accelerator.memory_allocated() == allocated - nbytes


def test_weak_ref_tensors_rejects_unknown_nested_types() -> None:
    with pytest.raises(ValueError):
        weak_ref_tensors((1, [None]))


def test_nvfp4_4over6_cache_dtype() -> None:
    from vllm.config.cache import CacheConfig
    from vllm.v1.kv_cache_interface import (
        KVQuantMode,
        get_kv_quant_mode,
        is_quantized_kv_cache,
    )

    cache_config = CacheConfig(cache_dtype="nvfp4_4over6")

    assert cache_config.cache_dtype == "nvfp4_4over6"
    assert get_kv_cache_torch_dtype(cache_config.cache_dtype) == torch.uint8
    assert is_quantized_kv_cache(cache_config.cache_dtype)
    assert get_kv_quant_mode(cache_config.cache_dtype) == KVQuantMode.NVFP4


def test_set_default_torch_dtype_restores_dtype_after_exception() -> None:
    original_dtype = torch.get_default_dtype()

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        assert torch.get_default_dtype() == torch.bfloat16
        raise RuntimeError("expected failure")

    assert torch.get_default_dtype() == original_dtype


@pytest.mark.parametrize(
    ("src_dtype", "tgt_dtype", "expected_result"),
    [
        # Different precision_levels
        (torch.bool, torch.int8, True),
        (torch.bool, torch.float16, True),
        (torch.bool, torch.complex32, True),
        (torch.int64, torch.bool, False),
        (torch.int64, torch.float16, True),
        (torch.int64, torch.complex32, True),
        (torch.float64, torch.bool, False),
        (torch.float64, torch.int8, False),
        (torch.float64, torch.complex32, True),
        (torch.complex128, torch.bool, False),
        (torch.complex128, torch.int8, False),
        (torch.complex128, torch.float16, False),
        # precision_level=0
        (torch.bool, torch.bool, True),
        # precision_level=1
        (torch.int8, torch.int16, True),
        (torch.int16, torch.int8, False),
        (torch.uint8, torch.int8, False),
        (torch.int8, torch.uint8, False),
        # precision_level=2
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, False),
        (torch.bfloat16, torch.float32, True),
        (torch.float32, torch.bfloat16, False),
        # precision_level=3
        (torch.complex32, torch.complex64, True),
        (torch.complex64, torch.complex32, False),
    ],
)
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result


@pytest.mark.parametrize(
    ("dtypes", "expected_result"),
    [
        ([torch.bool], torch.bool),
        ([torch.bool, torch.int8], torch.int8),
        ([torch.bool, torch.int8, torch.float16], torch.float16),
        ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32),  # noqa: E501
    ],
)
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result


def _test_stream_thread(main_expected_stream: torch.cuda.Stream):
    import threading

    child_stream = torch.cuda.Stream()
    thread_stream_ready = threading.Event()
    thread_can_exit = threading.Event()

    def child_thread_func():
        with torch.cuda.stream(child_stream):
            thread_stream_ready.set()
            thread_can_exit.wait(timeout=10)

    child_thread = threading.Thread(target=child_thread_func)
    child_thread.start()

    try:
        assert thread_stream_ready.wait(timeout=5), (
            "Child thread failed to enter stream context in time"
        )

        main_current_stream = current_stream()

        assert main_current_stream != child_stream, (
            "Main thread's current_stream was contaminated by child thread"
        )
        assert main_current_stream == main_expected_stream, (
            f"Main thread's stream changed unexpectedly. "
            f"Expected {main_expected_stream}, got {main_current_stream}"
        )

        thread_can_exit.set()

    finally:
        child_thread.join(timeout=5)
        if child_thread.is_alive():
            pytest.fail("Child thread failed to exit properly")


def test_current_stream_multithread():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    main_dedicated_stream = current_stream()

    assert main_dedicated_stream.cuda_stream != 0, (
        "ROCm/CUDA should create a dedicated stream, not use default stream (0x0)"
    )

    main_stream_again = current_stream()
    assert main_stream_again == main_dedicated_stream, (
        "Multiple calls to current_stream should return the same dedicated stream"
    )

    _test_stream_thread(main_dedicated_stream)


@pytest.fixture
def restore_torch_threads(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def test_startup_omp_num_threads_divides_between_local_workers():
    """Workers share the node's usable CPUs rather than each taking them all."""
    available = available_cpu_count()
    if available < 4:
        pytest.skip("needs at least 4 usable CPUs")
    assert startup_omp_num_threads(1) == available
    assert startup_omp_num_threads(2) == available // 2
    # Never zero, however many workers share the node.
    assert startup_omp_num_threads(available * 4) == 1


def test_set_torch_threads_for_runtime(restore_torch_threads):
    torch.set_num_threads(max(2, available_cpu_count()))
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 1


def test_runtime_threads_respect_user_omp_num_threads(
    restore_torch_threads, monkeypatch: pytest.MonkeyPatch
):
    """An externally-set OMP_NUM_THREADS is the user's choice; leave it alone."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    torch.set_num_threads(3)
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 3


def test_runtime_threads_override_vllm_set_omp_num_threads(
    restore_torch_threads, monkeypatch: pytest.MonkeyPatch
):
    """The value vLLM picked for worker startup is dropped once serving starts."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv(OMP_NUM_THREADS_SET_BY_VLLM, "1")
    torch.set_num_threads(3)
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 1


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_async_tensor_h2d_staging(device):
    """Staging must preserve values while normalizing dtype and layout."""
    # Non-contiguous source with a dtype conversion: would previously sync.
    src = torch.arange(12, dtype=torch.int64).reshape(3, 4).T
    result = async_tensor_h2d(src, device=device, dtype=torch.int32)
    assert result.dtype == torch.int32
    assert result.is_contiguous()
    assert torch.equal(result.cpu(), src.to(torch.int32))

    # numpy and list sources still work.
    assert torch.equal(
        async_tensor_h2d(np.arange(4, dtype=np.int64), device=device),
        torch.arange(4, device=device),
    )
    assert torch.equal(
        async_tensor_h2d([1, 2, 3], device=device, dtype=torch.int32),
        torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
    )


def test_set_random_seed_differs_per_data_parallel_index():
    """Unseeded requests draw from the global RNGs; DP engines sharing the engine
    seed must get distinct, reproducible streams, and index 0 keeps the seed."""

    def draws(*args):
        set_random_seed(*args)
        return np.random.randint(2**31, size=4).tolist(), torch.rand(4).tolist()

    assert draws(42, 0) == draws(42)
    assert draws(42, 1) == draws(42, 1)
    assert len({str(draws(42, i)) for i in range(4)}) == 4
    assert draws(42, 1) != draws(43, 1)


def test_engine_worker_imports_leave_compiler_and_deep_ep_unloaded():
    subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""
            import sys

            import vllm.v1.engine.core
            import vllm.v1.worker.gpu_worker
            import vllm.v1.worker.gpu.model_runner

            forbidden = {
                "torch._inductor.compile_fx",
                "torch._inductor.codecache",
                "deep_ep",
            }
            loaded = forbidden.intersection(sys.modules)
            assert not loaded, sorted(loaded)

            # Recent torch versions prefetch cache keys in a dedicated pool.
            helpers = sys.modules.get("torch.utils._functools")
            pool = getattr(helpers, "_prefetch_executor", None)
            if pool is not None:
                assert not pool._threads, pool._threads
        """),
        ],
        check=True,
    )


def test_lazy_torch_compile_imports_on_first_call():
    subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""
            import sys
            import torch
            from vllm.utils.torch_utils import lazy_torch_compile

            graphs = []
            def backend(gm, inputs):
                graphs.append(gm)
                return gm.forward

            @lazy_torch_compile(backend=backend, fullgraph=True)
            def increment(x):
                return x + 1

            assert "torch._dynamo" not in sys.modules
            assert "torch._inductor" not in sys.modules
            assert increment(torch.tensor(1)).item() == 2
            assert "torch._dynamo" in sys.modules
            assert len(graphs) == 1
        """),
        ],
        check=True,
    )


@pytest.mark.parametrize("with_kwargs", [False, True])
def test_lazy_torch_compile_binds_methods_and_compiles_once(with_kwargs):
    compile_kwargs = {"fullgraph": True} if with_kwargs else {}
    decorator = (
        lazy_torch_compile(**compile_kwargs) if with_kwargs else lazy_torch_compile
    )

    class Offset:
        def __init__(self, bias):
            self.bias = bias

        @decorator
        def add(self, x):
            return x + self.bias

    first, second = Offset(1), Offset(3)
    x = torch.tensor(2)
    assert hasattr(Offset.add, "__wrapped__")
    original = Offset.add.__wrapped__
    assert original(first, x).item() == 3
    real_compile = torch.compile
    with patch.object(
        torch,
        "compile",
        side_effect=lambda fn, **kw: real_compile(fn, backend="eager", **kw),
    ) as compile_spy:
        assert first.add(x).item() == 3
        assert second.add(x).item() == 5
        assert first.add(x).item() == 3
        compile_spy.assert_called_once_with(original, **compile_kwargs)


def test_lazy_torch_compile_inlines_in_fullgraph():
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    @lazy_torch_compile(backend="eager")
    def inner(x):
        return x + 1

    @torch.compile(backend=backend, fullgraph=True)
    def outer(x):
        return inner(x) * 2

    x = torch.tensor(2)
    with patch.object(torch, "compile", side_effect=AssertionError("nested compile")):
        assert outer(x).item() == 6
    assert len(graphs) == 1


def test_lazy_torch_compile_concurrent_first_call():
    compiling = threading.Event()
    second_started = threading.Event()

    def compile_fn(fn):
        compiling.set()
        assert second_started.wait(timeout=10)
        return fn

    @lazy_torch_compile
    def increment(x):
        return x + 1

    def second_call():
        second_started.set()
        return increment(2)

    with (
        patch.object(torch, "compile", side_effect=compile_fn) as compile_spy,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(increment, 1)
        assert compiling.wait(timeout=10)
        second = pool.submit(second_call)
        assert first.result(timeout=10) == 2
        assert second.result(timeout=10) == 3
    compile_spy.assert_called_once()
