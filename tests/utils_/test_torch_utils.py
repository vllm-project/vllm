# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    available_cpu_count,
    common_broadcastable_dtype,
    current_stream,
    get_kv_cache_torch_dtype,
    is_lossless_cast,
    is_quantized_kv_cache,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
    supports_xpu_fa_in_graph,
)


def test_nvfp4_4over6_cache_dtype() -> None:
    from vllm.config.cache import CacheConfig
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    cache_config = CacheConfig(cache_dtype="nvfp4_4over6")

    assert cache_config.cache_dtype == "nvfp4_4over6"
    assert get_kv_cache_torch_dtype(cache_config.cache_dtype) == torch.uint8
    assert is_quantized_kv_cache(cache_config.cache_dtype)
    assert get_kv_quant_mode(cache_config.cache_dtype) == KVQuantMode.NVFP4


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


@pytest.mark.parametrize(
    ("graph_supported", "xpu_ver", "expected"),
    [
        (True, "20260000", True),  # oneAPI 2026.0 -> FA capturable
        (True, "20260100", True),  # newer 2026.x
        (True, "20250302", False),  # oneAPI 2025.3 -> scratch not capturable
        (True, None, False),  # non-XPU torch build
        (True, "not-a-number", False),  # unparsable -> fail closed
        (False, "20260000", False),  # torch too old for any XPU graph
    ],
)
def test_supports_xpu_fa_in_graph(monkeypatch, graph_supported, xpu_ver, expected):
    monkeypatch.setattr(
        "vllm.utils.torch_utils.supports_xpu_graph", lambda: graph_supported
    )
    monkeypatch.setattr(torch.version, "xpu", xpu_ver, raising=False)
    assert supports_xpu_fa_in_graph() is expected


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
