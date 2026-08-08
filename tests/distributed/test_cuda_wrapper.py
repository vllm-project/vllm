# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.device_communicators import cuda_wrapper


class FakeFunction:
    pass


class FakeLibrary:
    def __init__(self, symbols: set[str]) -> None:
        self._funcs = {symbol: FakeFunction() for symbol in symbols}

    def __getattr__(self, name: str) -> FakeFunction:
        if name not in self._funcs:
            raise AttributeError(name)
        return self._funcs[name]


@pytest.fixture(autouse=True)
def clear_cudart_cache():
    cuda_wrapper.CudaRTLibrary.path_to_library_cache.clear()
    cuda_wrapper.CudaRTLibrary.path_to_dict_mapping.clear()
    yield
    cuda_wrapper.CudaRTLibrary.path_to_library_cache.clear()
    cuda_wrapper.CudaRTLibrary.path_to_dict_mapping.clear()


def test_cudart_skips_loaded_stub_missing_required_symbol(monkeypatch):
    stub_path = "/tmp/libcudart_stub.so"
    real_path = "/tmp/libcudart.so.13"
    all_symbols = {func.name for func in cuda_wrapper.CudaRTLibrary.exported_functions}
    libraries = {
        stub_path: FakeLibrary(all_symbols - {"cudaDeviceReset"}),
        real_path: FakeLibrary(all_symbols),
    }

    def loaded_library_paths(cls, lib_name):
        assert lib_name == "libcudart"
        return [stub_path, real_path]

    monkeypatch.delenv("VLLM_CUDART_SO_PATH", raising=False)
    monkeypatch.setattr(cuda_wrapper.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        cuda_wrapper.CudaRTLibrary,
        "_loaded_library_paths",
        classmethod(loaded_library_paths),
    )
    monkeypatch.setattr(cuda_wrapper.ctypes, "CDLL", libraries.__getitem__)

    lib = cuda_wrapper.CudaRTLibrary()

    assert lib.lib is libraries[real_path]
    assert set(lib.funcs) == all_symbols
