# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import CompilerManager
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.config.utils import Range

TestData = namedtuple("TestData", ["fx_graph", "inputs", "expected_output"])


@pytest.fixture
def test_data() -> TestData:
    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    # example_inputs must be a list for unpacking with * in tests
    example_inputs = [torch.tensor([[1.0, 2.0, 3.0]])]
    fx_graph = make_fx(model_fn)(example_inputs[0])
    expected_output = torch.tensor([[2.0, 3.0, 4.0]])
    return TestData(fx_graph, example_inputs, expected_output)


def make_manager(backend: str) -> CompilerManager:
    os.environ["VLLM_USE_STANDALONE_COMPILE"] = "0"
    if backend == "inductor_standalone":
        backend = "inductor"
        os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
    config = CompilationConfig(
        backend=backend,
        cudagraph_mode=CUDAGraphMode.NONE,
        use_inductor_graph_partition=False,
    )
    return CompilerManager(config)


@pytest.mark.parametrize(
    "backend, disable_cache",
    [
        pytest.param("eager", False, id="eager_enable_cache"),
        pytest.param("eager", True, id="eager_disable_cache"),
        pytest.param("inductor", True, id="inductor_disable_cache"),
        pytest.param(
            "inductor_standalone", True, id="inductor_standalone_disable_cache"
        ),
    ],
)
def test_no_cache(
    backend: str, disable_cache: bool, test_data: TestData, tmp_path: Path
):
    """
    Test that caching is disabled when disable_cache=True or when using Eager backend.

    Verifies that:
    - Eager backend never caches (regardless of disable_cache setting)
    - Inductor backend respects disable_cache=True (no cache written)
    - No cache file is written to disk
    """
    cache_dir = tmp_path
    compile_range = Range(0, 1)

    compiler_manager = make_manager(backend=backend)
    compiler_manager.initialize_cache(str(cache_dir), disable_cache=disable_cache)

    # Verify that cache should be empty at the start
    assert len(compiler_manager.cache) == 0
    assert not compiler_manager.is_cache_updated

    # Compile
    compiled_func = compiler_manager.compile(
        graph=test_data.fx_graph,
        example_inputs=test_data.inputs,
        additional_inductor_config={},
        compilation_config=compiler_manager.compilation_config,
        compile_range=compile_range,
        graph_index=0,
    )

    # Verify that the compiled function computes correctly
    assert torch.equal(compiled_func(*test_data.inputs), test_data.expected_output)

    # Verify that the compiled function is the original graph when using EagerAdaptor,
    # and is a different function when using InductorAdaptors
    if backend == "eager":
        assert compiled_func is test_data.fx_graph
    else:
        assert compiled_func is not test_data.fx_graph

    # Verify that the cache is empty and not updated
    assert len(compiler_manager.cache) == 0
    assert not compiler_manager.is_cache_updated

    # Verify that no cache file is written
    compiler_manager.save_to_file()
    assert not (cache_dir / "vllm_compile_cache.py").exists()


@pytest.mark.parametrize("backend", ["inductor", "inductor_standalone"])
def test_cache_miss_and_hit(backend: str, test_data: TestData, tmp_path: Path):
    """
    Test cache miss and hit behavior for Inductor backends.

    Verifies that:
    - First compilation triggers backend.compile (cache miss)
    - Second compilation uses cached result (cache hit, backend.compile NOT called)
    - Both compiled functions compute correctly
    """
    cache_dir = tmp_path
    compile_range = Range(0, 1)

    # Initialize manager and cache
    compiler_manager = make_manager(backend=backend)
    compiler_manager.initialize_cache(str(cache_dir), disable_cache=False)

    # Verify that cache should be empty at the start
    assert len(compiler_manager.cache) == 0
    assert not compiler_manager.is_cache_updated

    # === First compilation (Cache Miss) ===
    with patch.object(
        compiler_manager.compiler,
        "compile",
        wraps=compiler_manager.compiler.compile,
    ) as mock_backend_compile:
        compiled_func_1 = compiler_manager.compile(
            graph=test_data.fx_graph,
            example_inputs=test_data.inputs,
            additional_inductor_config={},
            compilation_config=compiler_manager.compilation_config,
            compile_range=compile_range,
            graph_index=0,
        )
        # Verify backend.compile is called (cache miss)
        mock_backend_compile.assert_called_once()

    # Verify that the compiled function computes correctly
    assert torch.equal(compiled_func_1(*test_data.inputs), test_data.expected_output)

    # Verify that the manager has written the product and handle into the cache
    assert len(compiler_manager.cache) > 0
    assert compiler_manager.is_cache_updated

    # === Second compilation (Cache Hit) ===
    with patch.object(
        compiler_manager.compiler,
        "compile",
        wraps=compiler_manager.compiler.compile,
    ) as mock_backend_compile:
        compiled_func_2 = compiler_manager.compile(
            graph=test_data.fx_graph,
            example_inputs=test_data.inputs,
            additional_inductor_config={},
            compilation_config=compiler_manager.compilation_config,
            compile_range=compile_range,
            graph_index=0,
        )
        # Verify backend.compile is NOT called (cache hit)
        mock_backend_compile.assert_not_called()

    # Verify that the loaded-from-cache function also computes correctly
    assert torch.equal(compiled_func_2(*test_data.inputs), test_data.expected_output)


@pytest.mark.parametrize("backend", ["inductor", "inductor_standalone"])
def test_cache_persistence_across_managers(
    backend: str, test_data: TestData, tmp_path: Path
):
    """Test that cache persists across different CompilerManager instances.

    Verifies:
    - First manager compiles and writes cache
    - Second manager loads from cache without recompiling
    """
    cache_dir = tmp_path
    compile_range = Range(0, 1)

    # === First manager: compile and save ===
    manager_1 = make_manager(backend=backend)
    manager_1.initialize_cache(cache_dir=str(cache_dir), disable_cache=False)
    with patch.object(
        manager_1.compiler,
        "compile",
        wraps=manager_1.compiler.compile,
    ) as mock_compile:
        manager_1.compile(
            graph=test_data.fx_graph,
            example_inputs=test_data.inputs,
            additional_inductor_config=manager_1.compilation_config.inductor_compile_config,
            compilation_config=manager_1.compilation_config,
            compile_range=compile_range,
            graph_index=0,
        )
        mock_compile.assert_called_once()
    manager_1.save_to_file()

    # Verify that cache file is written
    assert (cache_dir / "vllm_compile_cache.py").exists()

    # === Second manager: load from cache ===
    manager_2 = make_manager(backend=backend)
    manager_2.initialize_cache(cache_dir=str(cache_dir), disable_cache=False)
    with patch.object(
        manager_2.compiler,
        "compile",
        wraps=manager_2.compiler.compile,
    ) as mock_compile:
        compiled_func = manager_2.compile(
            graph=test_data.fx_graph,
            example_inputs=test_data.inputs,
            additional_inductor_config={},
            compilation_config=manager_2.compilation_config,
            compile_range=compile_range,
            graph_index=0,
        )
        mock_compile.assert_not_called()

    # Verify that the loaded-from-cache function also computes correctly
    assert torch.equal(compiled_func(*test_data.inputs), test_data.expected_output)
