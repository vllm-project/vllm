# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import vllm
from vllm.compilation.counter import compilation_counter
from vllm.config import (CompilationConfig, CUDAGraphMode, ModelConfig,
                         VllmConfig)
from vllm.utils import _is_torch_equal_or_newer


def test_version():
    assert _is_torch_equal_or_newer('2.8.0.dev20250624+cu128', '2.8.0.dev')
    assert _is_torch_equal_or_newer('2.8.0a0+gitc82a174', '2.8.0.dev')
    assert _is_torch_equal_or_newer('2.8.0', '2.8.0.dev')
    assert _is_torch_equal_or_newer('2.8.1', '2.8.0.dev')
    assert not _is_torch_equal_or_newer('2.7.1', '2.8.0.dev')


def test_use_cudagraphs_dynamic(monkeypatch):
    assert vllm.envs.VLLM_USE_V1
    # cudagraph_mode=PIECEWISE by default
    config = VllmConfig()
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE

    config2 = VllmConfig(compilation_config=CompilationConfig(
        cudagraph_mode=CUDAGraphMode.NONE))
    config3 = VllmConfig(compilation_config=CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL))
    config4 = VllmConfig(model_config=ModelConfig(enforce_eager=True))
    assert config2.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert config3.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert config4.compilation_config.cudagraph_mode == CUDAGraphMode.NONE

    monkeypatch.setenv('VLLM_USE_V1', '0')
    # TODO remove when V0 fully removed:
    # https://github.com/vllm-project/vllm/issues/18571

    config = VllmConfig()  # cudagraph_mode=NONE by default
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE

    config2 = VllmConfig(compilation_config=CompilationConfig(
        cudagraph_mode=CUDAGraphMode.PIECEWISE))
    config3 = VllmConfig(compilation_config=CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL))
    config4 = VllmConfig(model_config=ModelConfig(enforce_eager=True))

    assert config2.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert config3.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert config4.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
# NB: We don't test VLLM_DISABLE_COMPILE_CACHE=0 because that depends
# on the state of the cache directory on the current machine, which
# may be influenced by other tests.
@pytest.mark.parametrize("val", ["1"])
def test_VLLM_DISABLE_COMPILE_CACHE(vllm_runner, monkeypatch, val):
    assert vllm.envs.VLLM_USE_V1

    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
    monkeypatch.setenv('VLLM_DISABLE_COMPILE_CACHE', val)

    compilation_config = {
        "use_cudagraph": False,  # speed things up a bit
    }
    with (
            compilation_counter.expect(num_cache_entries_updated=0,
                                       num_compiled_artifacts_saved=0),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m',
                        compilation_config=compilation_config,
                        gpu_memory_utilization=0.4) as _):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.parametrize("enabled", [True, False])
def test_use_cudagraphs(vllm_runner, monkeypatch, enabled):
    assert vllm.envs.VLLM_USE_V1

    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

    compilation_config = {
        "cudagraph_capture_sizes": [100],
        "use_cudagraph": enabled,
    }
    with (
            compilation_counter.expect(
                num_graphs_seen=1,
                num_gpu_runner_capture_triggers=1 if enabled else 0,
                num_cudagraph_captured=13 if enabled else 0,
            ),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m',
                        compilation_config=compilation_config,
                        gpu_memory_utilization=0.4) as _):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
def test_dynamo_as_is(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

    with (
            compilation_counter.expect(dynamo_as_is_count=1),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m',
                        compilation_config={"level": 1},
                        gpu_memory_utilization=0.4) as _):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
def test_no_compilation(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

    with (
            compilation_counter.expect(num_graphs_seen=0,
                                       dynamo_as_is_count=0),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m',
                        compilation_config={"level": 0},
                        gpu_memory_utilization=0.4) as _):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
def test_enforce_eager(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

    with (
            compilation_counter.expect(num_graphs_seen=0,
                                       dynamo_as_is_count=0),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m',
                        enforce_eager=True,
                        gpu_memory_utilization=0.4) as _):
        pass
