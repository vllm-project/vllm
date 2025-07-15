# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import vllm
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.utils import _is_torch_equal_or_newer


def test_version():
    assert _is_torch_equal_or_newer('2.8.0.dev20250624+cu128', '2.8.0.dev')
    assert _is_torch_equal_or_newer('2.8.0a0+gitc82a174', '2.8.0.dev')
    assert _is_torch_equal_or_newer('2.8.0', '2.8.0.dev')
    assert _is_torch_equal_or_newer('2.8.1', '2.8.0.dev')
    assert not _is_torch_equal_or_newer('2.7.1', '2.8.0.dev')


def test_use_cudagraphs_dynamic(monkeypatch):
    assert vllm.envs.VLLM_USE_V1
    vllm_config = VllmConfig()
    assert vllm_config.compilation_config.use_cudagraph

    monkeypatch.setenv('VLLM_USE_V1', '0')
    vllm_config = VllmConfig()
    assert not vllm_config.compilation_config.use_cudagraph


# NB: We don't test VLLM_DISABLE_COMPILE_CACHE=0 because that depends
# on the state of the cache directory on the current machine, which
# may be influenced by other tests.
@pytest.mark.parametrize("val", ["1"])
def test_VLLM_DISABLE_COMPILE_CACHE(vllm_runner, monkeypatch, val):
    assert vllm.envs.VLLM_USE_V1

    # spawn means that the counters are in the same process.
    monkeypatch.setenv('VLLM_WORKER_MULTIPROC_METHOD', "spawn")
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
