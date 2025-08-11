# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Note: This file contains on-device integration tests that require a real
# TPU environment to run. For CPU-only unit tests of the TPU platform
# logic, see test_tpu_platform_logic.py.

import importlib
import sys

import pytest
import torch

from vllm.config import (CacheConfig, CompilationConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

# Guard the entire module to only run on TPU
if not current_platform.is_tpu():
    pytest.skip("Skipping TPU tests on non-TPU platform",
                allow_module_level=True)

# Import TPU-specific classes only after the guard
from vllm.platforms.tpu import TpuPlatform

# The module to reload in tests
TPU_PLATFORM_MODULE = "vllm.platforms.tpu"


@pytest.fixture
def reload_module_and_clear_registry():
    """
    Fixture to ensure a clean state for each test. It reloads the tpu
    module and clears the engine registry.
    """
    from vllm.dependency_injection.registry import _engine_core_proc_registry
    _engine_core_proc_registry.clear()

    # Reload the module to re-trigger import-time logic
    import vllm.platforms.tpu
    importlib.reload(vllm.platforms.tpu)
    yield
    _engine_core_proc_registry.clear()


def test_tpu_platform_discovery():
    """
    Tests if the current platform is correctly identified as TPU.
    """
    assert current_platform.is_tpu()
    assert not current_platform.is_cuda()
    assert isinstance(current_platform, TpuPlatform)


@pytest.mark.usefixtures("reload_module_and_clear_registry")
@pytest.mark.parametrize("disagg_enabled", [True, False])
def test_disaggregated_engine_registration(monkeypatch, disagg_enabled):
    """
    Tests that the DisaggEngineCoreProc is registered only when disagg mode
    is enabled via environment variables.
    """
    from vllm.dependency_injection.registry import retrieve_engine_core_proc

    # Set environment variable to control is_disagg_enabled()
    if disagg_enabled:
        monkeypatch.setenv("PREFILL_SLICES", "1")
    else:
        monkeypatch.delenv("PREFILL_SLICES", raising=False)

    # The registration happens at import time, so we need to reload the module
    importlib.reload(sys.modules[TPU_PLATFORM_MODULE])

    if disagg_enabled:
        try:
            engine_class = retrieve_engine_core_proc("disaggregated_tpu")
            from tpu_commons.core.core_tpu import DisaggEngineCoreProc
            assert engine_class is DisaggEngineCoreProc
        except ValueError:
            pytest.fail(
                "Failed to retrieve 'disaggregated_tpu' engine when it should "
                "have been registered.")
        except ImportError:
            pytest.skip(
                "Skipping registration test because tpu_commons is not "
                "installed. Run 'pip install vllm[tpu]' to run this test.")
    else:
        with pytest.raises(ValueError, match="is not registered"):
            retrieve_engine_core_proc("disaggregated_tpu")


def test_validate_request_raises_for_random_seed():
    """
    Tests that validate_request raises a ValueError for random-seed sampling.
    """
    # Setting temperature > 0 results in sampling_type being RANDOM
    params = SamplingParams(temperature=1.0)
    with pytest.raises(ValueError, match="does not support per-request seed"):
        TpuPlatform.validate_request(None, params, None)


@pytest.fixture
def vllm_config() -> VllmConfig:
    """A pytest fixture that provides a default VllmConfig object."""
    model_config = ModelConfig(model="dummy",
                               tokenizer="dummy",
                               tokenizer_mode="auto",
                               trust_remote_code=False,
                               download_dir=None,
                               load_format="auto",
                               dtype="float16",
                               seed=0,
                               revision=None,
                               code_revision=None,
                               tokenizer_revision=None,
                               max_model_len=None,
                               use_mla=False)
    cache_config = CacheConfig(block_size=16,
                               gpu_memory_utilization=0.9,
                               swap_space=4,
                               cache_dtype="auto",
                               num_gpu_blocks=None,
                               num_cpu_blocks=None)
    parallel_config = ParallelConfig(pipeline_parallel_size=1,
                                     tensor_parallel_size=1,
                                     worker_use_ray=False)
    scheduler_config = SchedulerConfig(max_num_batched_tokens=256,
                                       max_num_seqs=256,
                                       max_model_len=1024,
                                       is_multi_step=False,
                                       is_multimodal_model=False,
                                       disable_chunked_mm_input=False)
    return VllmConfig(model_config=model_config,
                      cache_config=cache_config,
                      parallel_config=parallel_config,
                      scheduler_config=scheduler_config,
                      device="tpu",
                      speculative_config=None,
                      lora_config=None,
                      compilation_config=CompilationConfig(level="DYNAMO_ONCE",
                                                           backend=""))


def test_check_and_update_config_forces_bfloat16(vllm_config: VllmConfig):
    """
    Tests that check_and_update_config forces bfloat16 for float dtypes.
    """
    vllm_config.model_config.dtype = torch.float32
    TpuPlatform.check_and_update_config(vllm_config)
    assert vllm_config.model_config.dtype == torch.bfloat16


def test_check_and_update_config_disables_multistep(vllm_config: VllmConfig):
    """
    Tests that check_and_update_config raises an error for multi-step.
    """
    vllm_config.scheduler_config.is_multi_step = True
    with pytest.raises(NotImplementedError):
        TpuPlatform.check_and_update_config(vllm_config)


def test_check_and_update_config_disables_chunked_mm(vllm_config: VllmConfig):
    """
    Tests that check_and_update_config disables chunked mm input.
    """
    vllm_config.scheduler_config.is_multimodal_model = True
    vllm_config.scheduler_config.disable_chunked_mm_input = False
    TpuPlatform.check_and_update_config(vllm_config)
    assert vllm_config.scheduler_config.disable_chunked_mm_input is True
