# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm
from vllm.compilation.counter import compilation_counter
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)

from .piecewise.test_simple import SillyModel


@pytest.fixture(scope="function", autouse=True)
def use_v1(monkeypatch):
    """
    TODO(rzou): The rest of tests/compile runs VLLM_USE_V1=0 right now,
    I'll switch them over later.
    """
    monkeypatch.setenv('VLLM_USE_V1', '1')


@pytest.mark.parametrize("enabled", [True, False])
def test_use_cudagraphs(enabled):
    assert vllm.envs.VLLM_USE_V1
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=enabled,
        cudagraph_capture_sizes=[100],
    ))
    with set_current_vllm_config(vllm_config):
        model = SillyModel(vllm_config=vllm_config, prefix='')

    inputs = torch.randn(100, device="cuda")

    with compilation_counter.expect(
            num_graphs_seen=1,  # one graph for the model
            num_cudagraph_captured=1 if enabled else 0,
    ):
        # first run is warmup
        model(inputs)
        # second run does CUDAGraphs recording (if enabled)
        model(inputs)
