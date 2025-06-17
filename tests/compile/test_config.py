# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm
from vllm.compilation.counter import compilation_counter
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)

from .piecewise.test_simple import SillyModel


def test_use_cudagraphs_dynamic(monkeypatch):
    assert vllm.envs.VLLM_USE_V1
    vllm_config = VllmConfig()
    assert vllm_config.compilation_config.use_cudagraph

    monkeypatch.setenv('VLLM_USE_V1', '0')
    vllm_config = VllmConfig()
    assert not vllm_config.compilation_config.use_cudagraph


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
