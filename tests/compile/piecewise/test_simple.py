# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test the piecewise compilation with a simple model so that we
can exactly calculate the expected output and side effects.
"""

import pytest
import torch
from torch import nn

from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, CUDAGraphMode,
                         VllmConfig, set_current_vllm_config)
from vllm.envs import VLLM_USE_V1
from vllm.forward_context import BatchDescriptor, set_forward_context

# This import automatically registers `torch.ops.silly.attention`
from ..silly_attention import get_global_counter, reset_global_counter


@support_torch_compile
class SillyModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overall effect:
        x = 3 * x + 19
        global_counter += 2
        """
        x = x + 1
        x = x + 2
        out = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, out)
        x = out
        x = x - 2
        x = x - 1
        out = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, out)
        x = out
        x = x + 1
        return x


@pytest.mark.parametrize("use_inductor", [True, False])
@torch.inference_mode()
def test_simple_piecewise_compile(use_inductor):
    assert VLLM_USE_V1

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=True,
        use_inductor=use_inductor,
        splitting_ops=["silly.attention"],
        cudagraph_copy_inputs=True,
        cudagraph_capture_sizes=[1, 2],
    ))
    with set_current_vllm_config(vllm_config):
        model = SillyModel(vllm_config=vllm_config, prefix='')

    inputs = torch.randn(100).cuda()

    with compilation_counter.expect(
            num_graphs_seen=1,  # one graph for the model
            num_piecewise_graphs_seen=5,  # 2 * num_layers + 1
            num_piecewise_capturable_graphs_seen=3,  # 1 + num_layers
            num_backend_compilations=3,  # num_piecewise_capturable_graphs_seen
            num_cudagraph_captured=
            6,  # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ), set_forward_context(None,
                           vllm_config=vllm_config):  # background context
        # warm up with background context
        model(inputs)

        # capturing/replaying should under context of cudagraph dispatching
        with set_forward_context(
                None,
                vllm_config=vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=BatchDescriptor(num_tokens=2, )):
            model(torch.randn(2).cuda())
        with set_forward_context(
                None,
                vllm_config=vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=BatchDescriptor(num_tokens=1, )):
            model(torch.randn(1).cuda())

        input = torch.zeros(2).cuda()
        reset_global_counter()
        with set_forward_context(
                None,
                vllm_config=vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=BatchDescriptor(num_tokens=2, )):
            output = model(input)
        assert get_global_counter() == 2
        assert torch.allclose(output.cpu(), torch.tensor([19.0, 19.0]))
