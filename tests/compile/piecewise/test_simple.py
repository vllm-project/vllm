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
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ...utils import create_new_process_for_each_test

# This import automatically registers `torch.ops.silly.attention`
from ..silly_attention import get_global_counter, reset_global_counter


@support_torch_compile
class SillyModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
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


def _run_simple_model(
    splitting_ops,
    use_inductor_graph_partition,
    use_inductor,
    expected_num_piecewise_graphs_seen,
    expected_num_piecewise_capturable_graphs_seen,
    expected_num_backend_compilations,
    expected_num_cudagraph_captured,
):
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor=use_inductor,
            splitting_ops=splitting_ops,
            use_inductor_graph_partition=use_inductor_graph_partition,
            cudagraph_copy_inputs=True,
            cudagraph_capture_sizes=[1, 2],
        )
    )
    with set_current_vllm_config(vllm_config):
        model = SillyModel(vllm_config=vllm_config, prefix="")

    inputs = torch.randn(100).cuda()

    with (
        compilation_counter.expect(
            num_graphs_seen=1,  # one graph for the model
            num_piecewise_graphs_seen=expected_num_piecewise_graphs_seen,
            num_piecewise_capturable_graphs_seen=expected_num_piecewise_capturable_graphs_seen,
            num_backend_compilations=expected_num_backend_compilations,
            num_cudagraph_captured=expected_num_cudagraph_captured,
        ),
        set_forward_context(None, vllm_config=vllm_config),
    ):  # background context
        # warm up with background context
        model(inputs)

        # capturing/replaying should under context of cudagraph dispatching
        with set_forward_context(
            None,
            vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            model(torch.randn(2).cuda())
        with set_forward_context(
            None,
            vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(
                num_tokens=1,
            ),
        ):
            model(torch.randn(1).cuda())

        input = torch.zeros(2).cuda()
        reset_global_counter()
        with set_forward_context(
            None,
            vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            output = model(input)
        assert get_global_counter() == 2
        assert torch.allclose(output.cpu(), torch.tensor([19.0, 19.0]))


@pytest.mark.parametrize("use_inductor", [True, False])
@torch.inference_mode()
@create_new_process_for_each_test("spawn")
def test_simple_piecewise_compile(use_inductor):
    _run_simple_model(
        splitting_ops=["silly::attention"],
        use_inductor_graph_partition=False,
        use_inductor=use_inductor,
        # 2 * num_layers + 1
        expected_num_piecewise_graphs_seen=5,
        # 1 + num_layers
        expected_num_piecewise_capturable_graphs_seen=3,
        # num_piecewise_capturable_graphs_seen
        expected_num_backend_compilations=3,
        # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
        expected_num_cudagraph_captured=6,
    )


@torch.inference_mode()
def test_simple_inductor_graph_partition(monkeypatch):
    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("inductor graph partition is only available in PyTorch 2.9+")

    # disable compile cache so that we run separately for different splitting_ops
    # and get the expected number of cudagraphs captured.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    _run_simple_model(
        splitting_ops=["silly::attention"],
        use_inductor_graph_partition=True,
        use_inductor=True,
        # Since not splitting at fx graph level
        expected_num_piecewise_graphs_seen=1,
        # Since not splitting at fx graph level
        expected_num_piecewise_capturable_graphs_seen=1,
        # Since not splitting at fx graph level
        expected_num_backend_compilations=1,
        # Inductor graph partition still captures 6 graph, same as fx graph partition
        expected_num_cudagraph_captured=6,
    )
