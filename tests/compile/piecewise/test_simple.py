"""
Test the piecewise compilation with a simple model so that we
can exactly calculate the expected output and side effects.
"""

import torch
from torch import nn
from torch.library import Library

from vllm.compilation.compile_context import set_compile_context
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CompilationConfig, CompilationLevel, VllmConfig
from vllm.plugins import set_current_vllm_config
from vllm.utils import direct_register_custom_op

global_counter = 0

# create a library to hold the custom op
silly_lib = Library("silly", "FRAGMENT")  # noqa


def silly_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor) -> None:
    global global_counter
    global_counter += 1
    print(f"{global_counter=}")
    out.copy_(q)
    out[0] += 1


def silly_attention_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         out: torch.Tensor) -> None:
    return


direct_register_custom_op(
    op_name="attention",
    op_func=silly_attention,
    mutates_args=["out"],
    fake_impl=silly_attention_fake,
    target_lib=silly_lib,
)


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
        x += 1
        x[0] += 2
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


def test_simple_piecewise_compile():

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=True,
        splitting_ops=["silly.attention"],
        cudagraph_copy_inputs=True,
    ))
    with set_current_vllm_config(vllm_config):
        model = SillyModel(vllm_config=vllm_config, prefix='')

    inputs = torch.randn(100).cuda()

    with compilation_counter.expect(
            num_graphs_seen=1,  # one graph for the model
            num_piecewise_graphs_seen=5,  # 2 * num_layers + 1
            num_piecewise_capturable_graphs_seen=3,  # 1 + num_layers
            num_inductor_compilations=3,  # num_piecewise_capturable_graphs_seen
            num_cudagraph_caputured=
            6,  # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ):

        with set_compile_context([1, 2]):
            model(inputs)

            model(torch.randn(2).cuda())
            model(torch.randn(1).cuda())

        input = torch.zeros(2).cuda()
        global global_counter
        global_counter = 0
        output = model(input)
        assert global_counter == 2
        assert torch.allclose(output.cpu(), torch.tensor([3., 1.]))
