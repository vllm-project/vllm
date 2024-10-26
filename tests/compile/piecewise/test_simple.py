import os

import torch
from torch import nn

from vllm.compilation.compile_context import set_compile_context
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.levels import CompilationLevel
from vllm.plugins import set_non_cudagraph_ops

set_non_cudagraph_ops(["silly.attention"])

global_counter = 0


@torch.library.custom_op("silly::attention", mutates_args=["out"])
def silly_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor) -> None:
    global global_counter
    global_counter += 1
    print(f"{global_counter=}")
    out.copy_(q)
    out[0] += 1


@silly_attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
      out: torch.Tensor) -> None:
    return


@support_torch_compile
class SillyModel(nn.Module):

    def __init__(self) -> None:
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

    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(CompilationLevel.PIECEWISE)

    model = SillyModel()

    input_buffer = torch.randn(100).cuda()

    with set_compile_context([1, 2]):
        model(input_buffer)

        model(input_buffer[:2])
        model(input_buffer[:1])

    input_buffer[:2].zero_()
    global global_counter
    global_counter = 0
    output = model(input_buffer[:2])
    assert global_counter == 2
    assert torch.allclose(output.cpu(), torch.tensor([3., 1.]))
