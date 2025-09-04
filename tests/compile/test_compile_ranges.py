# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn
from torch.library import Library

from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)
from vllm.forward_context import set_forward_context
from vllm.utils import direct_register_custom_op

# create a library to hold the custom op
silly_lib = Library("silly", "FRAGMENT")  # noqa

BATCH_SIZE = 64
MLP_SIZE = 128


def silly_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor) -> None:
    out.copy_(q)
    out += k
    out += v


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
class TestModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output
        x = x * 3
        return x


@torch.inference_mode
def run_model(vllm_config: VllmConfig, model: nn.Module,
              batch_sizes: list[int]):
    with set_forward_context({}, vllm_config=vllm_config):
        model(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())
        for batch_size in batch_sizes:
            model(torch.randn(batch_size, MLP_SIZE).cuda())


def test_compile_ranges():
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        compile_ranges_split_points=[8, 32],
    ))

    with set_current_vllm_config(vllm_config):
        model = TestModel(vllm_config=vllm_config, prefix='').eval().cuda()
    batch_sizes = [1, 16, 48]
    # A has support_torch_compile
    with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=1,
            num_backend_compilations=4,
            # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ):
        run_model(vllm_config, model, batch_sizes)
