# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import fx as fx
from torch import nn
from torch.library import Library

from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.inductor_pass import (
    CustomGraphPass,
    InductorPass,
    get_pass_context,
)
from vllm.config import (
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.config.scheduler import SchedulerConfig
from vllm.forward_context import set_forward_context

# create a library to hold the custom op
silly_lib = Library("silly", "FRAGMENT")  # noqa

BATCH_SIZE = 64
MLP_SIZE = 128


@support_torch_compile
class TestModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output
        x = x * 3
        return x


@torch.inference_mode
def run_model(vllm_config: VllmConfig, model: nn.Module, batch_sizes: list[int]):
    with set_forward_context({}, vllm_config=vllm_config):
        model(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())
        for batch_size in batch_sizes:
            model(torch.randn(batch_size, MLP_SIZE).cuda())


class PostGradPassManagerCheckRanges(CustomGraphPass):
    def __init__(self, ranges: list[tuple[int, int]]):
        self.ranges = ranges

    def __call__(self, graph: fx.Graph):
        compile_range = get_pass_context().compile_range
        assert compile_range in self.ranges, (
            f"Compile range {compile_range} not in {self.ranges}"
        )

    def uuid(self) -> str:
        state = {
            "ranges": self.ranges,
        }
        return InductorPass.hash_dict(state)


def test_compile_ranges():
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            compile_ranges_split_points=[8, 32],
        ),
        inductor_compile_config={
            "post_grad_custom_post_pass": PostGradPassManagerCheckRanges(
                [(1, 8), (8, 32), (32, 2049)]
            )
        },
    )

    with set_current_vllm_config(vllm_config):
        model = TestModel(vllm_config=vllm_config, prefix="").eval().cuda()
    batch_sizes = [1, 16, 48]
    # A has support_torch_compile
    with compilation_counter.expect(
        num_graphs_seen=1,
        num_piecewise_graphs_seen=1,
        num_backend_compilations=3,
        # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ):
        run_model(vllm_config, model, batch_sizes)
