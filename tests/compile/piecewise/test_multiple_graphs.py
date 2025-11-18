# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test (piecewise) compilation with a simple model where multiple submodules
are compiled and graph captured separately.
"""

import pytest
import torch
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer

# This import automatically registers `torch.ops.silly.attention`
from .. import silly_attention  # noqa: F401

BATCH_SIZE = 32
MLP_SIZE = 128
HIDDEN_SIZE = 1024
RANDOM_SEED = 0


@support_torch_compile
class ParentModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Attention(nn.Module):
    def __init__(self, mlp_size: int, hidden_size: int) -> None:
        super().__init__()
        self.pre_attn = nn.Linear(mlp_size, hidden_size, bias=False)
        self.post_attn = nn.Linear(hidden_size, mlp_size, bias=False)
        self.rms_norm_weight = nn.Parameter(torch.ones(hidden_size))

        # Initialize to same weights for testing
        nn.init.xavier_normal_(
            self.pre_attn.weight.data,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
            gain=0.001,
        )
        nn.init.xavier_normal_(
            self.post_attn.weight.data,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
            gain=0.001,
        )

    def rms_norm_ref(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        return (
            x_f32
            * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
            * self.rms_norm_weight
        ).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_attn(x)
        x = self.rms_norm_ref(x)
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output
        x = self.rms_norm_ref(x)
        x = self.post_attn(x)
        return x


@support_torch_compile
class CompiledAttention(nn.Module):
    def __init__(
        self,
        *,
        mlp_size: int,
        hidden_size: int,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn = Attention(mlp_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


@support_torch_compile
class CompiledAttentionTwo(CompiledAttention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x) + x


@ignore_torch_compile
class SimpleModelWithTwoGraphs(ParentModel):
    def __init__(
        self,
        *,
        mlp_size: int,
        hidden_size: int,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Test will fail without set_model_tag here with error:
        # "ValueError: too many values to unpack (expected 3)"
        # This is because CompiledAttention and CompiledAttentionTwo
        # have different implementations but the same torch.compile
        # cache dir will be used as default prefix is 'model_tag'
        with set_model_tag("attn_one"):
            self.attn_one = CompiledAttention(
                mlp_size=mlp_size,
                hidden_size=hidden_size,
                vllm_config=vllm_config,
                prefix=f"{prefix}.attn_one",
            )
        with set_model_tag("attn_two"):
            self.attn_two = CompiledAttentionTwo(
                mlp_size=mlp_size,
                hidden_size=hidden_size,
                vllm_config=vllm_config,
                prefix=f"{prefix}.attn_two",
            )

        self.hidden_states = torch.zeros((BATCH_SIZE, MLP_SIZE)).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # CUDAGraph expects same tensor addresses for each run
        self.hidden_states[:bsz].copy_(x)
        x = self.attn_one(self.hidden_states[:bsz])
        self.hidden_states[:bsz].copy_(x)
        x = self.attn_two(self.hidden_states[:bsz])
        return x


@torch.inference_mode
def run_model(
    vllm_config: VllmConfig,
    model: nn.Module,
    inputs: torch.Tensor,
    cudagraph_runtime_mode: CUDAGraphMode,
):
    with set_forward_context({}, vllm_config=vllm_config):
        # warmup for the model with cudagraph_mode NONE
        model(inputs)

        # simulate cudagraphs capturing
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            model(inputs[:2])
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=1,
            ),
        ):
            model(inputs[:1])

        # simulate cudagraphs replay
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            output = model(inputs[:2])

        output = output.cpu()
        return output.cpu()


@pytest.mark.parametrize("use_inductor_graph_partition", [False, True])
def test_multi_graph_piecewise_compile(use_inductor_graph_partition: bool):
    if use_inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("inductor graph partition is only available in PyTorch 2.9+")

    outputs = []

    # vllmcompile compile
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            splitting_ops=["silly::attention"],
            cudagraph_capture_sizes=[1, 2],
            use_inductor_graph_partition=use_inductor_graph_partition,
        )
    )
    cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE

    with set_current_vllm_config(vllm_config):
        model = (
            SimpleModelWithTwoGraphs(
                mlp_size=MLP_SIZE,
                hidden_size=HIDDEN_SIZE,
                vllm_config=vllm_config,
                prefix="",
            )
            .eval()
            .cuda()
        )

    # Pre-allocate memory for CUDAGraph which expects
    # static tensor addresses
    inputs = torch.randn(BATCH_SIZE, MLP_SIZE).cuda()

    if use_inductor_graph_partition:
        # Splitting happens at Inductor lowering level,
        # total piecewise fx graphs is equal to total graphs
        num_piecewise_fx = 2
        num_piecewise_capturable_fx = 2
    else:
        # attn_one, attn_two each has 3 piecewise graphs
        # (pre attn, post attn, silly_attention) each
        num_piecewise_fx = 6
        # attn_one, attn_two has pre attn and post attn each, total=4
        num_piecewise_capturable_fx = 4

    with compilation_counter.expect(
        num_graphs_seen=2,  # two graphs for the model
        num_piecewise_graphs_seen=num_piecewise_fx,
        num_piecewise_capturable_graphs_seen=num_piecewise_capturable_fx,
        num_backend_compilations=num_piecewise_capturable_fx,
        num_cudagraph_captured=8,  # num_cudagraph_sizes * num_partitions
    ):
        outputs.append(run_model(vllm_config, model, inputs, cudagraph_runtime_mode))

    # no compile or cudagraph
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
        )
    )
    cudagraph_runtime_mode = CUDAGraphMode.NONE

    with set_current_vllm_config(vllm_config):
        model = (
            SimpleModelWithTwoGraphs(
                mlp_size=MLP_SIZE,
                hidden_size=HIDDEN_SIZE,
                vllm_config=vllm_config,
                prefix="",
            )
            .eval()
            .cuda()
        )

    with compilation_counter.expect(
        num_graphs_seen=0,
        num_piecewise_graphs_seen=0,
        num_piecewise_capturable_graphs_seen=0,
        num_backend_compilations=0,
        num_cudagraph_captured=0,
    ):
        outputs.append(run_model(vllm_config, model, inputs, cudagraph_runtime_mode))

    # piecewise compile without CUDA graph
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            splitting_ops=["silly::attention"],
            use_inductor_graph_partition=use_inductor_graph_partition,
        )
    )
    cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE

    with set_current_vllm_config(vllm_config):
        model = (
            SimpleModelWithTwoGraphs(
                mlp_size=MLP_SIZE,
                hidden_size=HIDDEN_SIZE,
                vllm_config=vllm_config,
                prefix="",
            )
            .eval()
            .cuda()
        )

    with compilation_counter.expect(
        num_graphs_seen=2,
        num_piecewise_graphs_seen=num_piecewise_fx,
        num_piecewise_capturable_graphs_seen=num_piecewise_capturable_fx,
        num_backend_compilations=num_piecewise_capturable_fx,
        num_cudagraph_captured=0,  # no cudagraph captured
    ):
        outputs.append(run_model(vllm_config, model, inputs, cudagraph_runtime_mode))

    # Generally don't expect outputs with and without inductor
    # to be bitwise equivalent
    assert torch.allclose(outputs[0], outputs[1])

    # Expect bitwise equivalence using inductor w/ and w/o cudagraph
    assert torch.equal(outputs[0], outputs[2])
