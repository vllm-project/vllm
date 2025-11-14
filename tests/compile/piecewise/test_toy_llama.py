# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test the piecewise compilation with a simple model, comparing the output
with and without the piecewise compilation.

This is a tractable model, the weights and computation are specially designed
if the config `tractable_init` is set to True. Otherwise, the weights are
initialized randomly with a fixed seed.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

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
from .. import silly_attention  # noqa: F401


@dataclass
class LlamaConfig:
    hidden_size: int = 128
    mlp_size: int = 256
    vocab_size: int = 128
    num_layers: int = 2
    init_value: float = 1.0
    tractable_init: bool = False
    random_seed: int = 0

    def compute_hash(self) -> str:
        factors: list[Any] = []
        for k, v in self.__dict__.items():
            if k == "random_seed":
                continue
            factors.append((k, v))
        factors.sort()
        import hashlib

        return hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()

    def __post_init__(self):
        assert self.mlp_size >= self.hidden_size


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_up_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.mlp_size * 2,
            bias=False,
        )
        self.down_projection = nn.Linear(
            in_features=config.mlp_size,
            out_features=config.hidden_size,
            bias=False,
        )

        if config.tractable_init:
            nn.init.eye_(self.gate_up_projection.weight.data[: config.mlp_size])
            nn.init.eye_(self.gate_up_projection.weight.data[config.mlp_size :])
            nn.init.eye_(self.down_projection.weight.data)
        else:
            nn.init.xavier_normal_(
                self.gate_up_projection.weight.data,
                generator=torch.Generator().manual_seed(config.random_seed),
                gain=0.001,
            )
            nn.init.xavier_normal_(
                self.down_projection.weight.data,
                generator=torch.Generator().manual_seed(config.random_seed),
                gain=0.001,
            )

    def forward(self, x):
        # for tractable_init and positive input, this is
        # essentially an elementwise-square
        x = self.gate_up_projection(x)
        x = x[:, : x.size(1) // 2] * torch.nn.functional.relu(x[:, x.size(1) // 2 :])
        x = self.down_projection(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.qkv_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
            bias=False,
        )

        self.output_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=False,
        )

        if config.tractable_init:
            nn.init.eye_(self.qkv_projection.weight.data[: config.hidden_size])
            nn.init.eye_(
                self.qkv_projection.weight.data[
                    config.hidden_size : 2 * config.hidden_size
                ]
            )
            nn.init.eye_(self.qkv_projection.weight.data[2 * config.hidden_size :])
            nn.init.eye_(self.output_projection.weight.data)
        else:
            nn.init.xavier_normal_(
                self.qkv_projection.weight.data,
                generator=torch.Generator().manual_seed(config.random_seed),
                gain=0.001,
            )
            nn.init.xavier_normal_(
                self.output_projection.weight.data,
                generator=torch.Generator().manual_seed(config.random_seed),
                gain=0.001,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # for tractable_init, this is:
        # output = (hidden_states * 3 + positions * 2)
        qkv = self.qkv_projection(hidden_states)
        hidden_size = qkv.size(-1) // 3
        q, k, v = qkv.split([hidden_size, hidden_size, hidden_size], dim=-1)

        q = q + positions.unsqueeze(1)
        k = k + positions.unsqueeze(1)

        attn_output = torch.empty_like(q)
        torch.ops.silly.attention(q, k, v, attn_output)

        output = self.output_projection(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attention = LlamaAttention(config)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For tractable computation:
        - if residual is None, the outputs are:
            - residual = (hidden_states + 1) * 3 + positions * 2 + hidden_states = hidden_states * 4 + positions * 2 + 3
            - hidden_states = (residual + 1) ** 2
        - if residual is not None, the outputs are:
            - residual = (hidden_states + residual + 1) * 3 + positions * 2 + hidden_states + residual = (hidden_states + residual) * 4 + positions * 2 + 3
            - hidden_states = (residual + 1) ** 2
        """  # noqa
        if residual is None:
            residual = hidden_states
            hidden_states = hidden_states + 1
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = hidden_states + 1

        hidden_states = self.self_attention(
            positions=positions, hidden_states=hidden_states
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = hidden_states + 1
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: LlamaConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedding_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_layers)]
        )

        # this is the initial value of the hidden states
        self.embedding_tokens.weight.data.fill_(config.init_value)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embedding_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        return hidden_states


def tractable_computation(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    config: LlamaConfig,
    init_value: float = 1.0,
) -> torch.Tensor:
    hidden_states = (
        torch.ones(
            input_ids.size(0),
            config.hidden_size,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        * init_value
    )

    # first layer
    residual = hidden_states * 4 + positions.unsqueeze(1) * 2 + 3
    hidden_states = (residual + 1) ** 2

    # following layers
    for _ in range(config.num_layers - 1):
        hidden_states = hidden_states + residual
        residual = hidden_states * 4 + positions.unsqueeze(1) * 2 + 3
        hidden_states = (residual + 1) ** 2

    return hidden_states


@torch.inference_mode
def run_model(llama_config, compile_config: CompilationConfig) -> torch.Tensor:
    # Start with a fresh copy to make sure there's no cache dir sharing
    compile_config = deepcopy(compile_config)
    cudagraph_runtime_mode = compile_config.cudagraph_mode

    vllm_config = VllmConfig(
        compilation_config=compile_config, additional_config=llama_config
    )
    with set_current_vllm_config(vllm_config):
        model = (
            LlamaModel(config=llama_config, vllm_config=vllm_config, prefix="")
            .eval()
            .cuda()
        )

    with set_forward_context({}, vllm_config=vllm_config):  # background context
        B = 16  # max batch size
        input_ids = torch.randint(0, llama_config.vocab_size, (B,)).cuda()
        positions = torch.arange(B).cuda()

        # warmup for the model with cudagraph_mode NONE
        model(input_ids, positions)

        # simulate cudagraphs capturing
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            model(input_ids[:2], positions[:2])
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=1,
            ),
        ):
            model(input_ids[:1], positions[:1])

        input_ids[:2].zero_()
        # simulate cudagraphs replay
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            output = model(input_ids[:2], positions[:2])

        output = output.cpu()

        if llama_config.tractable_init:
            expected_output = tractable_computation(
                input_ids[:2], positions[:2], llama_config
            ).cpu()

            assert torch.allclose(output, expected_output)
        else:
            return output.cpu()


@pytest.mark.parametrize(
    "backend, use_inductor_graph_partition",
    [
        ("eager", False),  # No inductor
        ("inductor", False),  # Inductor, Dynamo partition
        ("inductor", True),  # Inductor, Inductor partition
    ],
)
@create_new_process_for_each_test("spawn")
def test_toy_llama(
    backend: str, use_inductor_graph_partition: bool, monkeypatch, tmp_path
):
    # We disable the vLLM compile cache into a new tmp dir for 1 reason:
    # 1. To make sure we can properly track the number of Inductor compilations.
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    if use_inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("Inductor graph partition only supported in torch>=2.9")

    # compare output with and without piecewise compilation

    llama_config = LlamaConfig(
        hidden_size=128, mlp_size=256, vocab_size=128, num_layers=12
    )

    tractable_config = LlamaConfig(
        hidden_size=128, mlp_size=256, vocab_size=128, num_layers=2, tractable_init=True
    )

    compile_config_no_compile = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        backend="eager",
    )

    compile_config_no_split = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        use_inductor_graph_partition=use_inductor_graph_partition,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        backend=backend,
        cudagraph_capture_sizes=[1, 2],
    )

    compile_config_split = deepcopy(compile_config_no_split)
    compile_config_split.splitting_ops = ["silly::attention"]

    outputs = []
    with compilation_counter.expect(
        num_graphs_seen=0,
        num_piecewise_graphs_seen=0,
        num_piecewise_capturable_graphs_seen=0,
        num_backend_compilations=0,
        num_cudagraph_captured=0,
    ):
        outputs.append(run_model(llama_config, compile_config_no_compile))

    run_model(tractable_config, compile_config_no_compile)

    if backend == "inductor":
        kwargs = {"num_inductor_compiles": 1, "num_eager_compiles": 0}
    else:
        kwargs = {"num_eager_compiles": 1, "num_inductor_compiles": 0}

    with compilation_counter.expect(
        num_graphs_seen=1,  # one graph for the model
        num_piecewise_graphs_seen=1,
        num_piecewise_capturable_graphs_seen=1,
        num_backend_compilations=1,  # num_piecewise_capturable_graphs_seen
        num_cudagraph_captured=2,
        **kwargs,
    ):
        outputs.append(run_model(llama_config, compile_config_no_split))

    run_model(tractable_config, compile_config_no_split)

    if use_inductor_graph_partition:
        num_piecewise_fx = 1
        num_piecewise_capturable_fx = 1
    else:
        num_piecewise_fx = 2 * llama_config.num_layers + 1
        num_piecewise_capturable_fx = 1 + llama_config.num_layers

    with compilation_counter.expect(
        num_graphs_seen=1,  # one graph for the model
        num_piecewise_graphs_seen=num_piecewise_fx,
        num_piecewise_capturable_graphs_seen=num_piecewise_capturable_fx,
        num_backend_compilations=num_piecewise_capturable_fx,
        # num_cudagraph_sizes * num_partitions
        num_cudagraph_captured=2 * (1 + llama_config.num_layers),
    ):
        outputs.append(run_model(llama_config, compile_config_split))
    run_model(tractable_config, compile_config_split)

    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i])


@torch.inference_mode
def benchmark():
    from triton.testing import do_bench

    # similar to llama 3.1-8B
    llama_config = LlamaConfig(
        hidden_size=4096, mlp_size=14336, vocab_size=128 * 1024, num_layers=32
    )

    # a tiny model to measure the overhead
    # of piecewise cudagraph
    llama_config = LlamaConfig(
        hidden_size=40, mlp_size=80, vocab_size=128, num_layers=2
    )

    cudagraph_sizes = [1, 2, 4] + [i * 8 for i in range(1, 33)]

    eager_time = {}
    full_cudagraph_time = {}
    piecewise_cudagraph_time = {}

    pool = torch.cuda.graph_pool_handle()

    for piecewise in [False, True]:
        if piecewise:
            compilation_config = CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                splitting_ops=["silly::attention"],
                cudagraph_capture_sizes=cudagraph_sizes,
            )
        else:
            compilation_config = CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_capture_sizes=cudagraph_sizes,
            )

        vllm_config = VllmConfig(compilation_config=compilation_config)
        with set_current_vllm_config(vllm_config):
            model = (
                LlamaModel(config=llama_config, vllm_config=vllm_config, prefix="")
                .eval()
                .cuda()
                .to(torch.bfloat16)
            )

        B = 256  # max batch size
        input_ids = torch.randint(0, llama_config.vocab_size, (B,)).cuda()
        positions = torch.arange(B).cuda().to(torch.bfloat16)

        graphs = {}

        model(input_ids, positions)
        for b in cudagraph_sizes[::-1]:
            if not piecewise:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    output = model(input_ids[:b], positions[:b])
                graphs[b] = (graph, output)
            else:
                output = model(input_ids[:b], positions[:b])
                graphs[b] = (model, output)
        for b in cudagraph_sizes:
            if piecewise:
                # noqa is for `Function definition does not bind loop variable`
                # it will be problematic if we save the created lambda function
                # and use it later, because it will look up the name `b` in the
                # enclosing scope, and the value of `b` will always be 256.
                # it is fine here, because we only use the lambda function once.
                runtime = do_bench(
                    lambda: graphs[b][0](  # noqa
                        input_ids[:b],  # noqa
                        positions[:b],  # noqa
                    )
                )
                piecewise_cudagraph_time[b] = runtime
            else:
                runtime = do_bench(lambda: graphs[b][0].replay())  # noqa
                eager_runtime = do_bench(lambda: model(input_ids[:b], positions[:b]))  # noqa
                full_cudagraph_time[b] = runtime
                eager_time[b] = eager_runtime

    # print in tabular format
    print("batch size\teager mode\tfull cudagraph\tpiecewise cudagraph")
    for b in cudagraph_sizes:
        print(
            f"{b}\t{eager_time[b]:.3f}\t{full_cudagraph_time[b]:.3f}"
            f"\t{piecewise_cudagraph_time[b]:.3f}"
        )


if __name__ == "__main__":
    # Protect against subprocess reimport when using spawn_new_process_for_each_test
    import os

    if os.environ.get("RUNNING_IN_SUBPROCESS") != "1":
        benchmark()
