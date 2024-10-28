"""
Test the piecewise compilation with a simple model, comparing the output
with and without the piecewise compilation.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from vllm.compilation.compile_context import set_compile_context
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.levels import CompilationLevel


@torch.library.custom_op("silly::attention", mutates_args=["out"])
def silly_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor) -> None:
    out.copy_(q)
    out += k
    out += v


@silly_attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
      out: torch.Tensor) -> None:
    return


@dataclass
class LlamaConfig:
    hidden_size: int = 128
    mlp_size: int = 256
    vocab_size: int = 128
    num_layers: int = 2


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

        self.gate_up_projection.weight.data.fill_(0.0)
        self.down_projection.weight.data.fill_(0.0)

    def forward(self, x):
        x = self.gate_up_projection(x)
        x = x[:, :x.size(1) // 2] * torch.nn.functional.relu(
            x[:, x.size(1) // 2:])
        x = self.down_projection(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.qkv_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
        )

        self.output_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
        )

        self.qkv_projection.weight.data.fill_(0.0)
        self.output_projection.weight.data.fill_(0.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
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
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = hidden_states / 2
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = hidden_states / 2

        hidden_states = self.self_attention(positions=positions,
                                            hidden_states=hidden_states)

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = hidden_states / 2
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embedding_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_layers)])

        self.embedding_tokens.weight.data.fill_(0.0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embedding_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        return hidden_states


directory = os.path.dirname(__file__)
config = os.path.join(directory, "compilation_config.json")


@torch.inference_mode
def run_model(use_compile: bool, split_attn: bool = False) -> torch.Tensor:

    os.environ["VLLM_TORCH_COMPILE_CONFIG"] = config

    if use_compile:
        os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(
            CompilationLevel.PIECEWISE)
    else:
        os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(
            CompilationLevel.NO_COMPILATION)

    if use_compile and split_attn:
        from vllm.plugins import set_non_cudagraph_ops
        set_non_cudagraph_ops(["silly.attention"])

    cls = LlamaModel
    if use_compile:
        cls = support_torch_compile(LlamaModel)
    llama_config = LlamaConfig(hidden_size=128,
                               mlp_size=256,
                               vocab_size=128,
                               num_layers=2)
    model = cls(llama_config).eval().cuda()

    B = 16  # max batch size
    input_ids = torch.randint(0, llama_config.vocab_size, (B, )).cuda()
    positions = torch.arange(B).cuda()

    with set_compile_context([1, 2]):
        model(input_ids, positions)
        model(input_ids[:2], positions[:2])
        model(input_ids[:1], positions[:1])

    input_ids[:2].zero_()
    output = model(input_ids[:2], positions[:2])

    return output.cpu()


def test_toy_llama():
    # compare output with and without piecewise compilation

    outputs = []
    outputs.append(run_model(use_compile=False))
    outputs.append(run_model(use_compile=True))
    outputs.append(run_model(use_compile=True, split_attn=True))

    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i])


@torch.inference_mode
def benchmark():
    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(CompilationLevel.PIECEWISE)
    from triton.testing import do_bench
    cls = support_torch_compile(LlamaModel)

    # similar to llama 3.1-8B
    llama_config = LlamaConfig(hidden_size=4096,
                               mlp_size=14336,
                               vocab_size=128 * 1024,
                               num_layers=32)

    # a tiny model to measure the overhead
    # of piecewise cudagraph
    llama_config = LlamaConfig(hidden_size=40,
                               mlp_size=80,
                               vocab_size=128,
                               num_layers=2)

    cudagraph_sizes = [1, 2, 4] + [i * 8 for i in range(1, 33)]

    full_cudagraph_time = {}
    piecewise_cudagraph_time = {}

    pool = torch.cuda.graph_pool_handle()

    for piecewise in [False, True]:
        if piecewise:
            os.environ["VLLM_TORCH_COMPILE_CONFIG"] = config
            from vllm.plugins import set_non_cudagraph_ops
            set_non_cudagraph_ops(["silly.attention"])

        model = cls(llama_config).eval().cuda().to(torch.bfloat16)

        B = 256  # max batch size
        input_ids = torch.randint(0, llama_config.vocab_size, (B, )).cuda()
        positions = torch.arange(B).cuda().to(torch.bfloat16)

        graphs = {}

        with set_compile_context(cudagraph_sizes):
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
                runtime = do_bench(lambda: graphs[b][0]  # noqa
                                   (input_ids[:b], positions[:b]))  # noqa
                piecewise_cudagraph_time[b] = runtime
            else:
                runtime = do_bench(lambda: graphs[b][0].replay())  # noqa
                full_cudagraph_time[b] = runtime

    # print in tabular format
    print("batch size\tfull cudagraph\tpiecewise cudagraph")
    for b in cudagraph_sizes:
        print(f"{b}\t{full_cudagraph_time[b]}\t{piecewise_cudagraph_time[b]}")


if __name__ == "__main__":
    benchmark()
