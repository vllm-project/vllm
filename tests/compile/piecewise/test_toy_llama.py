"""
Test the piecewise compilation with a simple model, comparing the output
with and without the piecewise compilation.

This is a tractable model, the weights and computation are specially designed
if the config `tractable_init` is set to True. Otherwise, the weights are
initialized randomly with a fixed seed.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from torch import nn
from torch.library import Library

from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)
from vllm.utils import direct_register_custom_op

# create a library to hold the custom op
silly_lib = Library("silly", "FRAGMENT")  # noqa


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
        factors: List[Any] = []
        for k, v in self.__dict__.items():
            if k == "random_seed":
                continue
            factors.append((k, v))
        factors.sort()
        import hashlib
        return hashlib.md5(str(factors).encode()).hexdigest()

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
            nn.init.eye_(self.gate_up_projection.weight.data[:config.mlp_size])
            nn.init.eye_(self.gate_up_projection.weight.data[config.mlp_size:])
            nn.init.eye_(self.down_projection.weight.data)
        else:
            nn.init.xavier_normal_(self.gate_up_projection.weight.data,
                                   generator=torch.Generator().manual_seed(
                                       config.random_seed),
                                   gain=0.001)
            nn.init.xavier_normal_(self.down_projection.weight.data,
                                   generator=torch.Generator().manual_seed(
                                       config.random_seed),
                                   gain=0.001)

    def forward(self, x):
        # for tractable_init and positive input, this is
        # essentially an elementwise-square
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
            bias=False,
        )

        self.output_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=False,
        )

        if config.tractable_init:
            nn.init.eye_(self.qkv_projection.weight.data[:config.hidden_size])
            nn.init.eye_(self.qkv_projection.weight.data[config.hidden_size:2 *
                                                         config.hidden_size])
            nn.init.eye_(self.qkv_projection.weight.data[2 *
                                                         config.hidden_size:])
            nn.init.eye_(self.output_projection.weight.data)
        else:
            nn.init.xavier_normal_(self.qkv_projection.weight.data,
                                   generator=torch.Generator().manual_seed(
                                       config.random_seed),
                                   gain=0.001)
            nn.init.xavier_normal_(self.output_projection.weight.data,
                                   generator=torch.Generator().manual_seed(
                                       config.random_seed),
                                   gain=0.001)

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
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For tractable computation:
        - if residual is None, the outputs are:
            - residual = (hidden_states + 1) * 3 + positions * 2 + hidden_states = hidden_states * 4 + positions * 2 + 3
            - hidden_states = (residual + 1) ** 2
        - if residual is not None, the outputs are:
            - residual = (hidden_states + residual + 1) * 3 + positions * 2 + hidden_states + residual = (hidden_states + residual) * 4 + positions * 2 + 3
            - hidden_states = (residual + 1) ** 2
        """ # noqa
        if residual is None:
            residual = hidden_states
            hidden_states = hidden_states + 1
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = hidden_states + 1

        hidden_states = self.self_attention(positions=positions,
                                            hidden_states=hidden_states)

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = hidden_states + 1
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 config: LlamaConfig,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__()
        self.embedding_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_layers)])

        # this is the initial value of the hidden states
        self.embedding_tokens.weight.data.fill_(config.init_value)

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


def tractable_computation(input_ids: torch.Tensor,
                          positions: torch.Tensor,
                          config: LlamaConfig,
                          init_value: float = 1.0) -> torch.Tensor:
    hidden_states = torch.ones(input_ids.size(0),
                               config.hidden_size,
                               device=input_ids.device,
                               dtype=input_ids.dtype) * init_value

    # first layer
    residual = hidden_states * 4 + positions.unsqueeze(1) * 2 + 3
    hidden_states = (residual + 1)**2

    # following layers
    for _ in range(config.num_layers - 1):
        hidden_states = hidden_states + residual
        residual = hidden_states * 4 + positions.unsqueeze(1) * 2 + 3
        hidden_states = (residual + 1)**2

    return hidden_states


@torch.inference_mode
def run_model(llama_config,
              use_compile: bool,
              split_attn: bool = False) -> torch.Tensor:

    if use_compile:
        compilation_config = CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            use_cudagraph=True,
            cudagraph_capture_sizes=[1, 2],
        )
        if split_attn:
            compilation_config.splitting_ops = ["silly.attention"]
    else:
        compilation_config = CompilationConfig(
            level=CompilationLevel.NO_COMPILATION, )

    vllm_config = VllmConfig(compilation_config=compilation_config,
                             additional_config=llama_config)
    with set_current_vllm_config(vllm_config):
        model = LlamaModel(config=llama_config,
                           vllm_config=vllm_config,
                           prefix="").eval().cuda()

    B = 16  # max batch size
    input_ids = torch.randint(0, llama_config.vocab_size, (B, )).cuda()
    positions = torch.arange(B).cuda()

    model(input_ids, positions)
    model(input_ids[:2], positions[:2])
    model(input_ids[:1], positions[:1])

    input_ids[:2].zero_()
    output = model(input_ids[:2], positions[:2])

    output = output.cpu()

    if llama_config.tractable_init:
        expected_output = tractable_computation(input_ids[:2], positions[:2],
                                                llama_config).cpu()

        assert torch.allclose(output, expected_output)
    else:
        return output.cpu()


def test_toy_llama():
    # compare output with and without piecewise compilation

    llama_config = LlamaConfig(hidden_size=128,
                               mlp_size=256,
                               vocab_size=128,
                               num_layers=12)

    tractable_config = LlamaConfig(hidden_size=128,
                                   mlp_size=256,
                                   vocab_size=128,
                                   num_layers=2,
                                   tractable_init=True)

    outputs = []
    with compilation_counter.expect(
            num_graphs_seen=0,
            num_piecewise_graphs_seen=0,
            num_piecewise_capturable_graphs_seen=0,
            num_inductor_compilations=0,
            num_cudagraph_caputured=0,
    ):
        outputs.append(run_model(llama_config, use_compile=False))
    run_model(tractable_config, use_compile=False)

    with compilation_counter.expect(
            num_graphs_seen=1,  # one graph for the model
            num_piecewise_graphs_seen=1,
            num_piecewise_capturable_graphs_seen=1,
            num_inductor_compilations=1,  # num_piecewise_capturable_graphs_seen
            num_cudagraph_caputured=
            2,  # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ):
        outputs.append(run_model(llama_config, use_compile=True))
    run_model(tractable_config, use_compile=True)

    with compilation_counter.expect(
            num_graphs_seen=1,  # one graph for the model
            num_piecewise_graphs_seen=2 * llama_config.num_layers +
            1,  # 2 * num_layers + 1
            num_piecewise_capturable_graphs_seen=1 +
            llama_config.num_layers,  # 1 + num_layers
            num_inductor_compilations=1 +
            llama_config.num_layers,  # num_piecewise_capturable_graphs_seen
            num_cudagraph_caputured=2 *
        (1 + llama_config.num_layers
         ),  # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ):
        outputs.append(
            run_model(llama_config, use_compile=True, split_attn=True))
    run_model(tractable_config, use_compile=True, split_attn=True)

    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i])


@torch.inference_mode
def benchmark():
    from triton.testing import do_bench

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

    eager_time = {}
    full_cudagraph_time = {}
    piecewise_cudagraph_time = {}

    pool = torch.cuda.graph_pool_handle()

    for piecewise in [False, True]:
        if piecewise:
            compilation_config = CompilationConfig(
                level=CompilationLevel.PIECEWISE,
                use_cudagraph=True,
                splitting_ops=["silly.attention"],
                cudagraph_capture_sizes=cudagraph_sizes,
            )
        else:
            compilation_config = CompilationConfig(
                level=CompilationLevel.PIECEWISE,
                cudagraph_capture_sizes=cudagraph_sizes,
            )

        vllm_config = VllmConfig(compilation_config=compilation_config)
        with set_current_vllm_config(vllm_config):
            model = LlamaModel(config=llama_config,
                               vllm_config=vllm_config,
                               prefix="").eval().cuda().to(torch.bfloat16)

        B = 256  # max batch size
        input_ids = torch.randint(0, llama_config.vocab_size, (B, )).cuda()
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
                runtime = do_bench(lambda: graphs[b][0]  # noqa
                                   (input_ids[:b], positions[:b]))  # noqa
                piecewise_cudagraph_time[b] = runtime
            else:
                runtime = do_bench(lambda: graphs[b][0].replay())  # noqa
                eager_runtime = do_bench(
                    lambda: model(input_ids[:b], positions[:b]))  # noqa
                full_cudagraph_time[b] = runtime
                eager_time[b] = eager_runtime

    # print in tabular format
    print("batch size\teager mode\tfull cudagraph\tpiecewise cudagraph")
    for b in cudagraph_sizes:
        print(f"{b}\t{eager_time[b]:.3f}\t{full_cudagraph_time[b]:.3f}"
              f"\t{piecewise_cudagraph_time[b]:.3f}")


if __name__ == "__main__":
    benchmark()
