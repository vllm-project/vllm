# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test (piecewise) compilation with a simple model where multiple submodules
are compiled and graph captured separately.
"""
import torch
from torch import nn
from torch.library import Library

from vllm.compilation.backends import set_model_tag
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)
from vllm.envs import VLLM_USE_V1
from vllm.forward_context import set_forward_context
from vllm.utils import direct_register_custom_op

# create a library to hold the custom op
silly_lib = Library("silly", "FRAGMENT")  # noqa

BATCH_SIZE = 32
MLP_SIZE = 128
HIDDEN_SIZE = 1024
RANDOM_SEED = 0


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
class ParentModel(nn.Module):
    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Attention(nn.Module):
    def __init__(self, mlp_size: int, hidden_size: int) -> None:
        super().__init__()
        self.pre_attn = nn.Linear(mlp_size, hidden_size, bias=False)
        self.post_attn = nn.Linear(hidden_size, mlp_size, bias=False)

        nn.init.xavier_normal_(self.pre_attn.weight.data,
                            generator=torch.Generator().manual_seed(
                                RANDOM_SEED),
                            gain=0.001)
        nn.init.xavier_normal_(self.post_attn.weight.data,
                            generator=torch.Generator().manual_seed(
                                RANDOM_SEED),
                            gain=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_attn(x)
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output
        x = self.post_attn(x)
        return x

@support_torch_compile
class CompiledAttention(nn.Module):
    def __init__(self,
                *,
                mlp_size: int,
                hidden_size: int,
                vllm_config: VllmConfig,
                prefix: str = '',
                **kwargs) -> None:
        super().__init__()
        self.attn = Attention(mlp_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)

@support_torch_compile
class CompiledAttentionTwo(CompiledAttention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x) + x

@ignore_torch_compile
class SimpleModel(ParentModel):
    def __init__(self,
                *,
                mlp_size: int,
                hidden_size: int,
                vllm_config: VllmConfig,
                prefix: str = '',
                **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.attn_one = Attention(mlp_size, hidden_size)
        self.attn_two = Attention(mlp_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_one(x)
        x = self.attn_two(x) + x
        return x

@ignore_torch_compile
class SimpleModelWithTwoGraphs(ParentModel):
    def __init__(self,
                *,
                mlp_size: int,
                hidden_size: int,
                vllm_config: VllmConfig,
                prefix: str = '',
                **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Test will fail without `set_model_tag`` here with error:
        # "ValueError: too many values to unpack (expected 3)"
        # This is because CompiledAttention and CompiledAttentionTwo 
        # have different implmentations but the same torch.compile
        # cache dir will be used by default
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


def test_ignore_torch_compile_decorator():
    assert VLLM_USE_V1

    # piecewise
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            use_cudagraph=True,
            splitting_ops=["silly.attention"],
            cudagraph_capture_sizes=[1, 2],
        )
    )

    with set_current_vllm_config(vllm_config):
        model = SimpleModel(
            mlp_size=MLP_SIZE,
            hidden_size=HIDDEN_SIZE,
            vllm_config=vllm_config, 
            prefix=''
        ).eval().cuda()

    with compilation_counter.expect(
        num_graphs_seen=0,
        num_piecewise_graphs_seen=0,
        num_piecewise_capturable_graphs_seen=0,
        num_backend_compilations=0,
        num_cudagraph_captured=0,
    ), set_forward_context({}, vllm_config=vllm_config):
        # first run is for compile
        model(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())

        # run cudagraph captured sizes
        model(torch.randn(2, MLP_SIZE).cuda())
        model(torch.randn(1, MLP_SIZE).cuda())
    

@torch.inference_mode
def run_model(vllm_config, model: nn.Module):
    with set_forward_context({}, vllm_config=vllm_config):
        # Pre-allocate memory for CUDAGraph which expects
        # static tensor addresses
        inputs = torch.randn(BATCH_SIZE, MLP_SIZE).cuda()

        # First run is for compile
        model(inputs)

        # Run CUDAGraph captured sizes
        model(inputs[:2])
        model(inputs[:1])

        inputs[:2].fill_(1.0)
        output = model(inputs[:2])

        output = output.cpu()
        return output.cpu()

def test_multi_graph_piecewise_compile(monkeypatch):
    assert VLLM_USE_V1

    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    
    outputs = []

    # piecewise compile
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            use_cudagraph=True,
            splitting_ops=["silly.attention"],
            cudagraph_capture_sizes=[1, 2],
        )
    )

    with set_current_vllm_config(vllm_config):
        model = SimpleModelWithTwoGraphs(
            mlp_size=MLP_SIZE,
            hidden_size=HIDDEN_SIZE,
            vllm_config=vllm_config, 
            prefix=''
        ).eval().cuda()

    with compilation_counter.expect(
        num_graphs_seen=2,  # two graphs for the model
        num_piecewise_graphs_seen=6,
        # attn_one, attn_two each has 3 piecewise graphs 
        # (pre_attn, post_attn, silly_attention) each
        num_piecewise_capturable_graphs_seen=4, 
        # attn_one, attn_two has pre_attn and post_attn each, total=4
        num_backend_compilations=4, # num_piecewise_capturable_graphs_seen
        num_cudagraph_captured=8, 
        # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ):
        outputs.append(run_model(vllm_config, model))

    # no compile or cudagraph
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            level=CompilationLevel.NO_COMPILATION,
        )
    )

    with set_current_vllm_config(vllm_config):
        model = SimpleModelWithTwoGraphs(
            mlp_size=MLP_SIZE,
            hidden_size=HIDDEN_SIZE,
            vllm_config=vllm_config, 
            prefix=''
        ).eval().cuda()

    with compilation_counter.expect(
        num_graphs_seen=0,
        num_piecewise_graphs_seen=0,
        num_piecewise_capturable_graphs_seen=0,
        num_backend_compilations=0,
        num_cudagraph_captured=0,
    ):
        outputs.append(run_model(vllm_config, model))

    assert torch.allclose(outputs[0], outputs[1])
