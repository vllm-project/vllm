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
from vllm.compilation.decorators import (ignore_torch_compile,
                                         support_torch_compile)
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
        self.rms_norm_weight = nn.Parameter(torch.ones(hidden_size))

        # Initialize to same weights for testing
        nn.init.xavier_normal_(
            self.pre_attn.weight.data,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
            gain=0.001)
        nn.init.xavier_normal_(
            self.post_attn.weight.data,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
            gain=0.001)

    def rms_norm_ref(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        return (x_f32 * torch.rsqrt(
            torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6) *
                self.rms_norm_weight).to(x.dtype)

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
class SimpleModelWithTwoGraphs(ParentModel):

    def __init__(self,
                 *,
                 mlp_size: int,
                 hidden_size: int,
                 vllm_config: VllmConfig,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Test will fail without set_model_tag here with error:
        # "ValueError: too many values to unpack (expected 3)"
        # This is because CompiledAttention and CompiledAttentionTwo
        # have different implmentations but the same torch.compile
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


def test_ignore_torch_compile_decorator():
    assert VLLM_USE_V1

    # piecewise
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=True,
        splitting_ops=["silly.attention"],
        cudagraph_capture_sizes=[1, 2],
    ))

    @support_torch_compile
    class A(nn.Module):

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

    @ignore_torch_compile
    class B(A):
        ...

    @support_torch_compile
    class C(B):
        ...

    with set_current_vllm_config(vllm_config):
        mod_A = A(vllm_config=vllm_config, prefix='').eval().cuda()

    # A has support_torch_compile
    with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=3,
            num_piecewise_capturable_graphs_seen=2,
            num_backend_compilations=2,
            num_cudagraph_captured=4,
            # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ), set_forward_context({}, vllm_config=vllm_config):
        # first run is for compile
        mod_A(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())
        # run cudagraph captured sizes
        mod_A(torch.randn(2, MLP_SIZE).cuda())
        mod_A(torch.randn(1, MLP_SIZE).cuda())

    with set_current_vllm_config(vllm_config):
        mod_B = B(vllm_config=vllm_config, prefix='').eval().cuda()

    # B's ignore_torch_compile should override A's support_torch_compile
    with compilation_counter.expect(
            num_graphs_seen=0,
            num_piecewise_graphs_seen=0,
            num_piecewise_capturable_graphs_seen=0,
            num_backend_compilations=0,
            num_cudagraph_captured=0,
    ), set_forward_context({}, vllm_config=vllm_config):
        mod_B(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())
        mod_B(torch.randn(2, MLP_SIZE).cuda())
        mod_B(torch.randn(1, MLP_SIZE).cuda())

    with set_current_vllm_config(vllm_config):
        mod_C = C(vllm_config=vllm_config, prefix='').eval().cuda()

    # C's support_torch_compile should override B's ignore_torch_compile
    with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=3,
            num_piecewise_capturable_graphs_seen=2,
            num_backend_compilations=2,
            num_cudagraph_captured=4,
            # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ), set_forward_context({}, vllm_config=vllm_config):
        mod_C(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())
        mod_C(torch.randn(2, MLP_SIZE).cuda())
        mod_C(torch.randn(1, MLP_SIZE).cuda())


@torch.inference_mode
def run_model(vllm_config, model: nn.Module, inputs: torch.Tensor):
    with set_forward_context({}, vllm_config=vllm_config):
        # First run is for compile
        model(inputs)

        # Run CUDAGraph captured sizes
        model(inputs[:2])
        model(inputs[:1])

        output = model(inputs[:2])

        output = output.cpu()
        return output.cpu()


def test_multi_graph_piecewise_compile_outputs_equal():
    outputs = []

    # piecewise compile
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=True,
        splitting_ops=["silly.attention"],
        cudagraph_capture_sizes=[1, 2],
    ))

    with set_current_vllm_config(vllm_config):
        model = SimpleModelWithTwoGraphs(mlp_size=MLP_SIZE,
                                         hidden_size=HIDDEN_SIZE,
                                         vllm_config=vllm_config,
                                         prefix='').eval().cuda()

    # Pre-allocate memory for CUDAGraph which expects
    # static tensor addresses
    inputs = torch.randn(BATCH_SIZE, MLP_SIZE).cuda()

    with compilation_counter.expect(
            num_graphs_seen=2,  # two graphs for the model
            num_piecewise_graphs_seen=6,
            # attn_one, attn_two each has 3 piecewise graphs
            # (pre attn, post attn, silly_attention) each
            num_piecewise_capturable_graphs_seen=4,
            # attn_one, attn_two has pre attn and post attn each, total=4
            num_backend_compilations=4,  # num_piecewise_capturable_graphs_seen
            num_cudagraph_captured=8,
            # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ):
        outputs.append(run_model(vllm_config, model, inputs))

    # no compile or cudagraph
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.NO_COMPILATION, ))

    with set_current_vllm_config(vllm_config):
        model = SimpleModelWithTwoGraphs(mlp_size=MLP_SIZE,
                                         hidden_size=HIDDEN_SIZE,
                                         vllm_config=vllm_config,
                                         prefix='').eval().cuda()

    with compilation_counter.expect(
            num_graphs_seen=0,
            num_piecewise_graphs_seen=0,
            num_piecewise_capturable_graphs_seen=0,
            num_backend_compilations=0,
            num_cudagraph_captured=0,
    ):
        outputs.append(run_model(vllm_config, model, inputs))

    # piecewise compile without CUDA graph
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=False,
        splitting_ops=["silly.attention"],
    ))

    with set_current_vllm_config(vllm_config):
        model = SimpleModelWithTwoGraphs(mlp_size=MLP_SIZE,
                                         hidden_size=HIDDEN_SIZE,
                                         vllm_config=vllm_config,
                                         prefix='').eval().cuda()

    with compilation_counter.expect(
            num_graphs_seen=2,
            num_piecewise_graphs_seen=6,
            num_piecewise_capturable_graphs_seen=4,
            num_backend_compilations=4,
            num_cudagraph_captured=0,  # no cudagraph captured
    ):
        outputs.append(run_model(vllm_config, model, inputs))

    # Generally don't expect outputs with and without inductor
    # to be bitwise equivalent
    assert torch.allclose(outputs[0], outputs[1])

    # Expect bitwise equivalence using inductor w/ and w/o cudagraph
    assert torch.equal(outputs[0], outputs[2])
