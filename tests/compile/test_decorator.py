# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn
from torch.library import Library

from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import (ignore_torch_compile,
                                         support_torch_compile)
from vllm.config import (CacheConfig, CompilationConfig, CompilationLevel,
                         VllmConfig, set_current_vllm_config)
from vllm.forward_context import set_forward_context
from vllm.utils import direct_register_custom_op

# create a library to hold the custom op
silly_lib = Library("silly", "FRAGMENT")  # noqa

BATCH_SIZE = 32
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


def test_ignore_torch_compile_decorator():
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


# Only enable torch.compile if
# vllm_config.cache_config.kv_sharing_fast_prefill=True
@support_torch_compile(compile_cond=lambda vllm_config: vllm_config.
                       cache_config.kv_sharing_fast_prefill)
class B(nn.Module):

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
        x = x + x
        return x


# Only enable torch.compile if
# vllm_config.cache_config.kv_sharing_fast_prefill=False
@support_torch_compile(compile_cond=lambda vllm_config: not vllm_config.
                       cache_config.kv_sharing_fast_prefill)
class A(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__()
        self.mod1 = B(vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.mod2 = B(vllm_config=vllm_config, prefix=prefix, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mod1(x)
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output
        x = self.mod2(x)
        return x


def test_support_torch_compile_cond():
    vllm_config = VllmConfig(cache_config=CacheConfig(
        kv_sharing_fast_prefill=True, ),
                             compilation_config=CompilationConfig(
                                 level=CompilationLevel.PIECEWISE,
                                 use_cudagraph=True,
                                 splitting_ops=["silly.attention"],
                                 cudagraph_capture_sizes=[1, 2],
                             ))

    with set_current_vllm_config(vllm_config):
        mod_A = A(vllm_config=vllm_config, prefix='').eval().cuda()

    # A has support_torch_compile but compile_cond is not satisified
    # compile_cond will be satisified for B, so we expect mod1 and mod2
    # to be compiled
    with compilation_counter.expect(
            num_graphs_seen=2,
            num_piecewise_graphs_seen=6,
            # 3 piecewise graphs per instance of B()
            num_piecewise_capturable_graphs_seen=4,
            num_backend_compilations=4,
            num_cudagraph_captured=8,
            # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ), set_forward_context({}, vllm_config=vllm_config):
        # first run is for compile
        mod_A(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())
        # run cudagraph captured sizes
        mod_A(torch.randn(2, MLP_SIZE).cuda())
        mod_A(torch.randn(1, MLP_SIZE).cuda())

    # Set kv_sharing_fast_prefill=False
    # which will cause A to be compiled and B to not be compiled
    vllm_config = VllmConfig(cache_config=CacheConfig(
        kv_sharing_fast_prefill=False, ),
                             compilation_config=CompilationConfig(
                                 level=CompilationLevel.PIECEWISE,
                                 use_cudagraph=True,
                                 splitting_ops=["silly.attention"],
                                 cudagraph_capture_sizes=[1, 2],
                             ))

    with set_current_vllm_config(vllm_config):
        mod_A = A(vllm_config=vllm_config, prefix='').eval().cuda()

    with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=7,
            # 3 attn ops and 4 non-attn ops
            num_piecewise_capturable_graphs_seen=4,
            num_backend_compilations=4,
            num_cudagraph_captured=8,
            # num_cudagraph_sizes * num_piecewise_capturable_graphs_seen
    ), set_forward_context({}, vllm_config=vllm_config):
        # first run is for compile
        mod_A(torch.randn(BATCH_SIZE, MLP_SIZE).cuda())
        # run cudagraph captured sizes
        mod_A(torch.randn(2, MLP_SIZE).cuda())
        mod_A(torch.randn(1, MLP_SIZE).cuda())
