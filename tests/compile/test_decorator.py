# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer

# This import automatically registers `torch.ops.silly.attention`
from . import silly_attention  # noqa: F401

BATCH_SIZE = 32
MLP_SIZE = 128


@torch.inference_mode
def run_model(
    vllm_config: VllmConfig, model: nn.Module, cudagraph_runtime_mode: CUDAGraphMode
):
    with set_forward_context({}, vllm_config=vllm_config):
        # warmup for the model with cudagraph_mode NONE
        model(torch.randn(BATCH_SIZE, MLP_SIZE).to(current_platform.device_name))

        # simulate cudagraphs capturing
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            model(torch.randn(2, MLP_SIZE).to(current_platform.device_name))
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=1,
            ),
        ):
            model(torch.randn(1, MLP_SIZE).to(current_platform.device_name))

        # simulate cudagraphs replay
        with set_forward_context(
            {},
            vllm_config=vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=2,
            ),
        ):
            output = model(torch.randn(2, MLP_SIZE).to(current_platform.device_name))

        output = output.cpu()
        return output.cpu()


@pytest.mark.parametrize("use_inductor_graph_partition", [True, False])
def test_ignore_torch_compile_decorator(use_inductor_graph_partition, monkeypatch):
    # disable compile cache so that we can count the number of compilations
    # appropriately
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    if use_inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("inductor graph partition is only available in PyTorch 2.9+")

    # piecewise
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            splitting_ops=["silly::attention"],
            cudagraph_capture_sizes=[1, 2],
            use_inductor_graph_partition=use_inductor_graph_partition,
        )
    )
    cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE

    expected_num_graphs_seen = 1
    expected_num_cudagraph_captured = (
        4  # num_cudagraph_sizes * num cudagraphs to capture
    )
    if use_inductor_graph_partition:
        expected_num_piecewise_graphs_seen = 1
        expected_num_piecewise_capturable_graphs_seen = 1
        expected_num_backend_compilations = 1
    else:
        expected_num_piecewise_graphs_seen = 3
        expected_num_piecewise_capturable_graphs_seen = 2
        expected_num_backend_compilations = 2

    @support_torch_compile
    class A(nn.Module):
        def __init__(
            self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs
        ) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + x
            attn_output = torch.empty_like(x)
            torch.ops.silly.attention(x, x, x, attn_output)
            x = attn_output
            x = x * 3
            return x

    @ignore_torch_compile
    class B(A): ...

    @support_torch_compile
    class C(B): ...

    with set_current_vllm_config(vllm_config):
        mod_A = (
            A(vllm_config=vllm_config, prefix="")
            .eval()
            .to(current_platform.device_name)
        )

    # A has support_torch_compile
    with compilation_counter.expect(
        num_graphs_seen=expected_num_graphs_seen,
        num_piecewise_graphs_seen=expected_num_piecewise_graphs_seen,
        num_piecewise_capturable_graphs_seen=expected_num_piecewise_capturable_graphs_seen,
        num_backend_compilations=expected_num_backend_compilations,
        num_cudagraph_captured=expected_num_cudagraph_captured,
    ):
        run_model(vllm_config, mod_A, cudagraph_runtime_mode)

    with set_current_vllm_config(vllm_config):
        mod_B = (
            B(vllm_config=vllm_config, prefix="")
            .eval()
            .to(current_platform.device_name)
        )

    # B's ignore_torch_compile should override A's support_torch_compile
    with compilation_counter.expect(
        num_graphs_seen=0,
        num_piecewise_graphs_seen=0,
        num_piecewise_capturable_graphs_seen=0,
        num_backend_compilations=0,
        num_cudagraph_captured=0,
    ):
        run_model(vllm_config, mod_B, cudagraph_runtime_mode)

    with set_current_vllm_config(vllm_config):
        mod_C = (
            C(vllm_config=vllm_config, prefix="")
            .eval()
            .to(current_platform.device_name)
        )

    # C's support_torch_compile should override B's ignore_torch_compile
    with compilation_counter.expect(
        num_graphs_seen=expected_num_graphs_seen,
        num_piecewise_graphs_seen=expected_num_piecewise_graphs_seen,
        num_piecewise_capturable_graphs_seen=expected_num_piecewise_capturable_graphs_seen,
        num_backend_compilations=expected_num_backend_compilations,
        num_cudagraph_captured=expected_num_cudagraph_captured,
    ):
        run_model(vllm_config, mod_C, cudagraph_runtime_mode)


# Only enable torch.compile if
# vllm_config.cache_config.kv_sharing_fast_prefill=True
@support_torch_compile(
    enable_if=lambda vllm_config: vllm_config.cache_config.kv_sharing_fast_prefill
)
class B(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output
        x = x + x
        return x


# Only enable torch.compile if
# vllm_config.cache_config.kv_sharing_fast_prefill=False
@support_torch_compile(
    enable_if=lambda vllm_config: not vllm_config.cache_config.kv_sharing_fast_prefill
)
class A(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
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


@pytest.mark.parametrize("use_inductor_graph_partition", [True, False])
def test_conditional_compile_enable_if(use_inductor_graph_partition, monkeypatch):
    # disable compile cache so that we can count the number of compilations
    # appropriately
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    if use_inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("inductor graph partition is only available in PyTorch 2.9+")

    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            kv_sharing_fast_prefill=True,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            splitting_ops=["silly::attention"],
            cudagraph_capture_sizes=[1, 2],
            use_inductor_graph_partition=use_inductor_graph_partition,
        ),
    )
    cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE

    with set_current_vllm_config(vllm_config):
        mod_A = (
            A(vllm_config=vllm_config, prefix="")
            .eval()
            .to(current_platform.device_name)
        )

    if use_inductor_graph_partition:
        expected_num_piecewise_graphs_seen = 2
        expected_num_piecewise_capturable_graphs_seen = 2
        expected_num_backend_compilations = 2
    else:
        expected_num_piecewise_graphs_seen = 6
        expected_num_piecewise_capturable_graphs_seen = 4
        expected_num_backend_compilations = 4

    # A has support_torch_compile but enable_if fn returns False
    # enalbe_if will be True for B, so we expect mod1 and mod2
    # to be compiled
    with compilation_counter.expect(
        num_graphs_seen=2,
        num_piecewise_graphs_seen=expected_num_piecewise_graphs_seen,
        # 3 piecewise graphs per instance of B()
        num_piecewise_capturable_graphs_seen=expected_num_piecewise_capturable_graphs_seen,
        num_backend_compilations=expected_num_backend_compilations,
        num_cudagraph_captured=8,
        # num_cudagraph_sizes * num cudagraphable graphs to capture
    ):
        run_model(vllm_config, mod_A, cudagraph_runtime_mode)

    # Set kv_sharing_fast_prefill=False
    # which will cause A to be compiled and B to not be compiled
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            kv_sharing_fast_prefill=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            splitting_ops=["silly::attention"],
            cudagraph_capture_sizes=[1, 2],
            use_inductor_graph_partition=use_inductor_graph_partition,
        ),
    )

    with set_current_vllm_config(vllm_config):
        mod_A = (
            A(vllm_config=vllm_config, prefix="")
            .eval()
            .to(current_platform.device_name)
        )

    if use_inductor_graph_partition:
        expected_num_piecewise_graphs_seen = 1
        expected_num_piecewise_capturable_graphs_seen = 1
        expected_num_backend_compilations = 1
    else:
        # 3 attn ops and 4 non-attn ops
        expected_num_piecewise_graphs_seen = 7
        expected_num_piecewise_capturable_graphs_seen = 4
        expected_num_backend_compilations = 4

    with compilation_counter.expect(
        num_graphs_seen=1,
        num_piecewise_graphs_seen=expected_num_piecewise_graphs_seen,
        # 3 attn ops and 4 non-attn ops
        num_piecewise_capturable_graphs_seen=expected_num_piecewise_capturable_graphs_seen,
        num_backend_compilations=expected_num_backend_compilations,
        num_cudagraph_captured=8,
        # num_cudagraph_sizes * num cudagraphable graphs to capture
    ):
        run_model(vllm_config, mod_A, cudagraph_runtime_mode)
