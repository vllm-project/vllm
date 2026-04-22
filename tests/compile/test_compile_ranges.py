# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from torch import fx as fx
from torch import nn

# This import automatically registers `torch.ops.silly.attention`
import tests.compile.silly_attention  # noqa
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.passes.inductor_pass import (
    InductorPass,
    get_pass_context,
)
from vllm.config import (
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.config.scheduler import SchedulerConfig
from vllm.config.utils import Range
from vllm.forward_context import set_forward_context

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
        model(torch.randn(BATCH_SIZE, MLP_SIZE))
        for batch_size in batch_sizes:
            model(torch.randn(batch_size, MLP_SIZE))


class PostGradRangeChecker(InductorPass):
    def __init__(self, ranges: list[Range]):
        self.ranges = ranges
        self.num_calls = 0

    def __call__(self, graph: fx.Graph):
        compile_range = get_pass_context().compile_range
        assert compile_range in self.ranges, (
            f"Compile range {compile_range} not in {self.ranges}"
        )
        self.num_calls += 1

    def uuid(self) -> str:
        state: dict[str, Any] = {}
        return InductorPass.hash_dict(state)


def test_compile_ranges(use_fresh_inductor_cache):
    post_grad_range_checker = PostGradRangeChecker(
        [
            Range(start=1, end=8),
            Range(start=16, end=16),
            Range(start=9, end=32),
            Range(start=64, end=64),
            Range(start=128, end=128),
            Range(start=33, end=8192),
        ]
    )
    torch.set_default_device("cuda")
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            compile_ranges_endpoints=[8, 32],
            compile_sizes=[16, 64, 128],
            inductor_compile_config={
                "post_grad_custom_post_pass": post_grad_range_checker,
            },
        ),
    )

    with set_current_vllm_config(vllm_config):
        model = TestModel(vllm_config=vllm_config, prefix="").eval()
        # Number of compilations: 3 compile ranges + 3 compile sizes
        batch_sizes = [1, 4, 16, 24, 48, 64, 8192]

        with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=1,
            num_backend_compilations=6,
        ):
            run_model(vllm_config, model, batch_sizes)
        assert post_grad_range_checker.num_calls == 6


def test_compile_config_get_compile_ranges():
    compilation_config = CompilationConfig(
        compile_ranges_endpoints=[8, 32],
    )
    VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=compilation_config,
    )
    assert compilation_config.get_compile_ranges() == [
        Range(start=1, end=8),
        Range(start=9, end=32),
        Range(start=33, end=8192),
    ]


class PostGradStaticShapeChecker(InductorPass):
    """Asserts that compile_sizes entries produce graphs with fully concrete
    (non-symbolic) shapes, and compile_ranges entries have symbolic shapes."""

    def __init__(self):
        self.num_static_calls = 0
        self.num_dynamic_calls = 0

    def __call__(self, graph: fx.Graph):
        from torch.fx.experimental.symbolic_shapes import is_symbolic

        compile_range = get_pass_context().compile_range
        is_single = compile_range.is_single_size()

        for node in graph.nodes:
            val = node.meta.get("val")
            if val is None:
                val = node.meta.get("example_value")
            if isinstance(val, torch.Tensor):
                has_symbolic = any(is_symbolic(d) for d in val.shape)
                if is_single:
                    assert not has_symbolic, (
                        f"compile_sizes entry {compile_range}: "
                        f"node '{node.name}' has symbolic shape "
                        f"{val.shape}"
                    )
                else:
                    # compile_ranges should have at least some
                    # symbolic shapes (the batch dimension)
                    if has_symbolic:
                        self.num_dynamic_calls += 1
                        return

        if is_single:
            self.num_static_calls += 1

    def uuid(self) -> str:
        state: dict[str, Any] = {}
        return InductorPass.hash_dict(state)


def test_compile_sizes_produce_static_shapes(use_fresh_inductor_cache):
    """Verify that compile_sizes entries are compiled with fully concrete
    shapes (no SymInts), while compile_ranges entries retain dynamic shapes."""
    checker = PostGradStaticShapeChecker()
    torch.set_default_device("cuda")
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            compile_ranges_endpoints=[8],
            compile_sizes=[16],
            inductor_compile_config={
                "post_grad_custom_post_pass": checker,
            },
        ),
    )

    with set_current_vllm_config(vllm_config):
        model = TestModel(vllm_config=vllm_config, prefix="").eval()
        # 3 compilations: Range(1,8), Range(9,8192), single-size 16
        with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=1,
            num_backend_compilations=3,
        ):
            run_model(vllm_config, model, [1, 16, 64])

    # compile_sizes=16 should produce static shapes
    assert checker.num_static_calls == 1, (
        f"Expected 1 static compilation, got {checker.num_static_calls}"
    )
    # compile_ranges should produce dynamic shapes
    assert checker.num_dynamic_calls == 2, (
        f"Expected 2 dynamic compilations, got {checker.num_dynamic_calls}"
    )


def test_inductor_cache_compile_ranges(monkeypatch, use_fresh_inductor_cache):
    # To force multiple compilations, we disable the compile cache
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    post_grad_range_checker = PostGradRangeChecker(
        ranges=[
            Range(start=1, end=8),
            Range(start=9, end=8192),
        ]
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=8192,
        max_model_len=8192,
        is_encoder_decoder=False,
    )
    torch.set_default_device("cuda")

    def create_vllm_config():
        return VllmConfig(
            scheduler_config=scheduler_config,
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                compile_ranges_endpoints=[8],
                inductor_compile_config={
                    "post_grad_custom_post_pass": post_grad_range_checker,
                },
            ),
        )

    vllm_config_1 = create_vllm_config()
    with set_current_vllm_config(vllm_config_1):
        model1 = TestModel(vllm_config=vllm_config_1, prefix="").eval()
        batch_sizes = [1, 16]
        run_model(vllm_config_1, model1, batch_sizes)
        assert post_grad_range_checker.num_calls == 2

    post_grad_range_checker.num_calls = 0
    # Create a new vllm config with the new pass context
    vllm_config_2 = create_vllm_config()
    with set_current_vllm_config(vllm_config_2):
        model2 = TestModel(vllm_config=vllm_config_2, prefix="").eval()
        batch_sizes = [4, 32]
        run_model(vllm_config_2, model2, batch_sizes)
        # Check that cache is used, so the number of calls
        # should be 0
        assert post_grad_range_checker.num_calls == 0
