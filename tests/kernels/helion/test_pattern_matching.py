# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test make_fx tracing and inductor pattern matching with HelionKernelWrapper."""

import contextlib
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

import helion
import helion.language as hl
from helion._compiler._dynamo.higher_order_ops import (
    helion_kernel_side_table,
    helion_kernel_wrapper_mutation,
)
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    fwd_only,
    register_replacement,
    select_decomp_table,
)
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.register import HelionKernelWrapper


@contextlib.contextmanager
def _helion_mock_context():
    configs = {
        "default": helion.Config(block_sizes=[64], num_warps=2, num_stages=2),
    }
    mock_config_manager = Mock(spec=ConfigManager)
    mock_config_manager.get_platform_configs = Mock(return_value=configs)

    with (
        patch(
            "vllm.kernels.helion.config_manager.ConfigManager.get_instance",
            return_value=mock_config_manager,
        ),
        patch(
            "vllm.kernels.helion.utils.get_canonical_gpu_name",
            return_value="nvidia_h200",
        ),
    ):
        yield


class TestMakeFxHop:
    def setup_method(self):
        helion_kernel_side_table.reset_table()

    def test_make_fx_symbolic(self):
        def raw_add_scale(
            x: torch.Tensor, y: torch.Tensor, scale: float
        ) -> tuple[torch.Tensor, int, torch.Tensor]:
            out_x = torch.empty_like(x)
            out_y = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out_x[tile] = x[tile] + y[tile] * scale
                out_y[tile] = out_x[tile] * 2.0
            return out_x, 42, out_y

        input_x = torch.randn(7, 13)
        input_y = torch.randn(7, 13)
        scale = 0.5

        with _helion_mock_context():
            wrapper = HelionKernelWrapper(
                raw_kernel_func=raw_add_scale,
                op_name="test_make_fx",
                fake_impl=lambda *a, **kw: None,
            )
            wrapper.register_config_picker(lambda args, keys: "default")

            def fn(x, y):
                return wrapper(x, y, scale)

            gm = make_fx(fn, tracing_mode="symbolic")(input_x, input_y)

        hop_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is helion_kernel_wrapper_mutation
        ]
        assert len(hop_nodes) == 1
        node = hop_nodes[0]

        assert node.kwargs["constant_args"]["scale"] == scale
        assert set(node.kwargs["tensor_args"]) == {"x", "y"}

        specs = node.kwargs["output_spec"]["leaf_specs"]
        tensor_specs = [s for s in specs if s["type"] == "tensor"]
        scalar_specs = [s for s in specs if s["type"] == "scalar"]
        assert len(tensor_specs) == 2
        assert len(scalar_specs) == 1

        for spec in tensor_specs:
            assert spec["dtype"] == input_x.dtype

        assert scalar_specs[0]["scalar_value"] == 42

        for val in node.meta["val"]:
            assert all(isinstance(s, torch.SymInt) for s in val.shape)

        # Both out_x and out_y are empty_like(x), so output shapes == input shape
        input_node = next(n for n in gm.graph.nodes if n.op == "placeholder")
        input_shape = input_node.meta["val"].shape
        for val in node.meta["val"]:
            assert len(val.shape) == len(input_shape)
            for out_s, in_s in zip(val.shape, input_shape):
                assert out_s == in_s

    def test_pattern_matcher_replaces_with_helion_hop(self):
        def raw_silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, N = x.size()
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([M, N]):
                out[tile_m, tile_n] = (
                    torch.nn.functional.silu(x[tile_m, tile_n]) * y[tile_m, tile_n]
                )
            return out

        with _helion_mock_context():
            wrapper = HelionKernelWrapper(
                raw_kernel_func=raw_silu_mul,
                op_name="test_pm_silu_mul",
                fake_impl=lambda *a, **kw: None,
            )
            wrapper.register_config_picker(lambda args, keys: "default")

            def pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.silu(x) * y

            def replacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return wrapper(x, y)

            inputs = [torch.randn(8, 16), torch.randn(8, 16)]

            pm_pass = PatternMatcherPass(pass_name="test_helion_replacement")
            register_replacement(pattern, replacement, inputs, fwd_only, pm_pass)

            def model(x, y):
                return torch.nn.functional.silu(x) * y

            decompositions = select_decomp_table()
            input_x = torch.randn(8, 16)
            input_y = torch.randn(8, 16)
            gm = make_fx(model, decompositions, tracing_mode="symbolic")(
                input_x, input_y
            )

            def count_hop_nodes(graph):
                return sum(
                    1
                    for n in graph.nodes
                    if n.op == "call_function"
                    and n.target is helion_kernel_wrapper_mutation
                )

            assert count_hop_nodes(gm.graph) == 0

            match_count = pm_pass.apply(gm.graph)
            gm.graph.lint()
            gm.recompile()

            assert match_count == 1
            assert count_hop_nodes(gm.graph) == 1

            hop_node = next(
                n
                for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is helion_kernel_wrapper_mutation
            )

            # raw_silu_mul returns empty_like(x), so output shape == input shape
            for val in hop_node.meta["val"]:
                assert all(isinstance(s, torch.SymInt) for s in val.shape)

            input_node = next(n for n in gm.graph.nodes if n.op == "placeholder")
            input_shape = input_node.meta["val"].shape
            output_shape = hop_node.meta["val"][0].shape
            assert len(output_shape) == len(input_shape)
            for out_s, in_s in zip(output_shape, input_shape):
                assert out_s == in_s
