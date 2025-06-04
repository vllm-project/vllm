# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Union

import torch.fx
from torch import SymInt

from vllm.logger import init_logger

from .fx_utils import is_func
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class NoOpEliminationPass(VllmInductorPass):
    """
    This is an inductor pass that removes redundant reshape/slice operations.
    It is required for RMSNorm-quant fusion to work properly.
    That's because apply_fp8_linear adds a reshape, which is redundant
    in the 2D-case. Additionally, torch internal no-op elimination pass does
    not handle certain slice variants.

    Example graph 1:
    getitem_1: "f16[s0, 4096]" = ...
    view_1: "f16[s0, 4096]" = torch.reshape(getitem_1, [-1, 4096])
    at = auto_functionalized(static_scaled_fp8_quant, input = view_1, ...)
    out: "f8e4m3fn[s0, 4096]" = at[1]

    Can be replaced with:
    getitem_1: "f16[s0, 4096]" = ...
    at = auto_functionalized(static_scaled_fp8_quant, input = getitem_1, ...)
    out: "f8e4m3fn[s0, 4096]" = at[1]

    Example graph 2:
    arg0: "s0" = SymInt(s0)
    scaled_mm: "f16[s0, 4096]" = ...
    slice_1: "f16[s0, 4096]" = torch.slice(scaled_mm, -1, 0, arg0)
    at = auto_functionalized(fused_add_rms_norm, input = slice_1, ...)
    out: "f16[s0, 4096]" = torch.slice_scatter(scaled_mm, at[1], 0, 0, arg0)

    Can be replaced with:
    arg0: "s0" = SymInt(s0)
    scaled_mm: "f16[s0, 4096]" = ...
    at = auto_functionalized(fused_add_rms_norm, input = scaled_mm, ...)
    out: "f16[s0, 4096]" = at[1]

    TODO(luka): This is currently tested in test_fusion,
     but separate tests could be good.
    """

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_noop_elimination")
        count = 0
        # Remove no-op reshapes/views:
        for node in graph.nodes:
            if is_func(node, torch.ops.aten.reshape.default):
                input, shape = node.args[:2]
                input_shape = input.meta["val"].shape
                if len(shape) != len(input_shape):
                    # Reshape changing rank, skip
                    continue

                if shape.count(-1) > 1:
                    # Invalid reshape args, skip
                    continue

                if self.all_dims_equivalent(shape, input_shape):
                    node.replace_all_uses_with(input)
                    graph.erase_node(node)
                    count += 1

            elif is_func(node, torch.ops.aten.slice.Tensor):
                input, dim_index, start, end = node.args[:4]
                input_shape = input.meta["val"].shape
                i_dim = input_shape[dim_index]

                if start == 0 and self.dims_equivalent(end, i_dim):
                    node.replace_all_uses_with(input)
                    graph.erase_node(node)
                    count += 1

            elif is_func(node, torch.ops.aten.slice_scatter.default):
                base, view, dim_index, start, end = node.args[:5]
                base_shape = base.meta["val"].shape
                view_shape = view.meta["val"].shape

                view_dim = view_shape[dim_index]

                # Check that view fully covers base and the full view is used
                # (if the view fully covered the base after slicing but was not
                # fully used, we could replace slice_scatter with a simple slice
                # but that's a niche case).
                if (base_shape == view_shape and start == 0
                        and self.dims_equivalent(end, view_dim)):
                    node.replace_all_uses_with(view)
                    graph.erase_node(node)
                    count += 1

        logger.debug("Removed %s no-op reshapes and slices", count)
        self.dump_graph(graph, "after_noop_elimination")
        self.end_and_log()

    def all_dims_equivalent(self, dims: Iterable[Union[int, torch.fx.Node]],
                            i_dims: Iterable[Union[int, SymInt]]):
        return all(
            self.dims_equivalent(s, i_s) for s, i_s in zip(dims, i_dims))

    def dims_equivalent(self, dim: Union[int, torch.fx.Node],
                        i_dim: Union[int, SymInt]) -> bool:
        """
        This function checks if two dimensions are equivalent.
        :param dim: The dimension arg to reshape/slice
        :param i_dim: The corresponding dimension in the input tensor
        :return: Are the dimensions equivalent?

        There are three cases in which the dimensions are equivalent:
        1. The dimensions are equal (both integers)
        2. The reshape dimension is -1 (i.e. inferred)
        3. The dimensions both correspond to the same SymInt

        While case 2 does not guarantee the dimensions are equal,
        they are equal if all other dimensions are equal.

        In case 3, the reshape dimension is a torch.fx.Node,
        and its value is a SymInt. That value is equal to the
        input dimension.

        """
        # Case 1 and 2
        if dim == i_dim or dim == -1:
            return True
        # Case 3
        return isinstance(dim, torch.fx.Node) and dim.meta["val"] == i_dim
