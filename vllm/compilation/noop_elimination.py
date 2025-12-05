# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch.fx
from torch import SymInt
from torch.fx.experimental.symbolic_shapes import statically_known_true

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

    Cases handled:
      1. A chain of reshapes is equivalent to the last reshape called on the
      base tensor (input of the first reshape).
      2. A reshape that produces the shape of the input is redundant
      3. A slice that produces the shape of the input is redundant

    Example graph 1:
    mul_1: "f16[s0, 4096]" = ...
    view_1: "f16[s0, 128, 32]" = torch.reshape(mul_1, [-1, 128, 32])
    view_2: "f16[s0, 4096]" = torch.reshape(view_2, [-1, 4096])
    view_3: "f16[s0, 128, 32]" = torch.reshape(view_3, [-1, 128, 32])

    Can be replaced with:
    mul_1: "f16[s0, 4096]" = ...
    view_3: "f16[s0, 128, 32]" = ...

    Example graph 2:
    getitem_1: "f16[s0, 4096]" = ...
    view_1: "f16[s0, 4096]" = torch.reshape(getitem_1, [-1, 4096])
    at = auto_functionalized(static_scaled_fp8_quant, input = view_1, ...)
    out: "f8e4m3fn[s0, 4096]" = at[1]

    Can be replaced with:
    getitem_1: "f16[s0, 4096]" = ...
    at = auto_functionalized(static_scaled_fp8_quant, input = getitem_1, ...)
    out: "f8e4m3fn[s0, 4096]" = at[1]

    Example graph 3:
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
    """

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        count = 0
        # Remove no-op reshapes/views:
        for node in graph.nodes:
            if is_func(node, torch.ops.aten.reshape.default):
                # Case 1: rewrite reshape chains to reshapes on the base tensor
                input = node.args[0]
                # If the input is a reshape, rebind to that node
                if is_func(input, torch.ops.aten.reshape.default):
                    # The new input is guaranteed not to be a reshape,
                    # because we process nodes in order
                    node.update_arg(0, input.args[0])
                    if len(input.users) == 0:
                        graph.erase_node(input)
                        count += 1

            # remove reshape/slice if it produces the original shape
            if is_func(node, torch.ops.aten.reshape.default) or is_func(
                node, torch.ops.aten.slice.Tensor
            ):
                input = node.args[0]
                input_shape = input.meta["val"].shape
                output_shape = node.meta["val"].shape
                if self.all_dims_equivalent(input_shape, output_shape):
                    node.replace_all_uses_with(input)
                    graph.erase_node(node)
                    count += 1
            elif is_func(node, torch.ops.aten.slice_scatter.default):
                base, view, dim_index, start, end = node.args[:5]
                base_shape = base.meta["val"].shape
                view_shape = view.meta["val"].shape

                if self.all_dims_equivalent(base_shape, view_shape):
                    node.replace_all_uses_with(view)
                    graph.erase_node(node)
                    count += 1

        logger.debug("Removed %s no-op reshapes and slices", count)

    # ---------------------- Shape comparison helpers ----------------------
    def dims_equivalent(self, dim: int | SymInt, i_dim: int | SymInt) -> bool:
        """
        This function checks if two dimensions are equivalent.
        :param dim: The dimension arg to reshape/slice
        :param i_dim: The corresponding dimension in the input tensor
        :return: Are the dimensions equivalent?

        There are two cases in which the dimensions are equivalent:
        1. The dimensions are equal (both integers)
        2. The dimensions both correspond to the same SymInt
        """
        # Case 1
        return statically_known_true(dim == i_dim)

    def all_dims_equivalent(
        self, dims: Iterable[int | SymInt], i_dims: Iterable[int | SymInt]
    ) -> bool:
        dims_ = list(dims)
        i_dims_ = list(i_dims)
        if len(dims_) != len(i_dims_):
            # Different ranks can't be equivalent
            return False
        return all(self.dims_equivalent(s, i_s) for s, i_s in zip(dims, i_dims))
