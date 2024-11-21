from typing import Union

import torch.fx
from torch import SymInt

from vllm.compilation.fusion import is_func
from vllm.compilation.inductor_pass import InductorPass
from vllm.logger import init_logger

logger = init_logger(__name__)


class RedundantReshapesPass(InductorPass):
    """
    This is an inductor pass that removes redundant reshape operations.
    It is required for RMSNorm-quant fusion to work properly.
    That's because apply_fp8_linear adds a reshape, which is redundant
    in the 2D-case.

    Example graph:

    getitem_1: "f16[s0, 4096]" = ...
    view_1: "f16[s0, 4096]" = torch.reshape(getitem_1, [-1, 4096])
    at = auto_functionalized(static_scaled_fp8_quant, input = view_1, ...)
    out: "f8e4m3fn[s0, 4096]" = at[1]

    Can be replaced with:
    getitem_1: "f16[s0, 4096]" = ...
    at = auto_functionalized(static_scaled_fp8_quant, input = getitem_1, ...)
    out: "f8e4m3fn[s0, 4096]" = at[1]
    """

    def __call__(self, graph: torch.fx.Graph):
        self.dump_graph(graph, "before_reshapes")
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

                if all(
                        self.dims_equivalent(s, i_s)
                        for s, i_s in zip(shape, input_shape)):
                    node.replace_all_uses_with(input)
                    graph.erase_node(node)
                    count += 1

        logger.debug("Removed %s no-op reshapes", count)

        self.dump_graph(graph, "after_reshapes")

    def dims_equivalent(self, dim: Union[int, torch.fx.Node],
                        i_dim: Union[int, SymInt]) -> bool:
        """
        This function checks if two dimensions are equivalent.
        :param dim: The dimension arg to reshape
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
