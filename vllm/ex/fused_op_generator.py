###############################################################################
#
# Fused operation generator base class.
#
###############################################################################

import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable


class FusionFail(Exception):
    """
    An exception used to indicate a failure in the fusion or fused op generation process.
    Should be recoverable, i.e. can fall back to the non-fused version of graph.
    """
    pass


class FusedOpGenerator(ABC):
    """
    The FusedOpGenerator is a class that is responsible for generating a fused CUDA/C++
    operation for sequences of gx graph nodes.

    Use of the class is broken up into two steps: 'make_fused_op' and 'build_ops'.
    """

    @abstractmethod
    def make_fused_op(
        self, inputs: List[torch.fx.Node], outputs: List[torch.fx.Node],
        nodes: List[torch.fx.Node], kwargs: Dict[str,
                                                 Dict[str,
                                                      torch.fx.node.Argument]]
    ) -> torch.fx.node.Target:
        raise FusionFail("no generator")

    @abstractmethod
    def build_ops(self) -> Dict[torch.fx.node.Target, Callable]:
        raise FusionFail("no generator")
