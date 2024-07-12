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
    The FusedOpGenerator is a class that is responsible for generating a fused
    operation for sequences of gx graph nodes.
    """

    @abstractmethod
    def make_fused_op(
            self, op_name: str, inputs: List[torch.fx.Node],
            outputs: List[torch.fx.Node], nodes: List[torch.fx.Node],
            kwargs: Dict[str, Dict[str, torch.fx.node.Argument]]) -> Callable:
        raise FusionFail("no generator")
