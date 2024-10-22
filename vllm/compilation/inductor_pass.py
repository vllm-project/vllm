from abc import ABC, abstractmethod

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class InductorPass(ABC):

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        # TODO(luka): rename env var to VLLM_TORCH_COMPILE_DUMP
        if stage in envs.VLLM_TORCH_COMPILE_DUMP:
            filename = f"{stage}.py"  # TODO(luka): add rank
            logger.info("Printing graph to %s", filename)
            with open(filename, "w") as f:
                src = graph.python_code(root_module="self", verbose=True).src
                print(src, file=f)

    @abstractmethod
    def __call__(self, graph: torch.fx.Graph):
        raise NotImplementedError
