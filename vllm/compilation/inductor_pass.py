from abc import ABC, abstractmethod

import torch

from vllm import envs
from vllm.distributed import get_tensor_model_parallel_rank as get_tp_rank
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tp_world_size)
from vllm.logger import init_logger

logger = init_logger(__name__)


class InductorPass(ABC):

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        if stage in envs.VLLM_TORCH_COMPILE_DUMP:
            # Make sure filename includes rank in the distributed setting
            rank = f"-{get_tp_rank()}" if get_tp_world_size() > 1 else ""
            filename = f"{stage}{rank}.py"

            logger.info("Printing graph to %s", filename)
            with open(filename, "w") as f:
                src = graph.python_code(root_module="self", verbose=True).src
                print(src, file=f)

    @abstractmethod
    def __call__(self, graph: torch.fx.Graph):
        raise NotImplementedError
