from abc import ABC, abstractmethod

import torch

from vllm.config import CompilationConfig
# yapf: disable
from vllm.distributed import get_tensor_model_parallel_rank as get_tp_rank
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tp_world_size)
from vllm.distributed import model_parallel_is_initialized as p_is_init
# yapf: enable
from vllm.logger import init_logger

logger = init_logger(__name__)


class InductorPass(ABC):

    @abstractmethod
    def __call__(self, graph: torch.fx.Graph):
        raise NotImplementedError

    def __init__(self, config: CompilationConfig):
        self.config = config

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        if stage in self.config.dump_graph_stages:
            # Make sure filename includes rank in the distributed setting
            parallel = p_is_init() and get_tp_world_size() > 1
            rank = f"-{get_tp_rank()}" if parallel else ""
            filepath = self.config.dump_graph_dir / f"{stage}{rank}.py"

            logger.info("Printing graph to %s", filepath)
            with open(filepath, "w") as f:
                src = graph.python_code(root_module="self", verbose=True).src
                # Add imports so it's not full of errors
                print("import torch; from torch import device", file=f)
                print(src, file=f)
