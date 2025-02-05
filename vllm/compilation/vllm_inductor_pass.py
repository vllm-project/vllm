# SPDX-License-Identifier: Apache-2.0

import time

import torch

from vllm.config import CompilationConfig
# yapf: disable
from vllm.distributed import get_tensor_model_parallel_rank as get_tp_rank
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tp_world_size)
from vllm.distributed import model_parallel_is_initialized as p_is_init
# yapf: enable
from vllm.logger import init_logger

from .inductor_pass import InductorPass

logger = init_logger(__name__)


class VllmInductorPass(InductorPass):
    """
    An inductor pass with access to vLLM PassConfig.
    It provides timing, logging, and dumping utilities.
    """

    def __init__(self, config: CompilationConfig.PassConfig):
        self.config = config
        self.pass_name = self.__class__.__name__

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        if stage in self.config.dump_graph_stages:
            # Make sure filename includes rank in the distributed setting
            parallel = p_is_init() and get_tp_world_size() > 1
            rank = f"-{get_tp_rank()}" if parallel else ""
            filepath = self.config.dump_graph_dir / f"{stage}{rank}.py"

            logger.info("%s printing graph to %s", self.pass_name, filepath)
            with open(filepath, "w") as f:
                src = graph.python_code(root_module="self", verbose=True).src
                # Add imports so it's not full of errors
                print("import torch; from torch import device", file=f)
                print(src, file=f)

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)
