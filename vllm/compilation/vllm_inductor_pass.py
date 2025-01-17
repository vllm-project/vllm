import time

import torch

from vllm.config import CompilationConfig
from vllm.logger import init_logger

from .dump_graph import dump_graph as utils_dump_graph
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
            utils_dump_graph(self.config, graph, stage)

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)
