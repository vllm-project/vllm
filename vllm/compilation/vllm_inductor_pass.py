# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import torch
from torch._dynamo.utils import lazy_format_graph_code

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .inductor_pass import InductorPass

logger = init_logger(__name__)


class VllmInductorPass(InductorPass):
    """
    An inductor pass with access to vLLM PassConfig.
    It provides timing, logging, and dumping utilities.
    """

    def __init__(self, config: VllmConfig):
        self.pass_config = config.compilation_config.pass_config
        self.model_dtype = config.model_config.dtype if config.model_config \
            else None
        self.device = config.device_config.device if config.device_config \
            else None
        self.pass_name = self.__class__.__name__

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        lazy_format_graph_code(stage, graph.owning_module)

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)


class PrinterInductorPass(VllmInductorPass):

    def __init__(self, name: str, config: VllmConfig):
        super().__init__(config)
        self.name = name

    def __call__(self, graph: torch.fx.Graph):
        self.dump_graph(graph, self.name)
