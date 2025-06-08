# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class StructuredOutputWorkerBackend:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def filter_logits(self, input_batch: InputBatch, device: torch.device,
                      scheduler_output: SchedulerOutput, logits: torch.Tensor,
                      sample_hidden_states: torch.Tensor, **kwargs) -> None:
        """
        Filters the logits produced by the model's forward pass.

        Called in v1.worker.XXXModelRunner.execute_model immediately 
        after the model forward pass.

        Args:
            input_batch (InputBatch): The batch of input data being processed.
            device (torch.device): The device on which the computation is 
                performed.
            scheduler_output (SchedulerOutput): The output from the scheduler
                containing additional information for processing.
            logits (torch.Tensor): The raw logits output from the model's 
                forward pass.
            sample_hidden_states (torch.Tensor): The hidden states of the
                samples from the model's forward pass.
        """
        pass

    def precompile(self, dummy_logits: torch.Tensor, **kwargs):
        return

    @abstractmethod
    def supported_backends(self) -> list[str]:
        """
        Specify the StructuredOutputBackend's the worker Supports
        """
        pass
