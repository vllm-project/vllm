from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

import torch

from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.model_input import ModelInput

T = TypeVar('T', bound="ModelInput")


class ModelRunnerBase(ABC, Generic[T]):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model. Model execution may communicate data with model runners in other
    processes, but it should not include control plane metadata communication.

    Each ModelRunnerBase subclass should define a corresponding ModelInput
    subclass.
    """

    @abstractmethod
    def make_model_input(self, **model_input_fields) -> T:
        """
        Make an instance of a ModelInput from the given
        fields.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> T:
        """
        Prepare the inputs to ModelRunnerBase.execute_model from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: T,
        kv_caches: Optional[List[torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        """
        Execute the model on the given input.
        """
        raise NotImplementedError
