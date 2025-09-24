# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence and its related classes."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import msgspec
import torch

if TYPE_CHECKING:
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorOutput)
else:
    LoRARequest = Any
    KVConnectorOutput = Any

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

VLLM_INVALID_TOKEN_ID = -1


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
        scheduler_time: The time spent in the scheduler when this request was
                        being considered by the scheduler.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
    """
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None
    scheduler_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None


class PoolingSequenceGroupOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
):
    """The model output associated with a pooling sequence group."""
    # Annotated as Any to be compatible with msgspec
    # The actual type is in SequenceGroup.pooled_data
    data: Any

    def get_data_nbytes(self) -> int:
        data: torch.Tensor = self.data
        return data.nbytes

    def __repr__(self) -> str:
        return f"PoolingSequenceGroupOutput(data={self.data}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoolingSequenceGroupOutput):
            raise NotImplementedError()
        return self.data == other.data


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    
    Each stage also needs to handle its own kv_connector_output.
    """

    tensors: dict[str, torch.Tensor]
    kv_connector_output: Optional[KVConnectorOutput]

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        if not isinstance(other, self.__class__):
            return False
        if self.tensors.keys() != other.tensors.keys():
            return False
        return all(
            torch.equal(self.tensors[k], other.tensors[k])
            for k in self.tensors)

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


class PoolerOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """The output from a pooling operation in the pooling model."""
    outputs: list[PoolingSequenceGroupOutput]

    def get_data_nbytes(self) -> int:
        return sum(o.get_data_nbytes() for o in self.outputs)

    def __getitem__(self, idx: int) -> PoolingSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value: PoolingSequenceGroupOutput):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs


class ExecuteModelRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    # Placeholder. Remove.
    pass
