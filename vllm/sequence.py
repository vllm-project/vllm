# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence and its related classes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
else:
    KVConnectorOutput = Any


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.

    Each stage also needs to handle its own kv_connector_output.
    """

    tensors: dict[str, torch.Tensor]
    kv_connector_output: KVConnectorOutput | None

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> None:
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors
        self.kv_connector_output = kv_connector_output

    def __getitem__(self, key: str | slice):
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
        return all(torch.equal(self.tensors[k], other.tensors[k]) for k in self.tensors)

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"
