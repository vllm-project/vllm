# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence and its related classes."""

from dataclasses import dataclass

import torch


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: dict[str, torch.Tensor]

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

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

    @staticmethod
    def empty_like(
        intermediate_tensors: "IntermediateTensors",
    ) -> "IntermediateTensors":
        tensors = {
            k: torch.empty_like(v) for k, v in intermediate_tensors.tensors.items()
        }
        return IntermediateTensors(tensors)


def get_intermediate_tensor_num_tokens(
    intermediate_tensors: IntermediateTensors,
) -> int:
    """Read the common leading dimension of nonempty, token-aligned PP tensors.

    Call after any residual reconstruction and token clipping. Intermediate
    tensors with different token layouts must be normalized before calling.
    """
    output_token_counts = {
        tensor.shape[0] for tensor in intermediate_tensors.tensors.values()
    }
    assert len(output_token_counts) == 1, (
        "Expected a common token count for PP intermediate tensors, got "
        f"{ {k: tuple(v.shape) for k, v in intermediate_tensors.items()} }"
    )
    return next(iter(output_token_counts))
