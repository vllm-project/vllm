"""Worker-local model inputs. These define the inputs to different model
runners."""
import dataclasses
from typing import Any, Dict, Optional, Type, Union

import torch


@dataclasses.dataclass(frozen=True)
class WorkerInput:
    """Local inputs to each worker. May contain device-specific data. Different
    worker backends may have different methods of converting from the global
    ExecuteModelRequest produced by the LLM engine to the worker-local
    WorkerInput objects.

    Subclasses of WorkerBase should inherit from this class and add their
    required fields.  For distributed executors, any fields that should be sent
    during a broadcast op should also be added to the broadcastable_fields.
    During execution, these fields will be extracted from the source copy and
    broadcasted to all workers using broadcast_tensor_dict.
    """

    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None

    @classmethod
    def _get_init_kwargs(cls: Type["WorkerInput"], **kwargs) -> Dict[str, Any]:
        """
        Helper method to extract all dataclass fields from the given kwargs.
        Override for fields that require some custom deserialization.
        """
        init_kwargs = {}
        for field in dataclasses.fields(cls):
            val = kwargs.get(field.name, None)
            if val is not None:
                init_kwargs[field.name] = val
        return init_kwargs

    @classmethod
    def new(cls: Type["WorkerInput"], **kwargs) -> "WorkerInput":
        """
        Create a new instance of this class. Populate the new instance with
        the given kwargs.
        """
        kwargs = cls._get_init_kwargs(**kwargs)
        return cls(**kwargs)

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        tensor_dict: Dict[str, Union[int, torch.Tensor]] = {}
        for field in dataclasses.fields(self):
            val = getattr(self, field.name, None)
            if val is not None:
                tensor_dict[field.name] = val

        return tensor_dict
