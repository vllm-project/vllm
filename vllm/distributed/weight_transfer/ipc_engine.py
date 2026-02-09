# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for IPC weight transfer backend. No init needed for IPC."""

    pass


@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for IPC weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple[Callable, tuple]]]
    """IPC handles mapping physical GPU UUID to (func, args) tuple.
    Each handle is a dictionary mapping GPU UUID strings to IPC handle tuples."""

    def __post_init__(self):
        """Validate that all lists have the same length."""
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )
        if len(self.ipc_handles) != num_params:
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )


class IPCWeightTransferEngine(
    WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]
):
    """
    Weight transfer engine using CUDA IPC for communication between trainer and workers.

    This implementation uses CUDA IPC to transfer weights from the trainer (rank 0)
    to all inference workers in a process group. IPC handles are used to share
    memory between processes on the same node.
    """

    # Define backend-specific dataclass types
    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        """
        Initialize the IPC weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        super().__init__(config, parallel_config)

    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.
        No initialization needed for IPC backend.

        Args:
            init_info: IPC initialization info (empty)
        """
        pass

    def receive_weights(
        self,
        update_info: IPCWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """
        Receive weights from the trainer via CUDA IPC handles.

        Args:
            update_info: IPC update info containing parameter names, dtypes, shapes,
                        and IPC handles. Each IPC handle is a mapping between physical
                        GPU UUID and the IPC handle tuple (func, args).
            load_weights: Callable that loads weights into the model. Called
                         incrementally for each weight to avoid OOM.
        """
        weights = []
        for name, _dtype_name, _shape, ipc_handle in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.shapes,
            update_info.ipc_handles,
        ):
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            physical_gpu_id = str(props.uuid)

            if physical_gpu_id not in ipc_handle:
                raise ValueError(
                    f"IPC handle not found for GPU UUID {physical_gpu_id}. "
                    f"Available UUIDs: {list(ipc_handle.keys())}"
                )

            handle = ipc_handle[physical_gpu_id]

            func, args = handle
            list_args = list(args)  # type: ignore
            # Index 6 is the device_index parameter in torch's IPC handle tuple
            # (rebuild_cuda_tensor function signature). We need to update it to match
            # the current device where we're reconstructing the tensor.
            list_args[6] = device_index
            weight = func(*list_args)  # type: ignore
            weights.append((name, weight))

        load_weights(weights)

    def shutdown(self) -> None:
        """
        Shutdown the weight transfer engine.
        """
        pass
