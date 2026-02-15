# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

import base64
import pickle
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import requests
import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


@dataclass
class IPCTrainerSendWeightsArgs:
    """Arguments for IPC trainer_send_weights method."""

    mode: str
    """Transport mode: 'http' or 'ray'."""
    llm_handle: Any = None
    """Ray ObjectRef to LLM handle (required for 'ray' mode)."""
    url: str | None = None
    """Base URL for HTTP endpoint (required for 'http' mode)."""

    def __post_init__(self):
        """Validate that required arguments are provided for the selected mode."""
        if self.mode == "ray" and self.llm_handle is None:
            raise ValueError("llm_handle is required for 'ray' mode")
        if self.mode == "http" and self.url is None:
            raise ValueError("url is required for 'http' mode")
        if self.mode not in ("ray", "http"):
            raise ValueError(f"mode must be 'ray' or 'http', got {self.mode}")


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

    def parse_update_info(self, update_dict: dict) -> IPCWeightTransferUpdateInfo:
        """
        Construct typed update info from dict with validation.

        Supports both direct IPC handles and pickled IPC handles
        (for HTTP transport). If 'ipc_handles_pickled' is present,
        it will be unpickled and used as 'ipc_handles'.

        Args:
            update_dict: Dictionary containing backend-specific update parameters

        Returns:
            Typed backend-specific update info dataclass

        Raises:
            ValueError: If update_dict is invalid for this backend
        """
        # Handle pickled IPC handles (used for HTTP transport)
        if "ipc_handles_pickled" in update_dict:
            pickled_data = update_dict.pop("ipc_handles_pickled")
            # Unpickle the IPC handles
            ipc_handles = pickle.loads(base64.b64decode(pickled_data))
            update_dict["ipc_handles"] = ipc_handles

        return super().parse_update_info(update_dict)

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

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | IPCTrainerSendWeightsArgs,
    ) -> None:
        """
        Send weights from trainer to inference workers via CUDA IPC.

        Supports two modes:
        - 'ray': Sends weights via Ray RPC to a Ray-based LLM handle
        - 'http': Sends weights via HTTP POST to a vLLM HTTP server

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor) tuples.
                     Tensors should be on the same GPU as the inference workers.
            trainer_args: Dictionary containing IPC-specific arguments.
                         Should contain keys from IPCTrainerSendWeightsArgs:
                         - mode: 'ray' or 'http'
                         - llm_handle: Ray ObjectRef (for 'ray' mode)
                         - url: Base URL string (for 'http' mode)

        Example (Ray mode):
            >>> from vllm.distributed.weight_transfer.ipc_engine import (
            ...     IPCWeightTransferEngine,
            ...     IPCTrainerSendWeightsArgs,
            ... )
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> args = IPCTrainerSendWeightsArgs(mode="ray", llm_handle=llm_handle)
            >>> IPCWeightTransferEngine.trainer_send_weights(param_iter, asdict(args))

        Example (HTTP mode):
            >>> args = IPCTrainerSendWeightsArgs(
            ...     mode="http", url="http://localhost:8000"
            ... )
            >>> IPCWeightTransferEngine.trainer_send_weights(param_iter, asdict(args))
        """
        # Parse trainer args - accept either dict or dataclass instance
        if isinstance(trainer_args, dict):
            args = IPCTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        # Get physical GPU UUID
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        gpu_uuid = str(props.uuid)

        # Collect weight metadata and create IPC handles
        names = []
        dtype_names = []
        shapes = []
        ipc_handles = []

        for name, tensor in iterator:
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            # Create IPC handle for this weight tensor
            # The tensor must remain in memory for IPC to work
            weight = tensor.detach().contiguous()
            ipc_handle = reduce_tensor(weight)
            ipc_handles.append({gpu_uuid: ipc_handle})

        # Send weights based on mode
        if args.mode == "ray":
            # Ray mode: send via Ray RPC
            import ray

            update_info = asdict(
                IPCWeightTransferUpdateInfo(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    ipc_handles=ipc_handles,
                )
            )
            ray.get(
                args.llm_handle.update_weights.remote(dict(update_info=update_info))
            )
        elif args.mode == "http":
            # HTTP mode: send via HTTP POST with pickled handles
            # Pickle and base64 encode IPC handles for HTTP transmission
            pickled_handles = base64.b64encode(pickle.dumps(ipc_handles)).decode(
                "utf-8"
            )

            url = f"{args.url}/update_weights"
            payload = {
                "update_info": {
                    "names": names,
                    "dtype_names": dtype_names,
                    "shapes": shapes,
                    "ipc_handles_pickled": pickled_handles,
                }
            }
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
