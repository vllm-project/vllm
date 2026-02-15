# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based weight transfer engine."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_consumer,
)


@dataclass
class NCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for NCCL weight transfer backend."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


@dataclass
class NCCLTrainerSendWeightsArgs:
    """Arguments for NCCL trainer_send_weights method."""

    group: Any
    """Process group (PyNcclCommunicator) for NCCL communication."""
    src: int = 0
    """Source rank (default 0, trainer is typically rank 0)."""
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = None
    """Optional function to apply to each (name, tensor) pair before broadcasting.
    If None, extracts just the tensor."""
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency.
    When True, multiple tensors are batched together before broadcasting
    to reduce NCCL communication overhead."""
    stream: torch.cuda.Stream | None = None
    """CUDA stream to use for broadcasting if packed is False.
    If packed is True, new streams will be created for each buffer."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer. Default is 1GB.
    Must match the value used in NCCLWeightTransferUpdateInfo."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed transfer.
    Must match the value used in NCCLWeightTransferUpdateInfo."""


@dataclass
class NCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for NCCL weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency.
    When True, multiple tensors are batched together before broadcasting
    to reduce NCCL communication overhead."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer. Default is 1GB.
    Both producer and consumer must use the same value."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed transfer.
    Both producer and consumer must use the same value."""

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


class NCCLWeightTransferEngine(
    WeightTransferEngine[NCCLWeightTransferInitInfo, NCCLWeightTransferUpdateInfo]
):
    """
    Weight transfer engine using NCCL for communication between trainer and workers.

    This implementation uses NCCL broadcast operations to transfer weights from
    the trainer (rank 0) to all inference workers in a process group.
    """

    # Define backend-specific dataclass types
    init_info_cls = NCCLWeightTransferInitInfo
    update_info_cls = NCCLWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        """
        Initialize the NCCL weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        super().__init__(config, parallel_config)
        self.model_update_group: PyNcclCommunicator | None = None

    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """
        Initialize NCCL process group with the trainer.

        Args:
            init_info: NCCL initialization info containing master address, port,
                      rank offset, and world size
        """

        # Calculate the global rank in the trainer-worker process group
        # Must account for data parallel to get unique ranks across all workers
        dp_rank = self.parallel_config.data_parallel_rank
        world_size_per_dp = self.parallel_config.world_size  # TP * PP
        rank_within_dp = self.parallel_config.rank

        # Unique rank across all DP groups
        worker_rank = dp_rank * world_size_per_dp + rank_within_dp
        rank = worker_rank + init_info.rank_offset
        # Create stateless process group
        self.model_update_group = (
            NCCLWeightTransferEngine._stateless_init_process_group(
                init_info.master_address,
                init_info.master_port,
                rank,
                init_info.world_size,
                torch.cuda.current_device(),
            )
        )

    def receive_weights(
        self,
        update_info: NCCLWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """
        Receive weights from trainer via NCCL broadcast and load them incrementally.

        If update_info.packed is True, uses packed tensor broadcasting for
        efficient transfer of multiple weights in batches. Otherwise, uses simple
        one-by-one broadcasting.

        Args:
            update_info: NCCL update info containing parameter names, dtypes, shapes,
                        and packed flag
            load_weights: Callable that loads weights into the model. Called
                         incrementally for each batch of weights to avoid OOM.
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        if update_info.packed:
            # Build iterator of (name, (shape, dtype)) from update_info
            def state_dict_info_iterator():
                for name, dtype_name, shape in zip(
                    update_info.names, update_info.dtype_names, update_info.shapes
                ):
                    dtype = getattr(torch, dtype_name)
                    yield (name, (shape, dtype))

            packed_broadcast_consumer(
                iterator=state_dict_info_iterator(),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_weights,
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
                num_buffers=update_info.packed_num_buffers,
            )
        else:
            # Use simple one-by-one broadcasting
            for name, dtype_name, shape in zip(
                update_info.names, update_info.dtype_names, update_info.shapes
            ):
                dtype = getattr(torch, dtype_name)
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                self.model_update_group.broadcast(
                    weight, src=0, stream=torch.cuda.current_stream()
                )
                load_weights([(name, weight)])
                del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # Clean up the communicator by removing the reference
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | NCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast weights from trainer to vLLM workers.

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor) tuples
            trainer_args: Dictionary or NCCLTrainerSendWeightsArgs instance containing
                         NCCL-specific arguments. If a dict, should contain keys from
                         NCCLTrainerSendWeightsArgs. Note: 'group' and 'stream' fields
                         contain non-serializable objects and should be passed directly.

        Example:
            >>> from vllm.distributed.weight_transfer.nccl_engine import (
            ...     NCCLWeightTransferEngine,
            ...     NCCLTrainerSendWeightsArgs,
            ... )
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> args = NCCLTrainerSendWeightsArgs(group=group, packed=True)
            >>> NCCLWeightTransferEngine.trainer_send_weights(param_iter, args)
        """
        # Parse trainer args - accept either dict or dataclass instance
        if isinstance(trainer_args, dict):
            args = NCCLTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        if args.post_iter_func is None:
            # Default: extract just the tensor from (name, tensor) tuple
            post_iter_func = lambda x: x[1]
        else:
            post_iter_func = args.post_iter_func

        if args.packed:
            # Use packed tensor broadcasting for efficiency
            from vllm.distributed.weight_transfer.packed_tensor import (
                packed_broadcast_producer,
            )

            packed_broadcast_producer(
                iterator=iterator,
                group=args.group,
                src=args.src,
                post_iter_func=post_iter_func,
                buffer_size_bytes=args.packed_buffer_size_bytes,
                num_buffers=args.packed_num_buffers,
            )
        else:
            # Use simple one-by-one broadcasting
            for item in iterator:
                tensor = post_iter_func(item)
                args.group.broadcast(
                    tensor,
                    src=args.src,
                    stream=args.stream or torch.cuda.current_stream(),
                )

    @staticmethod
    def trainer_init(
        init_info: NCCLWeightTransferInitInfo | dict,
    ) -> "PyNcclCommunicator":
        """
        Initialize NCCL process group for trainer-side weight transfer.

        The trainer is always rank 0 in the process group. Uses the current
        CUDA device (torch.cuda.current_device()).

        Args:
            init_info: Either an NCCLWeightTransferInitInfo object or a dict with keys:
                - master_address: str
                - master_port: int
                - world_size: int

        Returns:
            PyNcclCommunicator for weight transfer.

        Example:
            >>> from vllm.distributed.weight_transfer.nccl_engine import (
            ...     NCCLWeightTransferEngine,
            ... )
            >>> group = NCCLWeightTransferEngine.trainer_init(
            ...     dict(
            ...         master_address=master_address,
            ...         master_port=master_port,
            ...         world_size=world_size,
            ...     ),
            ... )
        """
        if isinstance(init_info, dict):
            master_address = init_info["master_address"]
            master_port = init_info["master_port"]
            world_size = init_info["world_size"]
        else:
            # NCCLWeightTransferInitInfo object
            master_address = init_info.master_address
            master_port = init_info.master_port
            world_size = init_info.world_size

        # Trainer is always rank 0
        return NCCLWeightTransferEngine._stateless_init_process_group(
            master_address, master_port, 0, world_size, torch.cuda.current_device()
        )

    @staticmethod
    def _stateless_init_process_group(
        master_address, master_port, rank, world_size, device
    ):
        """
        vLLM provides `StatelessProcessGroup` to create a process group
        without considering the global process group in torch.distributed.
        It is recommended to create `StatelessProcessGroup`, and then initialize
        the data-plane communication (NCCL) between external (train processes)
        and vLLM workers.
        """
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )
        pynccl = PyNcclCommunicator(pg, device=device)
        return pynccl
