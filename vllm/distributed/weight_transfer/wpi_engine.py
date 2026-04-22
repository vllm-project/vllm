# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WPI-based weight transfer engine using the Weight Propagation Interface.

The WPI (Weight Propagation Interface) driver manages a pre-allocated, persistent
VRAM buffer on each node and uses NCCL broadcast (NodePropagate) to distribute
weights across nodes at InfiniBand/RDMA speeds. This engine integrates WPI into
vLLM's WeightTransferEngine framework, providing zero-copy weight updates for
RLHF and online training workflows.

Architecture:
    Trainer (send side)                 vLLM Worker (receive side)
    ┌──────────────────┐                ┌───────────────────────┐
    │ trainer_init()   │                │ init_transfer_engine() │
    │  → stage_weight  │                │  → stage_weight        │
    │  → receive FD    │                │  → receive FD          │
    │  → CUDA import   │                │  → CUDA import         │
    │                  │                │  → connect notify sock │
    │ trainer_send_wts │                │                        │
    │  → pack into buf │  NodePropagate │ receive_weights()      │
    │  → propagate ────┼───(NCCL)──────→│  → wait_for_ready      │
    │  → HTTP metadata │                │  → unpack from buffer  │
    │    /update_weights├───(HTTP)─────→│  → load_weights()      │
    └──────────────────┘                └───────────────────────┘

Key differences from NCCL/IPC engines:
- NCCL communicator is managed by the WPI driver, not by vLLM
- VRAM buffer is persistent and reused across training steps (no per-update alloc)
- FD-based memory sharing via SCM_RIGHTS (cross-container zero-copy on same node)
- Supports sharded scatter for tensor-parallel deployments
"""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class WPITrainerContext:
    """Holds trainer-side WPI state for reuse across weight updates.

    Created by ``WPIWeightTransferEngine.trainer_init()`` and passed
    into ``trainer_send_weights()`` via ``trainer_args``.
    """

    client: Any  # WPIClient instance
    vram_buffer: torch.Tensor
    buffer_id: str
    buffer_size_bytes: int
    target_node_ids: list[str]


@dataclass
class WPITrainerSendWeightsArgs:
    """Arguments for WPI trainer_send_weights method."""

    mode: str
    """Transport mode: 'http' or 'ray'."""

    buffer_id: str = "vllm-weights"
    """WPI buffer identifier."""

    buffer_size_bytes: int = 0
    """Total size for the VRAM buffer in bytes."""

    socket_dir: str = "/run/wpi/sockets"
    """Path to WPI UNIX socket directory."""

    driver_port: int = 50051
    """WPI driver gRPC port."""

    target_node_ids: list[str] = field(default_factory=list)
    """List of target node IPs for NodePropagate."""

    shard_index: int = -1
    """Shard index for tensor-parallel (-1 = unsharded)."""

    total_shards: int = 0
    """Total number of shards (0 = unsharded)."""

    # Pre-initialized trainer context (from trainer_init())
    trainer_ctx: WPITrainerContext | None = None
    """Pre-initialized trainer context. If None, one will be created."""

    # Transport-specific fields
    llm_handle: Any = None
    """Ray ObjectRef to LLM handle (required for 'ray' mode)."""

    url: str | None = None
    """Base URL for HTTP endpoint (required for 'http' mode)."""

    def __post_init__(self):
        if self.mode == "ray" and self.llm_handle is None:
            raise ValueError("llm_handle is required for 'ray' mode")
        if self.mode == "http" and self.url is None:
            raise ValueError("url is required for 'http' mode")
        if self.mode not in ("ray", "http"):
            raise ValueError(f"mode must be 'ray' or 'http', got {self.mode}")


@dataclass
class WPIWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for WPI weight transfer backend.

    Sent to each vLLM worker via ``/init_weight_transfer_engine`` to set up
    the persistent VRAM buffer and WPI driver connections.
    """

    buffer_id: str = "vllm-weights"
    """WPI buffer identifier. Must be consistent across trainer and workers."""

    buffer_size_bytes: int = 0
    """Total VRAM buffer size in bytes. Must hold all model parameters."""

    socket_dir: str = "/run/wpi/sockets"
    """Path to WPI UNIX socket directory on the node."""

    driver_port: int = 50051
    """WPI driver gRPC port."""

    shard_index: int = -1
    """Shard index for tensor-parallel (-1 = unsharded).
    When >= 0, the worker stages only its assigned shard."""

    total_shards: int = 0
    """Total number of shards (0 = unsharded)."""


@dataclass
class WPIWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for WPI weight transfer backend.

    Contains the metadata (names, shapes, dtypes, offsets) needed to unpack
    tensors from the flat VRAM buffer after a WPI broadcast completes.
    """

    names: list[str] = field(default_factory=list)
    """Parameter names in packing order."""

    dtype_names: list[str] = field(default_factory=list)
    """Dtype strings (e.g., 'bfloat16') for each parameter."""

    shapes: list[list[int]] = field(default_factory=list)
    """Shapes for each parameter."""

    offsets: list[int] = field(default_factory=list)
    """Byte offsets into the flat VRAM buffer for each parameter."""

    total_bytes: int = 0
    """Total bytes packed into the buffer."""

    def __post_init__(self):
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {num_params}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {num_params}"
            )
        if len(self.offsets) != num_params:
            raise ValueError(
                f"`offsets` should be of the same size as `names`: "
                f"got {len(self.offsets)} and {num_params}"
            )


def _import_wpi_client():
    """Lazily import WPIClient with a clear error message."""
    try:
        from wpi_client.client import WPIClient
        return WPIClient
    except ImportError:
        raise ImportError(
            "WPI weight transfer backend requires the `wpi_verl_plugin` package. "
            "Install it with: pip install wpi_verl_plugin\n"
            "Or from the WPI source: cd weight-propagation-interface/consumer/"
            "wpi_verl_plugin && pip install -e ."
        ) from None


class WPIWeightTransferEngine(
    WeightTransferEngine[WPIWeightTransferInitInfo, WPIWeightTransferUpdateInfo]
):
    """Weight transfer engine using the Weight Propagation Interface (WPI).

    This engine leverages the WPI driver's pre-allocated, persistent VRAM buffer
    and NCCL broadcast (NodePropagate) for high-throughput cross-node weight
    transfer. Unlike the NCCL engine where vLLM workers participate directly
    in NCCL collectives, the WPI driver manages NCCL internally — workers
    simply wait for a READY notification and then read from shared GPU memory.

    Lifecycle:
        1. init_transfer_engine() — stage buffer, map CUDA memory, connect socket
        2. receive_weights() — wait for READY, unpack from buffer, load into model
        3. shutdown() — release resources

    Performance:
        WPI achieves 251 GB/s aggregate cross-node throughput on A4 nodes
        (8-shard scatter over InfiniBand with GPUDirect RDMA).
    """

    init_info_cls = WPIWeightTransferInitInfo
    update_info_cls = WPIWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)
        self._client = None  # WPIClient, created during init_transfer_engine
        self._vram_buffer: torch.Tensor | None = None
        self._buffer_id: str = ""
        self._buffer_size: int = 0
        self._staged: bool = False

    def init_transfer_engine(
        self, init_info: WPIWeightTransferInitInfo
    ) -> None:
        """Initialize WPI driver connection and stage the persistent VRAM buffer.

        This is called once when the trainer sends an init request. It:
        1. Creates a WPIClient and connects to the local WPI driver
        2. Calls NodeStageWeight to allocate a VRAM buffer (empty receive buffer)
        3. Receives the CUDA memory FD via SCM_RIGHTS UNIX socket
        4. Imports the FD as mapped CUDA memory (cuMemImportFromShareableHandle)
        5. Connects to the notify socket for READY signals

        Args:
            init_info: WPI initialization info with buffer ID, size, and driver addr
        """
        WPIClient = _import_wpi_client()

        self._buffer_id = init_info.buffer_id
        self._buffer_size = init_info.buffer_size_bytes
        shard_index = init_info.shard_index
        total_shards = init_info.total_shards

        # Map tp_rank to shard_index if sharding is requested but index not set
        if total_shards > 0 and shard_index < 0:
            shard_index = self.parallel_config.rank
            logger.info(
                "WPI: Auto-mapping tp_rank=%d to shard_index=%d",
                self.parallel_config.rank, shard_index,
            )

        self._client = WPIClient(
            socket_dir=init_info.socket_dir,
            driver_port=init_info.driver_port,
        )

        # Stage the empty receive buffer on the local WPI driver
        if not self._staged:
            self._client.stage_weight(
                buffer_id=self._buffer_id,
                size_bytes=self._buffer_size,
                claim_id=f"{self._buffer_id}-claim",
                shard_index=shard_index,
                total_shards=total_shards,
            )
            self._staged = True

        # Receive FD and import CUDA memory
        device_index = torch.accelerator.current_device_index()
        fd = self._client.receive_fd(
            self._buffer_id,
            gpu_id=device_index,
            shard_index=shard_index,
            total_shards=total_shards,
        )
        device_ptr = self._client.import_cuda_memory(
            fd, self._buffer_size, device_id=device_index,
        )
        self._vram_buffer = self._client.wrap_as_buffer(
            device_ptr, self._buffer_size,
        )

        # Connect to the notify socket for READY signals from NodePropagate
        self._client.connect_notify_socket(
            self._buffer_id,
            shard_index=shard_index,
            total_shards=total_shards,
        )

        logger.info(
            "WPI: Engine initialized — buffer=%s, size=%d bytes, "
            "device=%s, shard=%d/%d",
            self._buffer_id, self._buffer_size,
            self._vram_buffer.device, shard_index, total_shards,
        )

    def receive_weights(
        self,
        update_info: WPIWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Receive weights from the WPI VRAM buffer after a broadcast.

        This method:
        1. Waits for READY notification from the WPI driver (signals NCCL done)
        2. Unpacks individual tensors from the flat VRAM buffer using offset metadata
        3. Calls load_weights() incrementally for each parameter

        The VRAM buffer is NOT freed — it persists for the next weight update.

        Args:
            update_info: Metadata (names, shapes, dtypes, offsets) for unpacking
            load_weights: Callable that loads weights into the model
        """
        if self._vram_buffer is None or self._client is None:
            raise RuntimeError(
                "WPI engine not initialized. "
                "Call init_transfer_engine() first."
            )

        # Wait for the WPI driver to signal that NCCL broadcast is complete
        self._client.wait_for_ready(timeout=300.0)

        # Unpack tensors from the flat VRAM buffer and load incrementally
        for name, dtype_name, shape, offset in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.shapes,
            update_info.offsets,
        ):
            dtype = getattr(torch, dtype_name)
            num_elements = math.prod(shape)
            nbytes = num_elements * dtype.itemsize

            # Slice the flat uint8 buffer, reinterpret as the target dtype/shape
            weight = (
                self._vram_buffer[offset:offset + nbytes]
                .view(dtype=dtype)
                .view(shape)
            )
            load_weights([(name, weight)])

    def shutdown(self) -> None:
        """Shutdown the WPI engine and release resources.

        The VRAM buffer is owned by the WPI driver, so we only clean up
        our client connections. The driver retains the buffer based on
        its retentionPolicy.
        """
        if self._client is not None:
            if self._staged:
                try:
                    self._client.unstage_weight(f"{self._buffer_id}-claim")
                except Exception as e:
                    logger.warning("WPI: Error during unstage: %s", e)
                self._staged = False
            self._client.close()
            self._client = None

        self._vram_buffer = None
        logger.info("WPI: Engine shutdown complete")

    @staticmethod
    def trainer_init(
        init_info: WPIWeightTransferInitInfo | dict,
        target_node_ids: list[str] | None = None,
    ) -> WPITrainerContext:
        """Initialize the trainer-side WPI client and VRAM buffer.

        Call this once on the trainer before the first weight update.
        The returned context is passed to ``trainer_send_weights()`` via
        ``trainer_args["trainer_ctx"]``.

        Args:
            init_info: WPI init info (dict or dataclass) with buffer_id, size, etc.
            target_node_ids: IPs of the vLLM inference nodes for NodePropagate.

        Returns:
            WPITrainerContext with the initialized client and mapped VRAM buffer.

        Example:
            >>> ctx = WPIWeightTransferEngine.trainer_init(
            ...     dict(buffer_id="vllm-wts", buffer_size_bytes=20*1024**3,
            ...          socket_dir="/run/wpi/sockets"),
            ...     target_node_ids=["10.0.0.2", "10.0.0.3"],
            ... )
        """
        WPIClient = _import_wpi_client()

        if isinstance(init_info, dict):
            buffer_id = init_info.get("buffer_id", "vllm-weights")
            buffer_size_bytes = init_info["buffer_size_bytes"]
            socket_dir = init_info.get("socket_dir", "/run/wpi/sockets")
            driver_port = init_info.get("driver_port", 50051)
            shard_index = init_info.get("shard_index", -1)
            total_shards = init_info.get("total_shards", 0)
        else:
            buffer_id = init_info.buffer_id
            buffer_size_bytes = init_info.buffer_size_bytes
            socket_dir = init_info.socket_dir
            driver_port = init_info.driver_port
            shard_index = init_info.shard_index
            total_shards = init_info.total_shards

        client = WPIClient(socket_dir=socket_dir, driver_port=driver_port)

        # Stage the source buffer on the trainer's WPI driver
        client.stage_weight(
            buffer_id=buffer_id,
            size_bytes=buffer_size_bytes,
            claim_id=f"{buffer_id}-trainer-claim",
            shard_index=shard_index,
            total_shards=total_shards,
        )

        # Receive FD and import CUDA memory on trainer GPU
        import torch as _torch
        device_index = _torch.accelerator.current_device_index()
        fd = client.receive_fd(
            buffer_id, gpu_id=device_index,
            shard_index=shard_index, total_shards=total_shards,
        )
        device_ptr = client.import_cuda_memory(
            fd, buffer_size_bytes, device_id=device_index,
        )
        vram_buffer = client.wrap_as_buffer(device_ptr, buffer_size_bytes)

        logger.info(
            "WPI: Trainer initialized — buffer=%s, size=%d, "
            "targets=%s",
            buffer_id, buffer_size_bytes, target_node_ids,
        )

        return WPITrainerContext(
            client=client,
            vram_buffer=vram_buffer,
            buffer_id=buffer_id,
            buffer_size_bytes=buffer_size_bytes,
            target_node_ids=target_node_ids or [],
        )

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | WPITrainerSendWeightsArgs,
    ) -> None:
        """Send weights from trainer to vLLM workers via WPI.

        This method:
        1. Packs all weight tensors into the WPI shared VRAM buffer
        2. Calls NodePropagate to trigger the WPI driver's NCCL broadcast
        3. Sends metadata (offsets, shapes, dtypes) to vLLM via HTTP or Ray

        The WPI driver handles NCCL communicator setup, broadcast execution,
        and READY notification to consumers — all transparently.

        Args:
            iterator: Iterator of (name, tensor) tuples from the model.
            trainer_args: Dict or WPITrainerSendWeightsArgs with WPI config.

        Example (HTTP mode):
            >>> from vllm.distributed.weight_transfer.wpi_engine import (
            ...     WPIWeightTransferEngine,
            ...     WPITrainerSendWeightsArgs,
            ... )
            >>> # One-time init
            >>> ctx = WPIWeightTransferEngine.trainer_init(
            ...     dict(buffer_id="wts", buffer_size_bytes=20*1024**3),
            ...     target_node_ids=["10.0.0.2"],
            ... )
            >>> # Per-step weight sync
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> args = WPITrainerSendWeightsArgs(
            ...     mode="http",
            ...     url="http://vllm-server:8000",
            ...     trainer_ctx=ctx,
            ... )
            >>> WPIWeightTransferEngine.trainer_send_weights(param_iter, args)
        """
        # Parse trainer args
        if isinstance(trainer_args, dict):
            args = WPITrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        # Get or create the trainer context
        ctx = args.trainer_ctx
        if ctx is None:
            # Lazy initialization (creates WPI client + buffer on first call)
            ctx = WPIWeightTransferEngine.trainer_init(
                dict(
                    buffer_id=args.buffer_id,
                    buffer_size_bytes=args.buffer_size_bytes,
                    socket_dir=args.socket_dir,
                    driver_port=args.driver_port,
                    shard_index=args.shard_index,
                    total_shards=args.total_shards,
                ),
                target_node_ids=args.target_node_ids,
            )

        # --- Step 1: Pack weights into the flat VRAM buffer ---
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        offsets: list[int] = []
        offset = 0

        for name, tensor in iterator:
            weight = tensor.detach().contiguous()
            nbytes = weight.nbytes

            if offset + nbytes > ctx.buffer_size_bytes:
                raise RuntimeError(
                    f"Weight {name} ({weight.shape}, {weight.dtype}) would "
                    f"exceed WPI buffer. Current offset: {offset}, "
                    f"weight size: {nbytes}, buffer: {ctx.buffer_size_bytes}. "
                    f"Increase buffer_size_bytes."
                )

            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))
            offsets.append(offset)

            # Copy into the shared VRAM buffer
            ctx.vram_buffer[offset:offset + nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True,
            )
            offset += nbytes

        # Synchronize to ensure all copies are complete before broadcast
        torch.cuda.synchronize()

        total_bytes = offset
        logger.info(
            "WPI trainer: packed %d params (%d bytes) into buffer",
            len(names), total_bytes,
        )

        # --- Step 2: Trigger NCCL broadcast via WPI driver ---
        ctx.client.propagate(
            buffer_id=ctx.buffer_id,
            target_node_ids=ctx.target_node_ids,
        )

        logger.info("WPI trainer: NodePropagate complete")

        # --- Step 3: Send metadata to vLLM workers ---
        from dataclasses import asdict

        update_info = asdict(WPIWeightTransferUpdateInfo(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            offsets=offsets,
            total_bytes=total_bytes,
        ))

        if args.mode == "ray":
            import ray
            ray.get(
                args.llm_handle.update_weights.remote(
                    dict(update_info=update_info)
                )
            )
        elif args.mode == "http":
            import requests
            url = f"{args.url}/update_weights"
            payload = {"update_info": update_info}
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

        logger.info(
            "WPI trainer: weight update delivered (%d params, %.1f MB)",
            len(names), total_bytes / (1024 * 1024),
        )
