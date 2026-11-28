# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

import pickle
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import pybase64 as base64
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm import envs
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.device_allocator.alloc_conf import expandable_segments_enabled
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    PackedIpcConsumer,
    packed_ipc_producer,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for IPC weight transfer; the trainer ships
    `packed` here at the handshake."""

    packed: bool = False


@dataclass
class IPCTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for IPC weight transfer. Rank 0 ships the
    merged IPC handles and the chunk size with the first chunk."""

    backend: ClassVar[str] = "ipc"
    packed: bool = False
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES


@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the IPC weight transfer backend."""

    names: list[str]
    """Packed mode: the tensors that start in this chunk (`PackedIpcChunk`)."""
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple] | None = None
    """IPC handles mapping physical GPU UUID to rebuild_cuda_tensor args.
    For non-packed mode: list of per-parameter handle dicts.
    For packed mode: the packed buffer's handle dict in the first chunk of a
    transfer with bytes, else empty."""
    ipc_handles_pickled: str | None = None
    """Base64-encoded pickled IPC handles, used for HTTP transport."""
    chunk_capacity: int | None = None
    """Packed mode: the chunk size in bytes, set on the first chunk."""
    last: bool = False
    """Packed mode: whether this chunk ends the transfer."""

    def __post_init__(self):
        if self.ipc_handles_pickled is not None:
            if self.ipc_handles is not None:
                raise ValueError(
                    "Cannot specify both `ipc_handles` and `ipc_handles_pickled`"
                )

            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize `ipc_handles_pickled` without "
                    "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )

            self.ipc_handles = pickle.loads(base64.b64decode(self.ipc_handles_pickled))
            self.ipc_handles_pickled = None

        if self.ipc_handles is None:
            raise ValueError(
                "Either `ipc_handles` or `ipc_handles_pickled` must be provided"
            )
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
        # Unpacked transfers carry a per-parameter handle dict (a list); packed
        # transfers carry a single dict for the whole buffer, so the list check
        # only applies to the list case.
        if isinstance(self.ipc_handles, list) and len(self.ipc_handles) != num_params:
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )


class IPCWeightTransferEngine(
    WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]
):
    """Weight transfer engine using CUDA IPC between trainer and workers.

    This implementation uses CUDA IPC to transfer weights from the trainer (rank 0)
    to all inference workers in a process group. IPC handles are used to share
    memory between processes on the same node.
    """

    # Define backend-specific dataclass types
    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        # Set from the trainer-supplied init info at the handshake.
        self.packed = False
        # The current packed transfer, between start and finish.
        self._packed_consumer: PackedIpcConsumer | None = None

    def _local_gpu(self) -> tuple[str, int]:
        """This worker's GPU UUID and device index, for rebuilding handles."""
        device_index = self.device.index
        return str(torch.cuda.get_device_properties(device_index).uuid), device_index

    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
        """Initialize the weight transfer mechanism. No data-plane rendezvous is
        needed for IPC; this just records the trainer-supplied wire params so the
        worker decodes exactly as the trainer encoded.

        Args:
            init_info: IPC initialization info (carries `packed`).

        """
        self.packed = init_info.packed

    def start_weight_update(self) -> None:
        """Begin reloading the incoming checkpoint weights."""
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)
        if self.packed:
            gpu_uuid, device_index = self._local_gpu()
            self._packed_consumer = PackedIpcConsumer(device_index, gpu_uuid)

    def finish_weight_update(self) -> None:
        """Finish reloading after all weights have been received."""
        from vllm.model_executor.model_loader.reload import finish_reload

        consumer, self._packed_consumer = self._packed_consumer, None
        if consumer is not None:
            consumer.close()  # raises if the transfer was cut short
        finish_reload(self.model, self.model_config, self.device)

    def receive_weights(self, update_info: IPCWeightTransferUpdateInfo) -> None:
        """Receive weights from the trainer via CUDA IPC handles and load them.

        Whether the transfer is packed is read from `self.packed`, set at the
        init handshake from the trainer's init info, so it is guaranteed to
        match how the trainer encoded.

        Args:
            update_info: IPC update info containing parameter names, dtypes, shapes,
                        and IPC handles. Each IPC handle is a mapping between physical
                        GPU UUID and the rebuild_cuda_tensor args tuple.

        """
        # Use the worker's assigned device rather than the ambient current
        # device: the receive path is no longer wrapped in
        # `with torch.device(self.device)` by the caller, so the current device
        # is not guaranteed to match self.device. The IPC tensors must be
        # rebuilt on the device the model lives on.
        device_index = self.device.index

        if self.packed:
            assert self._packed_consumer is not None, (
                "packed IPC receive_weights called outside "
                "start_weight_update / finish_weight_update"
            )
            assert isinstance(update_info.ipc_handles, dict)
            weights = self._packed_consumer.receive(
                ipc_handle=update_info.ipc_handles,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                chunk_capacity=update_info.chunk_capacity,
                last=update_info.last,
            )
            if not weights:  # a chunk may complete no tensor
                return
        else:
            assert isinstance(update_info.ipc_handles, list)
            weights = []
            for name, ipc_handle in zip(
                update_info.names,
                update_info.ipc_handles,
            ):
                props = torch.cuda.get_device_properties(device_index)
                physical_gpu_id = str(props.uuid)

                if physical_gpu_id not in ipc_handle:
                    raise ValueError(
                        f"IPC handle not found for GPU UUID "
                        f"{physical_gpu_id}. "
                        f"Available UUIDs: {list(ipc_handle.keys())}"
                    )

                args = ipc_handle[physical_gpu_id]
                list_args = list(args)
                # Index 6 of the args from reduce_tensor is the device_index.
                # We need to overwrite it with the receiver's device index.
                list_args[6] = device_index
                weight = rebuild_cuda_tensor(*list_args)
                weights.append((name, weight))

        self.model.load_weights(weights)

    def shutdown(self) -> None:
        self._packed_consumer = None


class IPCTrainerWeightTransferEngine(TrainerWeightTransferEngine[IPCTrainerInitInfo]):
    """Trainer-side CUDA IPC weight transfer engine.

    Called on every trainer rank. For multi-rank (e.g. FSDP) trainers all ranks
    drive the source's `iter_full()` (materializing each tensor) and contribute
    to the IPC-handle all-gather, so every rank must own every parameter; only
    the sender (rank 0) ships the merged handles to the inference side. IPC
    transfer is straight-line (no concurrent broadcast like NCCL):
    `update_weights` *is* the transfer, and it rides the client, so it no-ops
    on non-senders.

    `packed` / `packed_buffer_size_bytes` come from `IPCTrainerInitInfo`; the
    sender propagates `packed` to the worker at `trainer_init`.
    """

    init_info_cls = IPCTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.device_index = torch.accelerator.current_device_index()
        self.gpu_uuid = str(torch.cuda.get_device_properties(self.device_index).uuid)

    @classmethod
    def trainer_init(
        cls,
        init_info: IPCTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("IPC trainer weight transfer requires a WeightSource.")
        if not source.implements_iter_full():
            raise TypeError(
                f"IPC sends full tensors, but {type(source).__name__} does not "
                "implement iter_full()."
            )
        # Each rank exports its own copies for its colocated worker, merged by
        # position, so every rank must hold every parameter.
        if len(source.owned_names(init_info.rank)) != len(source.metadata):
            raise ValueError(
                f"rank {init_info.rank} does not own every parameter, but IPC "
                "needs every trainer rank to own every parameter."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
        )
        # IPC needs no data-plane rendezvous. The sender ships the must-agree
        # `packed` flag so the worker decodes exactly as this trainer encodes.
        if engine.is_sender:
            engine.client.init_weight_transfer_engine({"packed": init_info.packed})
        return engine

    def send_weights(self) -> None:
        assert self.source is not None
        source = self.source
        if self.is_sender:
            self.client.start_weight_update()
        # Unpacked returns strong refs to its IPC-shared copies; they must stay
        # alive across `finish_weight_update` too.
        weight_refs = self._send(source)
        if self.is_sender:
            self.client.finish_weight_update()
        self._post_send_sync()
        del weight_refs

    # ---- data plane (runs on all ranks; only the sender ships) ----

    def _send(self, source: WeightSource) -> list[torch.Tensor] | None:
        if self.packed:
            self._send_packed(source)
            return None
        return self._send_unpacked(source)

    def _all_gather_and_merge_handles(
        self,
        handles: list[dict[str, tuple]],
    ) -> list[dict[str, tuple]]:
        """Merge each index's {gpu_uuid: ipc_args} dicts across the trainer
        ranks (the default group) on the sender; no-op without a group."""
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return handles

        world_size = torch.distributed.get_world_size()
        gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, handles)
        torch.distributed.barrier()
        torch.accelerator.synchronize()

        if self.is_sender:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(len(handles)):
                m: dict[str, tuple] = {}
                for rank_handles in gathered:
                    if rank_handles is not None:
                        m.update(rank_handles[param_idx])
                merged.append(m)
            return merged
        return [{} for _ in handles]

    @staticmethod
    def _barrier() -> None:
        """Barrier over the trainer ranks; no-op if single-GPU."""
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()

    @classmethod
    def _post_send_sync(cls) -> None:
        """Barrier + ipc_collect after a send."""
        cls._barrier()
        torch.cuda.ipc_collect()

    def _send_unpacked(self, source: WeightSource) -> list[torch.Tensor]:
        """Iterate the source, build one IPC handle per param, all-gather the
        handles across ranks, and (sender) ship them in one update call.

        Returns the strong refs to every contiguous copy. reduce_tensor's args
        do NOT keep storage alive, and non-contiguous inputs allocate fresh
        storage in .contiguous(); the caller must keep these alive until the
        post-send barrier (past `finish`) so the consumer's IPC views stay valid.
        """
        if expandable_segments_enabled():
            logger.warning_once(
                "Sharing weights over CUDA IPC from expandable segments: the "
                "receiver must open them with pidfd_getfd, which fails with "
                "'Operation not permitted' under the default Yama ptrace_scope. "
                "Use IPCTrainerInitInfo(packed=True), which shares a buffer "
                "allocated outside expandable segments, or turn them off."
            )
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, tuple]] = []
        weight_refs: list[torch.Tensor] = []

        for name, tensor in source.iter_full():
            # FSDP shards were gathered by the source; ensure contiguity here.
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            weight = tensor.detach().contiguous()
            weight_refs.append(weight)
            _, ipc_args = reduce_tensor(weight)
            ipc_handles.append({self.gpu_uuid: ipc_args})

        ipc_handles = self._all_gather_and_merge_handles(ipc_handles)
        self._do_send(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            ipc_handles=ipc_handles,
        )
        return weight_refs

    def _send_packed(self, source: WeightSource) -> None:
        """Send weights in chunks of one IPC buffer, between barriers so no rank
        refills its buffer before the workers read it."""
        for chunk in packed_ipc_producer(
            iterator=source.iter_full(),
            gpu_uuid=self.gpu_uuid,
            device_index=self.device_index,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=self.packed_buffer_size_bytes,
        ):
            if chunk.ipc_handle is not None:
                handle = self._all_gather_and_merge_handles([chunk.ipc_handle])[0]
            else:
                self._barrier()
                handle = {}
            self._do_send(
                names=chunk.names,
                dtype_names=chunk.dtype_names,
                shapes=chunk.shapes,
                ipc_handles=handle,
                chunk_capacity=chunk.chunk_capacity,
                last=chunk.last,
            )
            self._barrier()

    def _do_send(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        chunk_capacity: int | None = None,
        last: bool = False,
    ) -> None:
        """Build one update payload and ship it via the client. Only the sender
        ships (non-sender ranks already contributed to the handle all-gather).

        Emits raw `ipc_handles`; transports that cannot carry them natively
        (HTTP/JSON) pickle them in their client (see `HTTPVLLMWeightSyncClient`).
        """
        if not self.is_sender:
            return
        update_info = IPCWeightTransferUpdateInfo(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            ipc_handles=ipc_handles,
            chunk_capacity=chunk_capacity,
            last=last,
        )
        self.client.update_weights(asdict(update_info))
