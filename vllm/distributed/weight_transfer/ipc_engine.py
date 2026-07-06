# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

import pickle
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import pybase64 as base64
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
from typing_extensions import Self

from vllm import envs
from vllm.config.weight_transfer import IPCWeightTransferConfig
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
    packed_ipc_consumer,
    packed_ipc_producer,
)


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for IPC weight transfer. No rendezvous needed."""

    pass


@dataclass
class IPCTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for IPC weight transfer. No rendezvous needed;
    `rank` (from `TrainerInitInfo`) identifies this trainer process — rank 0
    ships the merged IPC handles. All ranks still join the handle all-gather."""


@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the IPC weight transfer backend.

    Whether the transfer is packed lives on `IPCWeightTransferConfig`; this
    carries only per-round metadata and the IPC handles.
    """

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple] | None = None
    """IPC handles mapping physical GPU UUID to rebuild_cuda_tensor args.
    For non-packed mode: list of per-parameter handle dicts.
    For packed mode: single handle dict for the packed buffer."""
    ipc_handles_pickled: str | None = None
    """Base64-encoded pickled IPC handles, used for HTTP transport."""
    tensor_sizes: list[int] | None = None
    """Per-parameter sizes in bytes within the packed buffer.
    Required when packed=True, unused otherwise."""

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
    """
    Weight transfer engine using CUDA IPC for communication between trainer and workers.

    This implementation uses CUDA IPC to transfer weights from the trainer (rank 0)
    to all inference workers in a process group. IPC handles are used to share
    memory between processes on the same node.
    """

    # Define backend-specific dataclass types
    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo

    # Narrow the base `config` type: the IPC engine reads the `packed` wire
    # param that lives on the subclass.
    config: IPCWeightTransferConfig

    def __init__(
        self,
        config: IPCWeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)

    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
        """
        Initialize the weight transfer mechanism.
        This is called once at the beginning of training.
        No initialization needed for IPC backend.

        Args:
            init_info: IPC initialization info (empty)
        """
        pass

    def start_weight_update(self) -> None:
        """Initialize layerwise reloading for the incoming checkpoint weights."""
        from vllm.model_executor.model_loader.reload import (
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        """Finalize layerwise reloading after all weights have been received."""
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
        )

        finalize_layerwise_reload(self.model, self.model_config)

    def receive_weights(self, update_info: IPCWeightTransferUpdateInfo) -> None:
        """
        Receive weights from the trainer via CUDA IPC handles and load them.

        Whether the transfer is packed is read from `self.config` so it is
        guaranteed to match the trainer's config.

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

        if self.config.packed:
            if update_info.tensor_sizes is None:
                raise ValueError("`tensor_sizes` is required when packed=True")
            assert isinstance(update_info.ipc_handles, dict)
            weights = packed_ipc_consumer(
                ipc_handle=update_info.ipc_handles,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                tensor_sizes=update_info.tensor_sizes,
                device_index=device_index,
            )
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
        pass


class IPCTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[IPCWeightTransferConfig, IPCTrainerInitInfo]
):
    """Trainer-side CUDA IPC weight transfer engine.

    Called on every trainer rank. For multi-rank (e.g. FSDP) trainers all ranks
    iterate the source (materializing each tensor) and contribute to the
    IPC-handle all-gather; only the sender (rank 0) ships the merged handles to
    the inference side. IPC transfer
    is straight-line (no concurrent broadcast like NCCL): `update_weights` *is*
    the transfer, and it rides the client, so it no-ops on non-senders.
    """

    init_info_cls = IPCTrainerInitInfo
    config_cls = IPCWeightTransferConfig

    def __init__(
        self,
        config: IPCWeightTransferConfig,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
    ) -> None:
        super().__init__(config, client=client, source=source, is_sender=is_sender)
        self.device_index = torch.accelerator.current_device_index()
        self.gpu_uuid = str(torch.cuda.get_device_properties(self.device_index).uuid)

    @classmethod
    def trainer_init(
        cls,
        config: IPCWeightTransferConfig,
        init_info: IPCTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
    ) -> Self:
        engine = cls(
            config, client=client, source=source, is_sender=init_info.is_sender
        )
        # IPC needs no data-plane rendezvous; this just lets the worker construct
        # its (empty) init info. Only the sender drives the inference side.
        if engine.is_sender:
            engine.client.init_weight_transfer_engine({})
        return engine

    def send_weights(self) -> None:
        source = self.source
        if self.is_sender:
            self.client.start_weight_update()
        self._send(source)
        if self.is_sender:
            self.client.finish_weight_update()
        self._post_send_sync()

    # ---- data plane (runs on all ranks; only the sender ships) ----

    def _send(self, source: WeightSource) -> None:
        if self.config.packed:
            self._send_packed(source)
        else:
            self._send_unpacked(source)

    @staticmethod
    def _all_gather_and_merge_handles(
        handles: list[dict[str, tuple]],
    ) -> list[dict[str, tuple]]:
        """All-gather and merge IPC handle dicts across ranks in one call.

        Each rank contributes a list of {gpu_uuid: ipc_args} dicts (one
        per parameter or one per chunk). A single all_gather_object
        collects every rank's full list, then rank 0 merges per-index so
        each dict maps every GPU UUID to its args.

        Non-rank-0 returns a list of empty dicts.
        No-op (returns handles unchanged) when no distributed group exists.
        """
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return handles

        world_size = torch.distributed.get_world_size()
        gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, handles)
        torch.distributed.barrier()
        torch.cuda.synchronize()

        if torch.distributed.get_rank() == 0:
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
    def _post_send_sync() -> None:
        """Barrier + ipc_collect after a send; no-op if single-GPU."""
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        torch.cuda.ipc_collect()

    def _send_unpacked(self, source: WeightSource) -> None:
        """Iterate the source, build one IPC handle per param, all-gather the
        handles across ranks, and (sender) ship them in one update call."""
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, tuple]] = []
        # Hold strong refs to every contiguous copy until the send + post-send
        # sync completes. reduce_tensor's returned args do NOT keep storage
        # alive, and non-contiguous inputs allocate fresh storage in
        # .contiguous() that would otherwise be GC'd before the consumer opens
        # the IPC handle.
        weight_refs: list[torch.Tensor] = []

        for name, tensor in source:
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

    def _send_packed(self, source: WeightSource) -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        for chunk in packed_ipc_producer(
            iterator=iter(source),
            gpu_uuid=self.gpu_uuid,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=self.config.packed_buffer_size_bytes,
        ):
            ipc_handle = self._all_gather_and_merge_handles([chunk.ipc_handle])[0]
            self._do_send(
                names=chunk.names,
                dtype_names=chunk.dtype_names,
                shapes=chunk.shapes,
                ipc_handles=ipc_handle,
                tensor_sizes=chunk.tensor_sizes,
            )
            # Per-chunk barrier: the producer reuses a single IPC buffer across
            # chunks, but only the sender waits for the consumers (via _do_send).
            # Without syncing every rank here, non-sender ranks race ahead and
            # overwrite their buffer while their colocated worker is still
            # reading the current chunk, silently corrupting the transfer.
            self._post_send_sync()

    def _do_send(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        tensor_sizes: list[int] | None = None,
    ) -> None:
        """Build one update payload and ship it via the client. Only the sender
        ships (non-sender ranks already contributed to the handle all-gather).

        Emits raw `ipc_handles`; transports that cannot carry them natively
        (HTTP/JSON) pickle them in their client (see `HTTPVLLMWeightSyncClient`).
        """
        if not self.is_sender:
            return
        update_fields: dict[str, Any] = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles": ipc_handles,
        }
        if tensor_sizes is not None:
            update_fields["tensor_sizes"] = tensor_sizes

        update_info = IPCWeightTransferUpdateInfo(**update_fields)
        self.client.update_weights(asdict(update_info))
