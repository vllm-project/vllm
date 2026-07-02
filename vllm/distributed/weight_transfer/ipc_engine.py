# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

import pickle
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import pybase64 as base64
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
from typing_extensions import Self

from vllm import envs
from vllm.config.weight_transfer import IPCWeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightIterator,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
    materialize_full_tensor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.packed_tensor import (
    packed_ipc_consumer,
    packed_ipc_producer,
)

# A callback that ships one update payload (an update_info dict) to the
# inference side. The trainer engine passes `client.update_weights`; non-rank-0
# trainer ranks pass `None` (they participate in the IPC handle all-gather but
# never send).
SendFn = Callable[[dict[str, Any]], None]


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for IPC weight transfer. No rendezvous needed."""

    pass


@dataclass
class IPCTrainerInitInfo(WeightTransferInitInfo):
    """Trainer-side init info for IPC weight transfer. No rendezvous needed."""

    pass


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

    def parse_update_info(
        self, update_dict: dict[str, Any]
    ) -> IPCWeightTransferUpdateInfo:
        """Parse update dict, deserializing pickled IPC handles if present.

        HTTP transport sends IPC handles as a base64-encoded pickle under the
        key ``ipc_handles_pickled``. This method deserializes them back into
        ``ipc_handles`` before constructing the typed dataclass, keeping
        serialization concerns out of the dataclass itself.

        Requires ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` because the
        payload is deserialized via ``pickle.loads``.
        """
        pickled = update_dict.pop("ipc_handles_pickled", None)
        if pickled is not None:
            if update_dict.get("ipc_handles") is not None:
                raise ValueError(
                    "Cannot specify both `ipc_handles` and `ipc_handles_pickled`"
                )

            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize `ipc_handles_pickled` without "
                    "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )

            update_dict["ipc_handles"] = pickle.loads(base64.b64decode(pickled))

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

    For multi-rank (e.g. FSDP) trainers, *all* ranks must participate in the
    IPC handle all-gather, but only rank 0 holds the engine and drives the
    inference-side RPCs. Non-rank-0 ranks call the static `participate()`
    helper to join the all-gather without sending anything. IPC transfer is
    straight-line (no concurrent broadcast like NCCL): `update_weights` *is*
    the transfer.
    """

    init_info_cls = IPCTrainerInitInfo
    config_cls = IPCWeightTransferConfig

    def __init__(
        self,
        config: IPCWeightTransferConfig,
        *,
        client: VLLMWeightSyncClient,
        weight_iterator: WeightIterator | None = None,
    ) -> None:
        super().__init__(config, client=client, weight_iterator=weight_iterator)
        self.device_index = torch.accelerator.current_device_index()
        self.gpu_uuid = str(torch.cuda.get_device_properties(self.device_index).uuid)

    @classmethod
    def trainer_init(
        cls,
        config: IPCWeightTransferConfig,
        init_info: IPCTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        weight_iterator: WeightIterator | None = None,
    ) -> Self:
        engine = cls(config, client=client, weight_iterator=weight_iterator)
        # IPC needs no data-plane rendezvous; this just lets the worker
        # construct its (empty) init info.
        client.init_weight_transfer_engine({})
        return engine

    def send_weights(self, weight_iterator: WeightIterator | None = None) -> None:
        iterator = self._resolve_iterator(weight_iterator)()
        self.client.start_weight_update()
        self._send(iterator, self.config, self.gpu_uuid, self.client.update_weights)
        self.client.finish_weight_update()
        self._post_send_sync()

    @classmethod
    def participate(
        cls, weight_iterator: WeightIterator, config: IPCWeightTransferConfig
    ) -> None:
        """Join the IPC handle all-gather from a non-rank-0 trainer rank.

        Runs the same data-plane gather as `send_weights` (so the collective
        all-gather lines up across ranks) but sends nothing and drives no
        client RPCs.
        """
        device_index = torch.accelerator.current_device_index()
        gpu_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        cls._send(weight_iterator(), config, gpu_uuid, send_fn=None)
        cls._post_send_sync()

    # ---- data plane (shared by rank 0 send_weights + non-rank-0 participate) ----

    @classmethod
    def _send(
        cls,
        iterator: Iterator[tuple[str, torch.Tensor]],
        config: IPCWeightTransferConfig,
        gpu_uuid: str,
        send_fn: SendFn | None,
    ) -> None:
        if config.packed:
            cls._send_packed(iterator, config, gpu_uuid, send_fn)
        else:
            cls._send_unpacked(iterator, gpu_uuid, send_fn)

    @staticmethod
    def _is_rank_zero() -> bool:
        """Return True if this is rank 0 or no distributed group exists."""
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

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

    @classmethod
    def _send_unpacked(
        cls,
        iterator: Iterator[tuple[str, torch.Tensor]],
        gpu_uuid: str,
        send_fn: SendFn | None,
    ) -> None:
        """Send all weights in a single API call (non-packed mode)."""
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

        for name, tensor in iterator:
            # FSDP shards are gathered here (once); regular tensors pass through.
            full = materialize_full_tensor(tensor)
            names.append(name)
            dtype_names.append(str(full.dtype).split(".")[-1])
            shapes.append(list(full.shape))

            weight = full.detach().contiguous()
            weight_refs.append(weight)
            _, ipc_args = reduce_tensor(weight)
            ipc_handles.append({gpu_uuid: ipc_args})

        ipc_handles = cls._all_gather_and_merge_handles(ipc_handles)

        if cls._is_rank_zero() and send_fn is not None:
            cls._do_send(
                send_fn=send_fn,
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                ipc_handles=ipc_handles,
            )

    @classmethod
    def _send_packed(
        cls,
        iterator: Iterator[tuple[str, torch.Tensor]],
        config: IPCWeightTransferConfig,
        gpu_uuid: str,
        send_fn: SendFn | None,
    ) -> None:
        """Send weights in bounded-memory chunks (packed mode)."""

        def post_iter_func(item):
            return materialize_full_tensor(item[1])

        for chunk in packed_ipc_producer(
            iterator=iterator,
            gpu_uuid=gpu_uuid,
            post_iter_func=post_iter_func,
            buffer_size_bytes=config.packed_buffer_size_bytes,
        ):
            ipc_handle = cls._all_gather_and_merge_handles([chunk.ipc_handle])[0]

            if cls._is_rank_zero() and send_fn is not None:
                cls._do_send(
                    send_fn=send_fn,
                    names=chunk.names,
                    dtype_names=chunk.dtype_names,
                    shapes=chunk.shapes,
                    ipc_handles=ipc_handle,
                    tensor_sizes=chunk.tensor_sizes,
                )

    @staticmethod
    def _do_send(
        send_fn: SendFn,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        tensor_sizes: list[int] | None = None,
    ) -> None:
        """Build a single update payload and ship it via `send_fn`.

        Emits raw `ipc_handles`; transports that cannot carry them natively
        (HTTP/JSON) pickle them in their client (see
        `HTTPVLLMWeightSyncClient`).
        """
        update_fields: dict[str, Any] = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles": ipc_handles,
        }
        if tensor_sizes is not None:
            update_fields["tensor_sizes"] = tensor_sizes

        update_info = IPCWeightTransferUpdateInfo(**update_fields)
        send_fn(asdict(update_info))
