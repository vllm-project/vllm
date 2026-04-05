# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

import asyncio
import pickle
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import pybase64 as base64
import ray
import requests
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm import envs
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    packed_ipc_consumer,
    packed_ipc_producer,
)


@dataclass
class IPCTrainerSendWeightsArgs:
    """Arguments for IPC trainer_send_weights method."""

    send_mode: (
        str
        | Callable[["IPCWeightTransferUpdateInfo"], None | Coroutine[Any, Any, None]]
    )
    """How to send updates to vLLM. Either a string ('ray' or 'http') for
    built-in transports, or a callable (sync or async) that receives an
    IPCWeightTransferUpdateInfo and performs the send. Use
    async_trainer_send_weights when the callable is async."""
    llm_handle: Any = None
    """Ray actor handle or list of handles (required for 'ray' send_mode)."""
    url: str | None = None
    """Base URL for HTTP endpoint (required for 'http' send_mode)."""
    packed: bool = False
    """Whether to use packed tensor transfer for bounded-memory chunking."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer when packed=True."""

    def __post_init__(self):
        """Validate that required arguments are provided for the selected mode."""
        if callable(self.send_mode):
            return
        if self.send_mode == "ray" and self.llm_handle is None:
            raise ValueError("llm_handle is required for 'ray' send_mode")
        if self.send_mode == "http" and self.url is None:
            raise ValueError("url is required for 'http' send_mode")
        if self.send_mode not in ("ray", "http"):
            raise ValueError(
                f"send_mode must be 'ray', 'http', or a callable, "
                f"got {self.send_mode!r}"
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
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple]
    """IPC handles mapping physical GPU UUID to rebuild_cuda_tensor args.
    For non-packed mode: list of per-parameter handle dicts.
    For packed mode: single handle dict for the packed buffer."""
    tensor_sizes: list[int] | None = None
    """Per-parameter sizes in bytes within the packed buffer.
    Required when packed=True, unused otherwise."""
    packed: bool = False
    """Whether this update uses packed tensor format."""

    def __post_init__(self):
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
        if (
            not self.packed
            and isinstance(self.ipc_handles, list)
            and len(self.ipc_handles) != num_params
        ):
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )
        if self.packed and self.tensor_sizes is None:
            raise ValueError("`tensor_sizes` is required when packed=True")


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
        if "ipc_handles_pickled" in update_dict:
            if "ipc_handles" in update_dict:
                raise ValueError(
                    "Cannot specify both `ipc_handles` and `ipc_handles_pickled`"
                )

            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize `ipc_handles_pickled` without "
                    "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )

            pickled = update_dict.pop("ipc_handles_pickled")
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
                        GPU UUID and the rebuild_cuda_tensor args tuple.
            load_weights: Callable that loads weights into the model. Called
                         incrementally for each weight to avoid OOM.
        """
        device_index = torch.accelerator.current_device_index()

        if update_info.packed:
            assert update_info.tensor_sizes is not None
            assert isinstance(update_info.ipc_handles, dict)
            weights = packed_ipc_consumer(
                ipc_handle=update_info.ipc_handles,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                tensor_sizes=update_info.tensor_sizes,
                device_index=device_index,
            )
            load_weights(weights)
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
                list_args[6] = device_index
                weight = rebuild_cuda_tensor(*list_args)
                weights.append((name, weight))

            load_weights(weights)

    def shutdown(self) -> None:
        pass

    @staticmethod
    def _trainer_send_weights_impl(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | IPCTrainerSendWeightsArgs,
    ) -> Iterator[Coroutine[Any, Any, None]]:
        """Generator that yields coroutines when send_mode is async.

        Single implementation for both sync and async entry points.
        Yields nothing when send_mode is sync (ray/http or sync callable).
        """
        args = (
            IPCTrainerSendWeightsArgs(**trainer_args)
            if isinstance(trainer_args, dict)
            else trainer_args
        )
        device_index = torch.accelerator.current_device_index()
        gpu_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        send_method = (
            IPCWeightTransferEngine._send_packed
            if args.packed
            else IPCWeightTransferEngine._send_unpacked
        )
        yield from send_method(iterator, args, gpu_uuid)

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | IPCTrainerSendWeightsArgs,
    ) -> None:
        """Send weights from trainer to inference workers via CUDA IPC.

        Supports two transport modes ('ray' and 'http') and two transfer
        strategies:
        - Non-packed (default): all weights in a single API call.
        - Packed (packed=True): chunked transfer with bounded GPU memory.

        For multi-GPU training, all ranks must call this method in
        parallel. IPC handles are all-gathered across ranks and merged
        so that each vLLM worker can find its own GPU UUID. Only rank 0
        sends the payload to vLLM.

        Use ``async_trainer_send_weights`` when ``send_mode`` is an
        async callable.

        Args:
            iterator: Iterator of (name, tensor) pairs. For multi-GPU,
                     each rank should yield the full tensor on its own GPU
                     (e.g. via FSDP full_tensor()).
            trainer_args: IPCTrainerSendWeightsArgs or equivalent dict.
        """
        gen = IPCWeightTransferEngine._trainer_send_weights_impl(iterator, trainer_args)
        if next(gen, None) is not None:
            raise ValueError(
                "Async send_mode requires async_trainer_send_weights; "
                "use IPCWeightTransferEngine.async_trainer_send_weights instead"
            )

    @staticmethod
    async def async_trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | IPCTrainerSendWeightsArgs,
    ) -> None:
        """Async variant of trainer_send_weights for async send_mode callables.

        Use this when ``send_mode`` is an async callable. For sync
        send_mode (ray/http or sync callable), either this or
        ``trainer_send_weights`` works; this one awaits each send.

        For multi-GPU training, **all ranks** must call this method in
        parallel.

        Args:
            iterator: Iterator of (name, tensor) pairs. For multi-GPU,
                     each rank should yield the full tensor on its own GPU
                     (e.g. via FSDP full_tensor()).
            trainer_args: IPCTrainerSendWeightsArgs or equivalent dict.
        """
        for coro in IPCWeightTransferEngine._trainer_send_weights_impl(
            iterator, trainer_args
        ):
            await coro

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

    @staticmethod
    def _send_unpacked(
        iterator: Iterator[tuple[str, torch.Tensor]],
        args: IPCTrainerSendWeightsArgs,
        gpu_uuid: str,
    ) -> Iterator[Coroutine[Any, Any, None]]:
        """Send all weights in a single API call (non-packed mode)."""
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, tuple]] = []

        for name, tensor in iterator:
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            weight = tensor.detach().contiguous()
            _, ipc_args = reduce_tensor(weight)
            ipc_handles.append({gpu_uuid: ipc_args})

        ipc_handles = IPCWeightTransferEngine._all_gather_and_merge_handles(ipc_handles)

        if IPCWeightTransferEngine._is_rank_zero():
            maybe_coro = IPCWeightTransferEngine._do_send(
                args=args,
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                ipc_handles=ipc_handles,
            )
            if maybe_coro is not None:
                yield maybe_coro

        IPCWeightTransferEngine._post_send_sync()

    @staticmethod
    def _send_packed(
        iterator: Iterator[tuple[str, torch.Tensor]],
        args: IPCTrainerSendWeightsArgs,
        gpu_uuid: str,
    ) -> Iterator[Coroutine[Any, Any, None]]:
        """Send weights in bounded-memory chunks (packed mode)."""
        post_iter_func: Callable = lambda item: item[1]

        for chunk in packed_ipc_producer(
            iterator=iterator,
            gpu_uuid=gpu_uuid,
            post_iter_func=post_iter_func,
            buffer_size_bytes=args.packed_buffer_size_bytes,
        ):
            ipc_handle = IPCWeightTransferEngine._all_gather_and_merge_handles(
                [chunk.ipc_handle]
            )[0]

            if IPCWeightTransferEngine._is_rank_zero():
                maybe_coro = IPCWeightTransferEngine._do_send(
                    args=args,
                    names=chunk.names,
                    dtype_names=chunk.dtype_names,
                    shapes=chunk.shapes,
                    ipc_handles=ipc_handle,
                    tensor_sizes=chunk.tensor_sizes,
                    packed=True,
                    first_chunk=chunk.is_first,
                    last_chunk=chunk.is_last,
                )
                if maybe_coro is not None:
                    yield maybe_coro

            IPCWeightTransferEngine._post_send_sync()

    @staticmethod
    def _do_send(
        args: IPCTrainerSendWeightsArgs,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        tensor_sizes: list[int] | None = None,
        packed: bool = False,
        first_chunk: bool = True,
        last_chunk: bool = True,
    ) -> Coroutine[Any, Any, None] | None:
        """Send a single update payload via the configured transport.

        Returns a coroutine when send_mode is an async callable; otherwise None.
        """
        update_fields: dict[str, Any] = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": packed,
            "first_chunk": first_chunk,
            "last_chunk": last_chunk,
        }
        if tensor_sizes is not None:
            update_fields["tensor_sizes"] = tensor_sizes

        update_fields["ipc_handles"] = ipc_handles
        update_info = IPCWeightTransferUpdateInfo(**update_fields)

        if callable(args.send_mode):
            if asyncio.iscoroutinefunction(args.send_mode):
                return args.send_mode(update_info)
            args.send_mode(update_info)
        elif args.send_mode == "ray":
            handles = (
                args.llm_handle
                if isinstance(args.llm_handle, list)
                else [args.llm_handle]
            )
            ray.get(
                [
                    h.update_weights.remote(dict(update_info=asdict(update_info)))
                    for h in handles
                ]
            )
        elif args.send_mode == "http":
            pickled_handles = base64.b64encode(pickle.dumps(ipc_handles)).decode(
                "utf-8"
            )
            http_fields = {k: v for k, v in update_fields.items() if k != "ipc_handles"}
            http_fields["ipc_handles_pickled"] = pickled_handles

            url = f"{args.url}/update_weights"
            payload = {"update_info": http_fields}
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
        return None
