# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based (dense) weight transfer engine."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Self

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    stateless_init_metadata_group,
    worker_init_payload,
    worker_init_process_group,
)
from vllm.distributed.weight_transfer.nccl_common import (
    trainer_init as open_trainer_endpoint,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_nccl_broadcast_consumer,
    packed_nccl_broadcast_producer,
)

# NCCLWeightTransferInitInfo is re-exported here for convenience; its canonical
# home is nccl_common, shared with the sparse backend.
__all__ = [
    "NCCLWeightTransferInitInfo",
    "NCCLTrainerInitInfo",
    "NCCLWeightTransferUpdateInfo",
    "NCCLWeightTransferEngine",
    "NCCLTrainerWeightTransferEngine",
]


@dataclass
class NCCLTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the dense NCCL weight transfer backend.

    Source ranks occupy the communicator prefix and inference workers follow
    them. Rank 0 drives the control plane. Additional source ranks join the
    communicator and broadcast their own manifests without touching the client.
    Trainer processes that only participate in source-side gathers leave
    `source_rank` unset.

    The trainer joins over a TCPStore rendezvous (`master_address` +
    `master_port`). Torch-free trainers (e.g. JAX) that cannot join a TCPStore
    mint an `ncclUniqueId` themselves and drive rank 0 out of band; they ship a
    `nccl_unique_id_b64` payload straight to the inference workers'
    `init_weight_transfer_engine` (see `NCCLWeightTransferInitInfo`) rather than
    going through this engine.

    `packed` / buffer sizes are the transfer's wire params. The trainer
    propagates them to the worker at `trainer_init` so the two sides cannot
    disagree. Note this defaults to packed, unlike the worker-side
    `NCCLWeightTransferInitInfo`, whose default only applies when no trainer
    ships a value. `backend` is the factory dispatch key."""

    backend: ClassVar[str] = "nccl"

    master_address: str
    master_port: int
    world_size: int
    packed: bool = True
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    num_source_ranks: int = 1
    source_rank: int | None = None
    coordination_port: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_source_ranks, int)
            or isinstance(self.num_source_ranks, bool)
            or not 0 < self.num_source_ranks < self.world_size
        ):
            raise ValueError(
                "`num_source_ranks` must leave at least one inference worker"
            )
        if self.source_rank is not None and (
            not isinstance(self.source_rank, int)
            or isinstance(self.source_rank, bool)
            or not 0 <= self.source_rank < self.num_source_ranks
        ):
            raise ValueError(
                f"`source_rank` must be in [0, {self.num_source_ranks}) or None"
            )
        if self.transfer_rank == 0 and self.rank != 0:
            raise ValueError("trainer rank 0 must own source rank 0")
        if self.rank == 0 and self.transfer_rank != 0:
            raise ValueError("trainer rank 0 must be source rank 0")
        if self.num_source_ranks > 1 and (
            not isinstance(self.coordination_port, int)
            or isinstance(self.coordination_port, bool)
            or not 0 < self.coordination_port < 65536
        ):
            raise ValueError(
                "`coordination_port` must be set for multiple source ranks"
            )

    @property
    def transfer_rank(self) -> int | None:
        if self.source_rank is not None:
            return self.source_rank
        return 0 if self.rank == 0 else None


@dataclass
class NCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the dense NCCL weight transfer backend.

    Whether the transfer is packed (and the buffer geometry) is a must-agree
    wire param carried on the init info (`NCCLTrainerInitInfo` /
    `NCCLWeightTransferInitInfo`), not here; this carries the per-round
    parameter metadata and source rank.

    Leave all three lists empty to reuse the metadata of the previous
    `update_weights` call on the same model from the same `src`, so a trainer
    that sends the same parameters every round ships the names, dtypes and
    shapes only once.
    """

    names: list[str] = field(default_factory=list)
    dtype_names: list[str] = field(default_factory=list)
    shapes: list[list[int]] = field(default_factory=list)
    src: int = 0

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
    """Weight transfer engine using NCCL for communication between trainer and workers.

    This implementation uses NCCL broadcast operations to transfer dense
    checkpoint-format weights from a selected trainer rank to all inference
    workers in a process group. Received weights are loaded via the model's
    `load_weights` using the layerwise reload lifecycle.
    """

    # Define backend-specific dataclass types
    init_info_cls = NCCLWeightTransferInitInfo
    update_info_cls = NCCLWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyNcclCommunicator | None = None
        self.rank_offset: int | None = None
        # Set from the trainer-supplied init info at the handshake; defaults are
        # only for the (unreachable) receive-before-init case.
        self.packed = False
        self.packed_buffer_size_bytes = DEFAULT_PACKED_BUFFER_SIZE_BYTES
        self.packed_num_buffers = DEFAULT_PACKED_NUM_BUFFERS
        # Metadata of the last update per (model, src), reused by empty update info.
        self._last_update_info: dict[
            tuple[torch.nn.Module, int], NCCLWeightTransferUpdateInfo
        ] = {}

    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """Initialize NCCL process group with the trainer and record the
        trainer-supplied wire params so the worker decodes exactly as the
        trainer encodes.

        Args:
            init_info: NCCL initialization info containing master address, port,
                      rank offset, world size, and the packed wire params

        """
        self.rank_offset = init_info.rank_offset
        self.packed = init_info.packed
        self.packed_buffer_size_bytes = init_info.packed_buffer_size_bytes
        self.packed_num_buffers = init_info.packed_num_buffers
        self._last_update_info.clear()
        self.model_update_group = worker_init_process_group(
            init_info, self.parallel_config
        )

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

    def receive_weights(self, update_info: NCCLWeightTransferUpdateInfo) -> None:
        """Receive weights from trainer via NCCL broadcast.

        Whether to use packed broadcasting (and the buffer geometry) is read
        from `self.packed` / `self.packed_*`, set at the init handshake from the
        trainer's init info, so it is guaranteed to match how the trainer
        encoded.

        Args:
            update_info: NCCL update info containing parameter names, dtypes,
                        shapes, and the source rank; empty lists to reuse
                        the previous call's from the same source

        """
        if self.model_update_group is None or self.rank_offset is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )
        if (
            not isinstance(update_info.src, int)
            or isinstance(update_info.src, bool)
            or not 0 <= update_info.src < self.rank_offset
        ):
            raise ValueError(
                f"NCCL source rank must be an integer in [0, {self.rank_offset}), "
                f"got {update_info.src!r}."
            )
        key = (self.model, update_info.src)
        if update_info.names:
            self._last_update_info[key] = update_info
        else:
            update_info = self._last_update_info.get(key, update_info)

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        with disable_mtp_completeness_check():
            if self.packed:
                # Build iterator of (name, (shape, dtype)) from update_info
                def state_dict_info_iterator():
                    for name, dtype_name, shape in zip(
                        update_info.names, update_info.dtype_names, update_info.shapes
                    ):
                        dtype = getattr(torch, dtype_name)
                        yield (name, (shape, dtype))

                packed_nccl_broadcast_consumer(
                    iterator=state_dict_info_iterator(),
                    group=self.model_update_group,
                    src=update_info.src,
                    post_unpack_func=self.model.load_weights,
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                    device=self.device,
                )
            else:
                # Use simple one-by-one broadcasting
                for name, dtype_name, shape in zip(
                    update_info.names, update_info.dtype_names, update_info.shapes
                ):
                    dtype = getattr(torch, dtype_name)
                    weight = torch.empty(shape, dtype=dtype, device=self.device)
                    self.model_update_group.broadcast(
                        weight,
                        src=update_info.src,
                        stream=torch.cuda.current_stream(),
                    )
                    self.model.load_weights([(name, weight)])
                    del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.destroy()
            self.model_update_group = None


class NCCLTrainerWeightTransferEngine(TrainerWeightTransferEngine[NCCLTrainerInitInfo]):
    """Trainer-side NCCL weight transfer engine.

    Every source root joins the shared communicator and participates in each
    broadcast. Rank 0 drives the inference control plane, while each source
    supplies the parameters selected by its `held_names()` ownership. Other
    trainer ranks only iterate their source to participate in source-side
    collectives.

    Packing parameters come from `NCCLTrainerInitInfo` and are propagated to
    workers during initialization.
    """

    init_info_cls = NCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        source_rank: int | None = None,
        num_source_ranks: int = 1,
        packed: bool = True,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.source_rank = 0 if is_sender and source_rank is None else source_rank
        self.num_source_ranks = num_source_ranks
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.packed_num_buffers = packed_num_buffers
        self.model_update_group: PyNcclCommunicator | None = None
        self.update_infos: list[NCCLWeightTransferUpdateInfo] = []
        self.source_metadata_group: StatelessProcessGroup | None = None
        self._group_lock = Lock()
        # Metadata the worker already holds; an unchanged round sends none.
        self._sent_update_infos: list[NCCLWeightTransferUpdateInfo] | None = None

    @classmethod
    def trainer_init(
        cls,
        init_info: NCCLTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("NCCL trainer weight transfer requires a WeightSource.")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            source_rank=init_info.transfer_rank,
            num_source_ranks=init_info.num_source_ranks,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )
        local_meta = None
        local_held_names = None
        local_metadata = None
        local_metadata_error = None
        try:
            local_meta = source.metadata()
            names = [item.name for item in local_meta]
            local_metadata = {
                "names": names,
                "dtype_names": [str(item.dtype).split(".")[-1] for item in local_meta],
                "shapes": [list(item.shape) for item in local_meta],
            }
            held = source.held_names()
            if held is None:
                local_held_names = names
            else:
                held_set = set(held)
                if not all(isinstance(name, str) for name in held_set):
                    raise TypeError("WeightSource.held_names() must contain strings")
                unknown = sorted(held_set - set(names))
                if unknown:
                    raise ValueError(
                        "WeightSource.held_names() contains names not present in "
                        f"metadata(), e.g. {unknown[:3]}"
                    )
                local_held_names = [name for name in names if name in held_set]
                if not local_held_names:
                    raise ValueError("WeightSource.held_names() must not be empty")
        except Exception as exc:
            local_metadata_error = f"{type(exc).__name__}: {exc}"

        source_rank = engine.source_rank
        if source_rank is None:
            if local_metadata_error is not None:
                raise RuntimeError(
                    "NCCL source metadata validation failed: " + local_metadata_error
                )
            return engine

        if init_info.num_source_ranks > 1:
            assert init_info.coordination_port is not None
            engine.source_metadata_group = stateless_init_metadata_group(
                init_info.master_address,
                init_info.coordination_port,
                source_rank,
                init_info.num_source_ranks,
            )
            envelopes = engine.source_metadata_group.all_gather_obj(
                {
                    "source_rank": source_rank,
                    "error": local_metadata_error,
                    "metadata": local_metadata,
                    "held_names": local_held_names,
                }
            )
        else:
            envelopes = [
                {
                    "source_rank": source_rank,
                    "error": local_metadata_error,
                    "metadata": local_metadata,
                    "held_names": local_held_names,
                }
            ]
        errors = [
            f"source rank {envelope['source_rank']}: {envelope['error']}"
            for envelope in envelopes
            if envelope["error"] is not None
        ]
        if errors:
            raise RuntimeError(
                "NCCL source metadata validation failed: " + "; ".join(errors)
            )
        engine.update_infos = engine._build_update_infos(
            envelopes, init_info.num_source_ranks
        )

        worker_init_info = NCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=init_info.num_source_ranks,
            world_size=init_info.world_size,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )

        executor = None
        future = None
        model_update_group = None
        local_init_error = None
        try:
            if engine.is_sender:
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    engine.client.init_weight_transfer_engine,
                    worker_init_payload(worker_init_info),
                )
            model_update_group = open_trainer_endpoint(init_info, rank=source_rank)
            if future is not None:
                future.result()
        except BaseException as exc:
            local_init_error = f"{type(exc).__name__}: {exc}"
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

        init_errors = []
        if engine.source_metadata_group is not None:
            outcomes = engine.source_metadata_group.all_gather_obj(
                {
                    "source_rank": engine.source_rank,
                    "error": local_init_error,
                }
            )
            init_errors = [
                f"source rank {outcome['source_rank']}: {outcome['error']}"
                for outcome in outcomes
                if outcome["error"] is not None
            ]
        elif local_init_error is not None:
            init_errors = [local_init_error]
        if init_errors:
            if model_update_group is not None:
                model_update_group.destroy()
            raise RuntimeError(
                "NCCL trainer initialization failed: " + "; ".join(init_errors)
            )

        engine.model_update_group = model_update_group
        return engine

    def send_weights(self) -> None:
        assert self.source is not None  # guaranteed by trainer_init / __init__
        source = self.source

        if self.source_rank is None:
            for _ in self._iter_owned(source):
                pass
            self._post_send_sync()
            return

        if self.model_update_group is None:
            raise RuntimeError("trainer_init() must be called before send_weights().")
        if self.source_metadata_group is None:
            meta = source.metadata()
            self.update_infos = [
                NCCLWeightTransferUpdateInfo(
                    names=[item.name for item in meta],
                    dtype_names=[str(item.dtype).split(".")[-1] for item in meta],
                    shapes=[list(item.shape) for item in meta],
                    src=self.source_rank,
                )
            ]

        start_failure = None
        if self.is_sender:
            try:
                self.client.start_weight_update()
            except BaseException as exc:
                start_failure = exc
        self._exchange_source_outcome("start", -1, start_failure)
        if start_failure is not None:
            raise start_failure

        reuse_metadata = self.update_infos == self._sent_update_infos
        for update_index, update_info in enumerate(self.update_infos):
            self._send_update(update_index, update_info, reuse_metadata)

        finish_failure = None
        if self.is_sender:
            try:
                self.client.finish_weight_update()
            except BaseException as exc:
                finish_failure = exc
        self._exchange_source_outcome("finish", -1, finish_failure)
        if finish_failure is not None:
            raise finish_failure
        self._sent_update_infos = self.update_infos

    def _send_update(
        self,
        update_index: int,
        update_info: NCCLWeightTransferUpdateInfo,
        reuse_metadata: bool = False,
    ) -> None:
        executor = None
        future = None
        failure = None
        if self.is_sender:
            wire_info = (
                NCCLWeightTransferUpdateInfo(src=update_info.src)
                if reuse_metadata
                else update_info
            )
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self.client.update_weights, asdict(wire_info))
                future.add_done_callback(self._abort_on_rpc_failure)
                if future.done():
                    future.result()
            except BaseException as exc:
                failure = exc

        try:
            if failure is None:
                try:
                    self._broadcast_update(update_info)
                except BaseException as exc:
                    failure = exc
            if failure is None and future is not None:
                try:
                    future.result()
                except BaseException as exc:
                    failure = exc
            if failure is not None:
                # Peers may be blocked in this update's broadcasts and never
                # reach the outcome exchange, so raise instead of waiting on it.
                self._abort_model_update_group()
                raise failure
            self._exchange_source_outcome("update", update_index, None)
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def _abort_on_rpc_failure(self, future) -> None:
        try:
            future.result()
        except BaseException:
            self._abort_model_update_group()

    def _exchange_source_outcome(
        self, phase: str, update_index: int, failure: BaseException | None
    ) -> None:
        if self.source_metadata_group is None:
            return
        outcomes = self.source_metadata_group.all_gather_obj(
            {
                "phase": phase,
                "source_rank": self.source_rank,
                "update_index": update_index,
                "error": (
                    None if failure is None else f"{type(failure).__name__}: {failure}"
                ),
            }
        )
        expected = (phase, update_index)
        if [outcome.get("source_rank") for outcome in outcomes] != list(
            range(self.num_source_ranks)
        ) or any(
            (outcome.get("phase"), outcome.get("update_index")) != expected
            for outcome in outcomes
        ):
            self._abort_model_update_group()
            raise RuntimeError("NCCL source ranks reached different update phases")
        errors = [
            f"source rank {outcome['source_rank']}: {outcome['error']}"
            for outcome in outcomes
            if outcome["error"] is not None
        ]
        if errors:
            self._abort_model_update_group()
        if errors and failure is None:
            raise RuntimeError("NCCL weight transfer failed: " + "; ".join(errors))

    def _broadcast_update(self, update_info: NCCLWeightTransferUpdateInfo) -> None:
        group = self.model_update_group
        if group is None:
            raise RuntimeError("NCCL model update group is unavailable")
        meta = [
            ParamMeta(name, getattr(torch, dtype_name), tuple(shape))
            for name, dtype_name, shape in zip(
                update_info.names,
                update_info.dtype_names,
                update_info.shapes,
            )
        ]
        try:
            if self.packed:
                if self.source_rank == update_info.src:
                    assert self.source is not None
                    packed_nccl_broadcast_producer(
                        iterator=self._checked_iter(
                            self._iter_owned(self.source), meta
                        ),
                        group=group,
                        src=update_info.src,
                        post_iter_func=lambda item: item[1],
                        buffer_size_bytes=self.packed_buffer_size_bytes,
                        num_buffers=self.packed_num_buffers,
                    )
                else:
                    packed_nccl_broadcast_consumer(
                        iterator=(
                            (item.name, (list(item.shape), item.dtype)) for item in meta
                        ),
                        group=group,
                        src=update_info.src,
                        post_unpack_func=lambda weights: None,
                        buffer_size_bytes=self.packed_buffer_size_bytes,
                        num_buffers=self.packed_num_buffers,
                        device=group.device,
                    )
            else:
                stream = torch.cuda.current_stream()
                if self.source_rank == update_info.src:
                    assert self.source is not None
                    tensors = self._checked_iter(self._iter_owned(self.source), meta)
                else:
                    tensors = (
                        (
                            item.name,
                            torch.empty(
                                item.shape,
                                dtype=item.dtype,
                                device=group.device,
                            ),
                        )
                        for item in meta
                    )
                for _name, tensor in tensors:
                    send = tensor if tensor.is_contiguous() else tensor.contiguous()
                    group.broadcast(send, src=update_info.src, stream=stream)
            self._post_send_sync()
        except BaseException:
            self._abort_model_update_group()
            raise

    def _abort_model_update_group(self) -> None:
        lock = getattr(self, "_group_lock", None)
        if lock is None:
            group = self.model_update_group
            self.model_update_group = None
        else:
            with lock:
                group = self.model_update_group
                self.model_update_group = None
        if group is not None:
            group.destroy()

    @staticmethod
    def _build_update_infos(
        envelopes: list[dict], num_source_ranks: int
    ) -> list[NCCLWeightTransferUpdateInfo]:
        if [envelope["source_rank"] for envelope in envelopes] != list(
            range(num_source_ranks)
        ):
            raise ValueError("source manifests must be ordered by source rank")

        metadata = envelopes[0]["metadata"]
        if any(envelope["metadata"] != metadata for envelope in envelopes[1:]):
            raise ValueError("WeightSource metadata() differs across source ranks")
        if not isinstance(metadata, dict):
            raise TypeError("source metadata must be a dictionary")
        full_info = NCCLWeightTransferUpdateInfo(src=0, **metadata)
        if len(set(full_info.names)) != len(full_info.names):
            raise ValueError("WeightSource metadata() contains duplicate names")

        owners: dict[str, int] = {}
        known_names = set(full_info.names)
        for envelope in envelopes:
            source_rank = envelope["source_rank"]
            held_names = envelope["held_names"]
            if not isinstance(held_names, list) or not held_names:
                raise ValueError(
                    f"source rank {source_rank} must hold at least one parameter"
                )
            for name in held_names:
                if name not in known_names:
                    raise ValueError(
                        f"source rank {source_rank} holds unknown parameter {name!r}"
                    )
                if name in owners:
                    raise ValueError(
                        f"parameter {name!r} is declared by source ranks "
                        f"{owners[name]} and {source_rank}"
                    )
                owners[name] = source_rank

        missing = [name for name in full_info.names if name not in owners]
        if missing:
            raise ValueError(f"no source rank holds parameter {missing[0]!r}")

        return [
            NCCLWeightTransferUpdateInfo(
                names=[name for name in full_info.names if owners[name] == source_rank],
                dtype_names=[
                    dtype_name
                    for name, dtype_name in zip(full_info.names, full_info.dtype_names)
                    if owners[name] == source_rank
                ],
                shapes=[
                    shape
                    for name, shape in zip(full_info.names, full_info.shapes)
                    if owners[name] == source_rank
                ],
                src=source_rank,
            )
            for source_rank in range(num_source_ranks)
        ]

    @staticmethod
    def _iter_owned(
        source: WeightSource,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        held = source.held_names()
        iter_groups = getattr(type(source), "iter_groups", WeightSource.iter_groups)
        if held is None and iter_groups is WeightSource.iter_groups:
            yield from source
            return

        held_set = None if held is None else set(held)
        for names, tensors in source.iter_groups():
            for name, tensor in zip(names, tensors, strict=True):
                if held_set is None or name in held_set:
                    if tensor is None:
                        raise ValueError(
                            f"WeightSource holds {name!r} but yielded None"
                        )
                    yield name, tensor
                elif tensor is not None:
                    raise ValueError(
                        f"WeightSource does not hold {name!r} but yielded a tensor"
                    )

    @staticmethod
    def _checked_iter(
        pairs: Iterator[tuple[str, torch.Tensor]], meta: list[ParamMeta]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield the source's pairs, checking each against what the worker was
        told to expect.

        The worker sizes its receive buffers — and in packed mode cuts its chunk
        boundaries — from the update info, which is built from `metadata()`. If
        iteration disagrees with it, the two sides split the stream differently
        and the transfer hangs in NCCL or loads garbage. Checking here costs one
        comparison per parameter and turns that into an error naming the first
        divergent parameter.
        """
        sent = 0
        for name, tensor in pairs:
            if sent >= len(meta):
                raise ValueError(
                    f"WeightSource yielded more parameters than metadata() "
                    f"declared ({len(meta)}); first extra is {name!r}."
                )
            expected = meta[sent]
            if (
                name != expected.name
                or tensor.dtype != expected.dtype
                or tuple(tensor.shape) != expected.shape
            ):
                raise ValueError(
                    "WeightSource metadata() disagrees with iteration at index "
                    f"{sent}: declared {expected.name!r} "
                    f"{expected.dtype} {tuple(expected.shape)}, got {name!r} "
                    f"{tensor.dtype} {tuple(tensor.shape)}. Both channels must "
                    "enumerate the same parameters in the same order."
                )
            sent += 1
            yield name, tensor
        if sent != len(meta):
            raise ValueError(
                f"WeightSource yielded {sent} parameters but metadata() "
                f"declared {len(meta)}; the worker is waiting for the rest."
            )

    def _post_send_sync(self) -> None:
        """Wait for this rank's transfer work to land before returning.

        Broadcasts are only *enqueued* by `send_weights`: the unpacked path on
        the current stream, the packed path on the producer's own streams (which
        it drains itself). Waiting here lets a caller mutate parameters, or
        start the next step on another stream, as soon as `send_weights`
        returns, instead of silently depending on same-stream ordering. Every
        rank waits: a non-sender's `full_tensor()` gathers feed the sender's
        broadcast, so they must have landed before it may touch its shards.

        """
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

    def shutdown(self) -> None:
        self._abort_model_update_group()
        self.source_metadata_group = None
