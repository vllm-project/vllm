# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based (dense) weight transfer engine."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
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
from vllm.logger import init_logger

logger = init_logger(__name__)

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
    """

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
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
                        shapes, and the source rank

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
    supplies its own manifest when selected. Other trainer ranks only iterate
    their source to participate in source-side collectives.

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
        source_rank: int | None = 0,
        num_source_ranks: int = 1,
        packed: bool = True,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.source_rank = source_rank
        self.num_source_ranks = num_source_ranks
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.packed_num_buffers = packed_num_buffers
        self.model_update_group: PyNcclCommunicator | None = None
        self.update_infos: list[NCCLWeightTransferUpdateInfo] = []
        self.source_metadata_group: StatelessProcessGroup | None = None

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
        local_metadata_error = None
        try:
            local_meta = source.metadata()
        except Exception as exc:
            local_metadata_error = f"{type(exc).__name__}: {exc}"

        source_rank = engine.source_rank
        if source_rank is None:
            if local_metadata_error is not None:
                raise RuntimeError(
                    "NCCL source metadata validation failed: " + local_metadata_error
                )
            return engine

        local_update_info = None
        if local_metadata_error is None:
            assert local_meta is not None
            try:
                local_update_info = NCCLWeightTransferUpdateInfo(
                    names=[m.name for m in local_meta],
                    dtype_names=[str(m.dtype).split(".")[-1] for m in local_meta],
                    shapes=[list(m.shape) for m in local_meta],
                    src=source_rank,
                )
            except Exception as exc:
                local_metadata_error = f"{type(exc).__name__}: {exc}"

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
                    "update_info": (
                        asdict(local_update_info)
                        if local_update_info is not None
                        else None
                    ),
                }
            )
            errors = [
                f"source rank {envelope['source_rank']}: {envelope['error']}"
                for envelope in envelopes
                if envelope["error"] is not None
            ]
            if errors:
                raise RuntimeError(
                    "NCCL source metadata validation failed: " + "; ".join(errors)
                )
            engine.update_infos = [
                NCCLWeightTransferUpdateInfo(**envelope["update_info"])
                for envelope in envelopes
            ]
            engine._validate_update_infos()
        else:
            if local_metadata_error is not None:
                raise RuntimeError(
                    "NCCL source metadata validation failed: " + local_metadata_error
                )
            assert local_update_info is not None
            engine.update_infos = [local_update_info]

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
            for _ in source:
                pass
            self._post_send_sync()
            return

        if self.model_update_group is None:
            raise RuntimeError("trainer_init() must be called before send_weights().")
        if not self.update_infos:
            meta = source.metadata()
            self.update_infos = [
                NCCLWeightTransferUpdateInfo(
                    names=[item.name for item in meta],
                    dtype_names=[str(item.dtype).split(".")[-1] for item in meta],
                    shapes=[list(item.shape) for item in meta],
                    src=self.source_rank,
                )
            ]

        start_error = None
        if self.is_sender:
            try:
                self.client.start_weight_update()
            except BaseException as exc:
                start_error = f"{type(exc).__name__}: {exc}"
        self._check_start_outcomes(start_error)

        if not self.is_sender:
            try:
                self._send_updates()
            except BaseException:
                try:
                    self._receive_round_outcome()
                except BaseException:
                    logger.exception(
                        "failed to receive NCCL round outcome after local failure"
                    )
                raise
            self._receive_round_outcome()
            return

        try:
            self._send_updates()
            self.client.finish_weight_update()
        except BaseException as exc:
            try:
                self._publish_round_outcome(exc)
            except BaseException:
                logger.exception("failed to publish NCCL round failure")
            raise
        self._publish_round_outcome(None)

    def _send_updates(self) -> None:
        for update_info in self.update_infos:
            future = None
            executor = None
            if self.is_sender:
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    self.client.update_weights, asdict(update_info)
                )
                if future.done():
                    future.result()
            try:
                self._broadcast_update(update_info)
                if future is not None:
                    future.result()
            finally:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)

    def _check_start_outcomes(self, local_error: str | None) -> None:
        if self.source_metadata_group is None:
            if local_error is not None:
                raise RuntimeError(
                    "NCCL worker failed to start its update: " + local_error
                )
            return
        outcomes = self.source_metadata_group.all_gather_obj(
            {"source_rank": self.source_rank, "error": local_error}
        )
        errors = [
            f"source rank {outcome['source_rank']}: {outcome['error']}"
            for outcome in outcomes
            if outcome["error"] is not None
        ]
        if errors:
            raise RuntimeError("NCCL update failed to start: " + "; ".join(errors))

    def _publish_round_outcome(self, failure: BaseException | None) -> None:
        if self.source_metadata_group is None:
            return
        envelope = {
            "ok": failure is None,
            "error": (
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
        }
        for rank in range(1, self.num_source_ranks):
            self.source_metadata_group.send_obj(envelope, rank)
        expected_ack = {"phase": "round_outcome_consumed"}
        for rank in range(1, self.num_source_ranks):
            ack = self.source_metadata_group.recv_obj(rank)
            if ack != expected_ack:
                raise RuntimeError(
                    f"source rank {rank} sent an invalid round-outcome acknowledgement"
                )

    def _receive_round_outcome(self) -> None:
        if self.source_metadata_group is None:
            return
        outcome = self.source_metadata_group.recv_obj(0)
        self.source_metadata_group.send_obj({"phase": "round_outcome_consumed"}, 0)
        if not isinstance(outcome, dict) or set(outcome) != {"ok", "error"}:
            raise RuntimeError("source rank 0 sent an invalid NCCL round outcome")
        ok = outcome["ok"]
        error = outcome["error"]
        if not isinstance(ok, bool) or ok != (error is None):
            raise RuntimeError("source rank 0 sent an invalid NCCL round outcome")
        if not ok:
            if not isinstance(error, str) or not error:
                raise RuntimeError("source rank 0 sent an invalid NCCL round error")
            raise RuntimeError("NCCL worker update failed on source rank 0: " + error)

    def _broadcast_update(self, update_info: NCCLWeightTransferUpdateInfo) -> None:
        assert self.model_update_group is not None
        is_source = self.source_rank == update_info.src
        meta = [
            ParamMeta(name, getattr(torch, dtype_name), tuple(shape))
            for name, dtype_name, shape in zip(
                update_info.names,
                update_info.dtype_names,
                update_info.shapes,
            )
        ]
        if self.packed:
            if is_source:
                assert self.source is not None
                packed_nccl_broadcast_producer(
                    iterator=self._checked_iter(self.source, meta),
                    group=self.model_update_group,
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
                    group=self.model_update_group,
                    src=update_info.src,
                    post_unpack_func=lambda weights: None,
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                    device=self.model_update_group.device,
                )
        else:
            stream = torch.cuda.current_stream()
            tensors: Iterator[tuple[str, torch.Tensor]]
            if is_source:
                assert self.source is not None
                tensors = self._checked_iter(self.source, meta)
            else:
                tensors = (
                    (
                        item.name,
                        torch.empty(
                            item.shape,
                            dtype=item.dtype,
                            device=self.model_update_group.device,
                        ),
                    )
                    for item in meta
                )
            for _name, tensor in tensors:
                send = tensor if tensor.is_contiguous() else tensor.contiguous()
                self.model_update_group.broadcast(
                    send, src=update_info.src, stream=stream
                )
        self._post_send_sync()

    def _validate_update_infos(self) -> None:
        if [info.src for info in self.update_infos] != list(
            range(self.num_source_ranks)
        ):
            raise ValueError("source manifests must be ordered by source rank")
        owners: dict[str, int] = {}
        for info in self.update_infos:
            for name in info.names:
                if name in owners:
                    raise ValueError(
                        f"parameter {name!r} is declared by source ranks "
                        f"{owners[name]} and {info.src}"
                    )
                owners[name] = info.src

    @staticmethod
    def _checked_iter(
        source: WeightSource, meta: list[ParamMeta]
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
        for name, tensor in source:
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
        if self.model_update_group is not None:
            self.model_update_group.destroy()
            self.model_update_group = None
        self.source_metadata_group = None
