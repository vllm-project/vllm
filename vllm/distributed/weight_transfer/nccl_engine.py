# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based (dense) weight transfer engine."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Self

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.weight_transfer.nccl_common import (
        WeightTransferCommunicator,
    )
    from vllm.model_executor.model_loader.reload import LoadPlan

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
    build_worker_init_payload,
    run_trainer_rendezvous,
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
from vllm.utils.torch_utils import current_stream

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

    The sender opens its endpoint as NCCL rank 0, so it needs no `rank_offset`.
    `rank` (from `TrainerInitInfo`) identifies this trainer process.

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
    """Transfer group size: rank 0 and each distinct `ParamMeta.owners[0]`, plus
    the inference workers (`1 + num_workers` for a single sender)."""
    packed: bool = True
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    use_load_plan: bool = False
    """Have the workers write received tensors of the target model through a
    load plan instead of `load_weights` where the plan allows (off by default).
    The workers get `source.metadata` at init and plan from it at the first
    weight update, inside `start_weight_update`, so that call takes longer by
    the plan's trace time (with `HTTPVLLMWeightSyncClient`, raise its
    `timeout` for a very large model). Every update must send a subset of that
    metadata with the same dtypes and shapes; `send_weights` checks this on
    every trainer rank before contacting the workers. The check stays until
    re-init, even if planning failed and the workers fell back to
    `load_weights`. The inference server
    must run with `VLLM_RELOAD_MODE=direct` and without EPLB; the workers
    refuse otherwise at init."""


@dataclass
class NCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the dense NCCL weight transfer backend.

    Carries the parameter metadata and `src`, the broadcasting group rank; the
    packed wire params live on the init info.

    Empty lists reuse the previous update's metadata for the same model and `src`.
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


# name -> (dtype, shape) of the tensors a load plan is built for
Manifest = Mapping[str, tuple[torch.dtype, tuple[int, ...]]]


def _manifest(metadata: Iterable[ParamMeta]) -> Manifest:
    return {m.name: (m.dtype, tuple(m.shape)) for m in metadata}


def _dtype_name(dtype: torch.dtype) -> str:
    """The wire name of `dtype` (`torch.bfloat16` -> `"bfloat16"`)."""
    return str(dtype).split(".")[-1]


def _check_manifest(
    manifest: Manifest, sent: Iterable[tuple[str, torch.dtype, Sequence[int]]]
) -> None:
    """Every sent (name, dtype, shape) must be in the manifest sent at init; a
    subset is fine (a broadcaster sends only its own). The trainer checks
    before an update; the worker checks again for trainers that drive it out
    of band."""
    for name, dtype, shape in sent:
        declared = manifest.get(name)
        if declared == (dtype, tuple(shape)):
            continue
        if declared is None:
            problem = "was not declared in the metadata sent at init"
        else:
            problem = f"was declared at init as ({declared[0]}, {list(declared[1])})"
        raise ValueError(
            f"NCCL weight transfer: {name!r} sent as ({dtype}, {list(shape)}) "
            f"{problem}; with use_load_plan, re-init the engine to re-plan"
        )


def _metadata_payload(
    metadata: Iterable[ParamMeta],
) -> list[tuple[str, str, list[int]]]:
    return [(m.name, _dtype_name(m.dtype), list(m.shape)) for m in metadata]


class NCCLWeightTransferEngine(
    WeightTransferEngine[NCCLWeightTransferInitInfo, NCCLWeightTransferUpdateInfo]
):
    """Weight transfer engine using NCCL for communication between trainer and workers.

    Receives dense checkpoint-format weights broadcast from the trainer ranks
    below `rank_offset` and loads them with the layerwise reload lifecycle.
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
        self.model_update_group: WeightTransferCommunicator | None = None
        # Set from the trainer-supplied init info at the handshake; defaults are
        # only for the (unreachable) receive-before-init case.
        self.rank_offset = 1
        self.packed = False
        self.packed_buffer_size_bytes = DEFAULT_PACKED_BUFFER_SIZE_BYTES
        self.packed_num_buffers = DEFAULT_PACKED_NUM_BUFFERS
        # Metadata of the last update per (model, src), reused by empty update info.
        self._last_update_info: dict[
            tuple[torch.nn.Module, int], NCCLWeightTransferUpdateInfo
        ] = {}
        # With `use_load_plan`: the trainer's metadata from init, and the plan
        # for the target model, built at its first update.
        self._plan_manifest: Manifest | None = None
        self._plan_metadata: list[ParamMeta] | None = None
        self._plan: LoadPlan | None = None

    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """Initialize NCCL process group with the trainer and record the
        trainer-supplied wire params so the worker decodes exactly as the
        trainer encodes.

        Args:
            init_info: NCCL initialization info containing master address, port,
                      rank offset, world size, the packed wire params and,
                      with `use_load_plan`, the trainer's metadata

        Raises:
            ValueError: `use_load_plan` without metadata.
            RuntimeError: `use_load_plan` with a configuration no load plan
                supports (`check_plannable`).

        """
        from vllm.model_executor.model_loader.reload import check_plannable

        self.rank_offset = init_info.rank_offset
        self.packed = init_info.packed
        self.packed_buffer_size_bytes = init_info.packed_buffer_size_bytes
        self.packed_num_buffers = init_info.packed_num_buffers
        self._last_update_info.clear()
        self._plan = self._plan_metadata = self._plan_manifest = None
        self.model_update_group = worker_init_process_group(
            init_info, self.parallel_config
        )
        # After the group forms, so these errors reach the trainer; before any
        # `start_reload`, so a refusal leaves the model untouched. A refusal
        # destroys the group, so a retry starts clean.
        if init_info.use_load_plan:
            try:
                if not init_info.metadata:
                    raise ValueError(
                        "NCCL weight transfer: use_load_plan needs the trainer's "
                        "metadata in the init info"
                    )
                check_plannable(self._default_model)
                self._plan_metadata = [
                    ParamMeta(name, getattr(torch, dtype_name), tuple(shape))
                    for name, dtype_name, shape in init_info.metadata
                ]
            except Exception:
                self._plan_metadata = None
                self.shutdown()
                raise
            self._plan_manifest = _manifest(self._plan_metadata)

    def start_weight_update(self) -> None:
        """Begin reloading the incoming checkpoint weights; with
        `use_load_plan`, the first target update also builds the load plan,
        reused by every later one."""
        from vllm.model_executor.model_loader.reload import plan_loads, start_reload

        start_reload(self.model)
        if (
            self._plan_metadata is not None
            and self._plan is None
            and self.model is self._default_model  # not a draft session
        ):
            # The bake traces the real tensors `start_reload` just viewed back
            # to their checkpoint layout, so every offset and destination is
            # the one direct mode applied. Keep it after `start_reload`.
            try:
                self._plan = plan_loads(self.model, self._plan_metadata)
            except Exception as e:
                self._plan_metadata = self._plan_manifest = None
                e.add_note(
                    "NCCL weight transfer: load planning failed after "
                    "start_reload, so the model is mid-reload; later updates "
                    "use load_weights, which repairs it; re-init the engine "
                    "to retry planning"
                )
                raise

    def finish_weight_update(self) -> None:
        """Finish reloading after all weights have been received."""
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)
        # Release the cached receive buffers and post-load temporaries so memory
        # allocated outside torch (NCCL, other processes on the GPU) can use it.
        torch.accelerator.empty_cache()

    def receive_weights(self, update_info: NCCLWeightTransferUpdateInfo) -> None:
        """Receive weights from trainer via NCCL broadcast.

        Whether to use packed broadcasting (and the buffer geometry) is read
        from `self.packed` / `self.packed_*`, set at the init handshake from the
        trainer's init info, so it is guaranteed to match how the trainer
        encoded.

        With a load plan, a target update checks the update against the
        metadata sent at init (`_check_manifest`) and writes through
        `load_planned`; draft updates use `load_weights`.

        Args:
            update_info: Parameter names, dtypes, shapes and `src`; empty lists
                        reuse the previous call's from the same `src`

        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )
        src = update_info.src
        if not 0 <= src < self.rank_offset:
            raise ValueError(
                f"NCCL update src must be a sender rank in [0, {self.rank_offset}), "
                f"got {src!r}."
            )
        key = (self.model, src)
        if update_info.names:
            self._last_update_info[key] = update_info
        else:
            update_info = self._last_update_info.get(key, update_info)

        from vllm.model_executor.model_loader.reload import load_planned

        plan = self._plan if self.model is self._default_model else None
        if plan is not None:
            assert self._plan_manifest is not None
            _check_manifest(
                self._plan_manifest,
                (
                    (name, getattr(torch, dtype_name), shape)
                    for name, dtype_name, shape in zip(
                        update_info.names, update_info.dtype_names, update_info.shapes
                    )
                ),
            )
        model = self.model
        load_weights = (
            model.load_weights if plan is None else partial(load_planned, model, plan)
        )
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
                src=src,
                post_unpack_func=load_weights,
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
                    weight, src=src, stream=current_stream()
                )
                load_weights([(name, weight)])
                del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.destroy()
            self.model_update_group = None


class NCCLTrainerWeightTransferEngine(TrainerWeightTransferEngine[NCCLTrainerInitInfo]):
    """Trainer-side NCCL weight transfer engine.

    Each parameter is broadcast by its first owner (rank 0 when unset); rank 0
    also drives the client RPCs. See "Broadcasting ranks" in `nccl.md`.

    Other trainer ranks hold no communicator and only drive `iter_full()` to
    stay in the trainer-side gathers (e.g. FSDP `full_tensor()`).

    `packed` / buffer sizes come from `NCCLTrainerInitInfo`; the sender
    propagates them to the worker at `trainer_init` (on the worker-side init
    info), so per-round payloads carry only parameter metadata.
    """

    init_info_cls = NCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        rank: int = 0,
        packed: bool = True,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender, rank=rank)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.packed_num_buffers = packed_num_buffers
        self.model_update_group: WeightTransferCommunicator | None = None
        # This rank's rank in the transfer group, set by trainer_init.
        self._group_rank = 0
        # Metadata the worker already holds; an unchanged round sends none.
        self._sent_meta: list[ParamMeta] | None = None
        # With `use_load_plan`: the metadata the workers plan from, sent at init.
        self._plan_manifest: Manifest | None = None

    @classmethod
    def _build_worker_init_payload(
        cls,
        init_info: NCCLTrainerInitInfo,
        *,
        metadata: list[ParamMeta] | None = None,
    ) -> dict:
        """`metadata`: what the workers plan from; passed only with
        `use_load_plan`, so overrides without it keep working otherwise."""
        return build_worker_init_payload(
            init_info,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
            use_load_plan=init_info.use_load_plan,
            metadata=None if metadata is None else _metadata_payload(metadata),
        )

    def _open_trainer_endpoint(self, init_info: NCCLTrainerInitInfo) -> None:
        self.model_update_group = open_trainer_endpoint(init_info, self._group_rank)

    @staticmethod
    def _broadcast_plan(
        meta: list[ParamMeta],
    ) -> tuple[list[int], list[tuple[int, list[ParamMeta]]]]:
        """The transfer group's trainer ranks, in group-rank order, and each
        broadcaster's parameters (in metadata order), in broadcaster order."""
        slices: dict[int, list[ParamMeta]] = {}
        for m in meta:
            slices.setdefault(0 if m.owners is None else m.owners[0], []).append(m)
        group_ranks = sorted({0, *slices})
        return group_ranks, sorted(slices.items())

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
        if not source.implements_iter_full():
            raise TypeError(
                f"NCCL broadcast sends full tensors, but {type(source).__name__} "
                "does not implement iter_full()."
            )
        # Identical on every rank, so every rank raises alike.
        group_ranks, slices = cls._broadcast_plan(source.metadata)
        for m in source.metadata:
            if m.owners is not None and 0 in m.owners[1:]:
                raise ValueError(
                    f"parameter {m.name!r} has owners {m.owners}: rank 0 only "
                    "gathers in its own turn, so it must be the first owner of "
                    "every parameter it owns."
                )
        if len(slices) > 1:
            broadcaster_of: dict[int, int] = {}
            for broadcaster, metas in slices:
                for m in metas:
                    if m.owners is None:
                        raise ValueError(
                            f"parameter {m.name!r} has no owners, so every rank "
                            "owns it, but the source has several broadcasting "
                            f"ranks {[r for r, _ in slices]}. Every parameter "
                            "must declare its owners."
                        )
                    for r in m.owners:
                        prev = broadcaster_of.setdefault(r, broadcaster)
                        if prev != broadcaster:
                            raise ValueError(
                                f"rank {r} owns parameters broadcast by ranks "
                                f"{prev} and {broadcaster}, but each broadcaster "
                                "sends in its own turn, so every parameter a "
                                "rank owns must share its first owner."
                            )
        if init_info.world_size <= len(group_ranks):
            raise ValueError(
                f"world_size {init_info.world_size} leaves no inference workers "
                f"after the {len(group_ranks)} broadcasting trainer ranks."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            rank=init_info.rank,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )
        plan_meta = source.metadata if init_info.use_load_plan else None
        if plan_meta is not None:
            engine._plan_manifest = _manifest(plan_meta)
        if engine.rank not in group_ranks:
            # Ranks that broadcast nothing stay out of the transfer group; they
            # only join the trainer-side gathers in send_weights.
            return engine
        engine._group_rank = group_ranks.index(engine.rank)
        if not engine.is_sender:
            engine._open_trainer_endpoint(init_info)
            return engine

        # without the plan, keep calling overrides that take no `metadata`
        if plan_meta is None:
            payload = cls._build_worker_init_payload(init_info)
        else:
            payload = cls._build_worker_init_payload(init_info, metadata=plan_meta)
        payload["rank_offset"] = len(group_ranks)
        try:
            run_trainer_rendezvous(
                init_worker=lambda: engine.client.init_weight_transfer_engine(payload),
                init_trainer=lambda: engine._open_trainer_endpoint(init_info),
            )
        except BaseException:
            engine.shutdown()  # init failed (e.g. the workers refused): free the group
            raise

        return engine

    def send_weights(self) -> None:
        assert self.source is not None  # guaranteed by trainer_init / __init__
        source = self.source

        # Metadata is declared without gathering. For Megatron it is itself a
        # collective, so every rank runs it; only the sender ships it.
        meta = source.metadata
        if self._plan_manifest is not None:
            # Before any RPC or broadcast, on every rank alike: a worker that
            # refused mid-broadcast would leave the trainer waiting.
            _check_manifest(
                self._plan_manifest, ((m.name, m.dtype, m.shape) for m in meta)
            )
        group_ranks, slices = self._broadcast_plan(meta)

        if self.rank not in group_ranks:
            # Ranks outside the transfer group only join the trainer-side
            # gather collective.
            for _ in source.iter_full():
                pass
            self._post_send_sync()
            return

        reuse = self.is_sender and meta == self._sent_meta
        if self.is_sender:
            self.client.start_weight_update()
        for broadcaster, metas in slices:
            src = group_ranks.index(broadcaster)
            if self.is_sender:
                self._send_update(metas, src, reuse)
            else:
                self._broadcast(metas, src)
        if self.is_sender:
            self.client.finish_weight_update()
            self._sent_meta = meta
        self._post_send_sync()

    def _send_update(self, meta: list[ParamMeta], src: int, reuse: bool) -> None:
        """Rank 0's part of one broadcaster's turn: run its `update_weights`
        concurrently with `_broadcast`."""
        if reuse:
            update_info = NCCLWeightTransferUpdateInfo(src=src)
        else:
            update_info = NCCLWeightTransferUpdateInfo(
                names=[m.name for m in meta],
                dtype_names=[_dtype_name(m.dtype) for m in meta],
                shapes=[list(m.shape) for m in meta],
                src=src,
            )
        payload = asdict(update_info)
        if not src:
            # Keep single-broadcaster payloads readable by older servers.
            del payload["src"]
        # update_weights (workers receive) must run concurrently with the
        # trainer-side broadcast — both rendezvous inside the same NCCL calls.
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = exe.submit(self.client.update_weights, payload)
            # Cheap best-effort: if update_weights already failed (e.g. a bad
            # request rejected before any NCCL call), surface it now instead of
            # hanging in broadcast waiting for a peer that will never arrive.
            if future.done():
                future.result()
            self._broadcast(meta, src)
            future.result()  # surface inference-side errors
        finally:
            # Never wait for the RPC thread here. If the broadcast raised, the
            # worker is still blocked in the matching NCCL call and will never
            # return, so joining would turn the error into a permanent hang.
            # Let the exception out instead; the transfer group is unusable
            # either way and the caller has to tear it down. (The orphaned
            # thread is still joined at interpreter exit, so a caller that wants
            # to exit cleanly after such a failure must not wait on it.)
            exe.shutdown(wait=False)

    def _broadcast(self, meta: list[ParamMeta], src: int) -> None:
        """Broadcast `meta` from group rank `src`: that rank sends from
        `iter_full()`, every other group member receives and discards."""
        assert self.model_update_group is not None, (
            "trainer_init() must be called before _broadcast()."
        )
        if src != self.model_update_group.rank:
            self._discard(meta, src)
            return
        assert self.source is not None
        pairs = self._checked_iter(self.source, meta)
        if self.packed:
            packed_nccl_broadcast_producer(
                iterator=pairs,
                group=self.model_update_group,
                src=src,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=self.packed_buffer_size_bytes,
                num_buffers=self.packed_num_buffers,
            )
        else:
            stream = current_stream()
            for _name, tensor in pairs:
                # NCCL sends `numel` elements straight from `data_ptr()`, so a
                # non-contiguous view would ship whatever follows its base
                # pointer. Keep the copy referenced until the broadcast is
                # enqueued. (The packed path linearizes in `pack_tensors`.)
                send = tensor if tensor.is_contiguous() else tensor.contiguous()
                self.model_update_group.broadcast(send, src=src, stream=stream)

    def _discard(self, meta: list[ParamMeta], src: int) -> None:
        """Receive another broadcaster's parameters and drop them."""
        assert self.model_update_group is not None
        device = self.model_update_group.device
        if self.packed:
            packed_nccl_broadcast_consumer(
                iterator=((m.name, (list(m.shape), m.dtype)) for m in meta),
                group=self.model_update_group,
                src=src,
                post_unpack_func=lambda weights: None,
                buffer_size_bytes=self.packed_buffer_size_bytes,
                num_buffers=self.packed_num_buffers,
                device=device,
            )
        else:
            stream = current_stream()
            for m in meta:
                weight = torch.empty(m.shape, dtype=m.dtype, device=device)
                self.model_update_group.broadcast(weight, src=src, stream=stream)

    @staticmethod
    def _checked_iter(
        source: WeightSource, meta: list[ParamMeta]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield the sender's pairs, checking each against what the worker was
        told to expect.

        The worker sizes its receive buffers — and in packed mode cuts its chunk
        boundaries — from the update info, which is built from `metadata`. If
        iteration disagrees with it, the two sides split the stream differently
        and the transfer hangs in NCCL or loads garbage. Checking here costs one
        comparison per parameter and turns that into an error naming the first
        divergent parameter.
        """
        sent = 0
        for name, tensor in source.iter_full():
            if sent >= len(meta):
                raise ValueError(
                    f"WeightSource yielded more parameters than metadata "
                    f"declared ({len(meta)}); first extra is {name!r}."
                )
            expected = meta[sent]
            if (
                name != expected.name
                or tensor.dtype != expected.dtype
                or tuple(tensor.shape) != expected.shape
            ):
                raise ValueError(
                    "WeightSource metadata disagrees with iteration at index "
                    f"{sent}: declared {expected.name!r} "
                    f"{expected.dtype} {tuple(expected.shape)}, got {name!r} "
                    f"{tensor.dtype} {tuple(tensor.shape)}. Both channels must "
                    "enumerate the same parameters in the same order."
                )
            sent += 1
            yield name, tensor
        if sent != len(meta):
            raise ValueError(
                f"WeightSource yielded {sent} parameters but metadata "
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

        Unlike IPC there is no cross-rank barrier here. Nothing in this backend
        outlives the collective it travelled in (IPC's barrier keeps shared
        buffers alive until every consumer has opened them), so a barrier would
        only add a dependency on the default process group that this backend
        otherwise does not have.
        """
        if torch.accelerator.is_available():
            torch.accelerator.current_stream().synchronize()

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.destroy()
            self.model_update_group = None
