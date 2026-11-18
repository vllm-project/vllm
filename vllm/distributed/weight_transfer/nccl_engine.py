# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based (dense) weight transfer engine."""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Self

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
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
    chunk_layout,
    packed_nccl_broadcast_consumer,
    packed_nccl_broadcast_producer,
    release_slot_buffers,
    slot_buffers,
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
    # PP-stage groups (`_stage_handshake`): `stage_setup` runs the handshake;
    # `stages` holds per stage its rounds (dicts of the fields above), on the
    # stage's group; `tail` then holds turns on the main group.
    stage_setup: bool = False
    stages: list[list[dict] | None] | None = None
    tail: list[dict] | None = None

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


def _stage_handshake(
    group: "PyNcclCommunicator",
    row: torch.Tensor,
    n_trainer: int,
    owner_flags=None,
) -> list[dict]:
    """Form one sub-communicator per pipeline stage, so stages receive only
    their own weights, in parallel.

    Every group rank contributes `row`: its PP rank (-1 on trainer ranks)
    followed by a 0/1 mask over the planned names of what it needs. Each stage
    needs the union of its workers' masks. Group rank 0 (a trainer rank) calls
    `owner_flags(need)` for which trainer ranks own a name some stage needs;
    a stage's group is those trainer ranks plus its workers, so a trainer rank
    joins only the rings it sends or must relay on."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    world = group.world_size
    out = torch.empty(world * row.numel(), dtype=row.dtype, device=group.device)
    group.all_gather(out, row)
    rows = out.view(world, -1).cpu()
    pp = rows[n_trainer:, 0].tolist()
    stages = sorted(set(pp))
    workers = [[n_trainer + i for i, r in enumerate(pp) if r == c] for c in stages]
    need = [rows[ranks, 1:].bool().any(0) for ranks in workers]
    flags = torch.zeros(len(stages) * n_trainer, dtype=torch.int32, device=group.device)
    if group.rank == 0:
        flags.copy_(owner_flags(need).flatten())
    group.broadcast(flags, src=0)
    flags = flags.view(len(stages), n_trainer).cpu()
    out_stages = []
    for c, ranks in enumerate(workers):
        members = [b for b in range(n_trainer) if flags[c, b]] + ranks
        uid = torch.zeros(128, dtype=torch.uint8, device=group.device)
        if group.rank == members[0]:
            raw = bytes(group.nccl.ncclGetUniqueId().internal)
            uid.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
        group.broadcast(uid, src=members[0])
        out_stages.append(
            {"members": members, "need": need[c], "uid": bytes(uid.cpu().tolist())}
        )
    for st in out_stages:  # in stage order on every rank: no init cycle
        st["comm"] = None
        if group.rank in st["members"]:
            st["comm"] = PyNcclCommunicator.from_unique_id_bytes(
                st["uid"],
                rank=st["members"].index(group.rank),
                world_size=len(st["members"]),
                device=group.device,
            )
    return out_stages


def _pairs(ops, rows: tuple[int, int] | None, view: torch.Tensor):
    """(destination, source) per planned write of a received piece; `None`
    when the tensor was split (`rows`: its dim-0 range here) and a write
    does not map dim 0 onto dim 0, so the tensor is needed whole."""
    if rows is None:
        return [(dst, view[box]) for box, dst in ops]
    out = []
    for box, dst in ops:
        if [b.stop - b.start for b in box] != list(dst.shape):
            return None
        lo, hi = max(box[0].start, rows[0]), min(box[0].stop, rows[1])
        if lo < hi:
            out.append(
                (
                    dst[lo - box[0].start : hi - box[0].start],
                    view[(slice(lo - rows[0], hi - rows[0]), *box[1:])],
                )
            )
    return out


def _coalesce(
    dsts: list[torch.Tensor], srcs: list[torch.Tensor], roles: list[str]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Merge copies with equal geometry whose destinations and sources are
    evenly spaced in one storage each (one weight of every expert in a chunk)
    into one copy over a view with a leading count dim. `roles` (the name
    without its indices) keeps apart copies that share a destination but
    interleave, e.g. w1 and w3 into each expert's fused w13. The plan's writes
    never overlap, so their order does not matter."""
    groups: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for d, s, role in zip(dsts, srcs, roles):
        key = (
            role,
            d.dtype,
            s.dtype,
            tuple(d.shape),
            d.stride(),
            s.stride(),
            d.untyped_storage().data_ptr(),
            s.untyped_storage().data_ptr(),
        )
        groups.setdefault(key, []).append((d, s))
    out_d: list[torch.Tensor] = []
    out_s: list[torch.Tensor] = []
    for pairs in groups.values():
        i = 0
        while i < len(pairs):
            (d0, s0), j = pairs[i], i
            if i + 1 < len(pairs):
                dstep = pairs[i + 1][0].storage_offset() - d0.storage_offset()
                sstep = pairs[i + 1][1].storage_offset() - s0.storage_offset()
                while (
                    dstep > 0
                    and sstep > 0
                    and j + 1 < len(pairs)
                    and pairs[j + 1][0].storage_offset() - pairs[j][0].storage_offset()
                    == dstep
                    and pairs[j + 1][1].storage_offset() - pairs[j][1].storage_offset()
                    == sstep
                ):
                    j += 1
            if j > i:
                n = j - i + 1
                d0 = d0.as_strided(
                    (n, *d0.shape), (dstep, *d0.stride()), d0.storage_offset()
                )
                s0 = s0.as_strided(
                    (n, *s0.shape), (sstep, *s0.stride()), s0.storage_offset()
                )
            out_d.append(d0)
            out_s.append(s0)
            i = j + 1
    return out_d, out_s


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
        self.model_update_group: PyNcclCommunicator | None = None
        # Set from the trainer-supplied init info at the handshake; defaults are
        # only for the (unreachable) receive-before-init case.
        self.rank_offset = 1
        self.packed = False
        self.packed_buffer_size_bytes = DEFAULT_PACKED_BUFFER_SIZE_BYTES
        self.packed_num_buffers = DEFAULT_PACKED_NUM_BUFFERS
        # Metadata of the last update per (model, src), reused by empty update info.
        self._last_update_info: dict[tuple, NCCLWeightTransferUpdateInfo] = {}
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
        self.__dict__.pop("_layouts", None)
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
        # The receive buffers' memory goes back (the KV cache wakes into it);
        # their views stay valid and the storage regrows at the next update.
        release_slot_buffers()
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
        if update_info.stage_setup:
            self._setup_stages()
            return
        if update_info.stages is None:
            self._receive(update_info, self.model_update_group)
            return
        stage, group = self._stage
        for one in update_info.stages[stage] or ():
            self._receive(NCCLWeightTransferUpdateInfo(**one), group)
        for one in update_info.tail or ():
            self._receive(NCCLWeightTransferUpdateInfo(**one), self.model_update_group)

    def _receive(
        self, update_info: NCCLWeightTransferUpdateInfo, group: "PyNcclCommunicator"
    ) -> None:
        assert group is not None
        main = group is self.model_update_group
        src = update_info.src
        if main and not 0 <= src < self.rank_offset:
            raise ValueError(
                f"NCCL update src must be a sender rank in [0, {self.rank_offset}), "
                f"got {src!r}."
            )
        key = (self.model, id(group), src)
        reused = not update_info.names
        if update_info.names:
            self._last_update_info[key] = update_info
        else:
            update_info = self._last_update_info.get(key, update_info)

        from vllm.model_executor.model_loader.reload import load_planned

        plan = self._plan if self.model is self._default_model else None
        if plan is not None and not reused:
            # reused metadata was checked when it first came
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
        if self.packed and plan is not None:
            self._receive_planned(update_info, src, plan, group)
        elif not main:
            raise RuntimeError("PP-stage groups need a packed load plan")
        elif self.packed:
            # Build iterator of (name, (shape, dtype)) from update_info
            def state_dict_info_iterator():
                for name, dtype_name, shape in zip(
                    update_info.names, update_info.dtype_names, update_info.shapes
                ):
                    dtype = getattr(torch, dtype_name)
                    yield (name, (shape, dtype))

            packed_nccl_broadcast_consumer(
                iterator=state_dict_info_iterator(),
                group=group,
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
                group.broadcast(weight, src=src, stream=torch.cuda.current_stream())
                load_weights([(name, weight)])
                del weight

    def _setup_stages(self) -> None:
        """Worker side of `_stage_handshake`: needs the plan (built by the
        first `start_weight_update`), which knows exactly what this rank writes."""
        from vllm.distributed.parallel_state import get_pp_group

        group = self.model_update_group
        assert group is not None and self._plan is not None
        assert self._plan_manifest is not None
        names = list(self._plan_manifest)
        row = torch.zeros(1 + len(names), dtype=torch.int32)
        row[0] = get_pp_group().rank_in_group
        # a fallback (None) may write anything here: count it as needed
        row[1:] = torch.tensor(
            [self._plan.dests.get(n) != () for n in names], dtype=torch.int32
        )
        stages = _stage_handshake(group, row.to(group.device), self.rank_offset)
        mine = [i for i, st in enumerate(stages) if st["comm"] is not None]
        assert len(mine) == 1, mine
        self._stage = (mine[0], stages[mine[0]]["comm"])

    def _planned_layout(
        self, update_info: NCCLWeightTransferUpdateInfo, src: int, plan, group
    ) -> dict:
        """Per packed chunk, the copies from the receive buffer into the
        model, built once per update metadata: an update then runs no
        per-tensor Python."""
        import gc

        import regex as re

        nbuf = self.packed_num_buffers
        cache = self.__dict__.setdefault("_layouts", {})
        layout = cache.get((id(group), src))
        if layout is not None and layout["names"] == update_info.names:
            return layout
        names, shapes = update_info.names, update_info.shapes
        dtypes = [getattr(torch, d) for d in update_info.dtype_names]
        wire = chunk_layout(zip(shapes, dtypes), self.packed_buffer_size_bytes)
        need = max(wire.sizes, default=0)
        bufs = slot_buffers(nbuf, need)
        index_of = re.compile(r"\.\d+(?=\.|$)")
        chunks = []
        for k, pieces in enumerate(wire.chunks):
            dsts: list[torch.Tensor] = []
            srcs: list[torch.Tensor] = []
            roles: list[str] = []
            reshaped: list = []  # sources not viewable as their destination
            pending: list = []  # unaligned, or split and needed whole: per receive
            fallbacks: list = []
            for piece, view in zip(pieces, wire.views(k, bufs[k % nbuf])):
                index, rows, dtype = piece[0], piece[1], piece[3]
                name = names[index]
                dest = plan.get_dest(name)
                ops = (
                    None
                    if dest is None
                    else [(tuple(slice(*b) for b in box), t) for box, t in dest]
                )
                pairs = None if ops is None else _pairs(ops, rows, view)
                if view.dtype != dtype or (rows is not None and pairs is None):
                    pending.append((piece, view, name, ops))
                    continue
                if pairs is None:
                    fallbacks.append((name, view))
                    continue
                for dst, part in pairs:
                    try:
                        part = part.view(dst.shape)
                    except RuntimeError:
                        reshaped.append((dst, part))
                        continue
                    for word in (torch.int64, torch.int32, torch.int16):
                        # move whole words (a byte-wise strided copy is far
                        # below bandwidth); a cast stays elementwise
                        if (
                            word.itemsize <= dst.element_size()
                            or dst.dtype != part.dtype
                        ):
                            break
                        try:
                            dst, part = dst.view(word), part.view(word)
                            break
                        except RuntimeError:
                            continue
                    dsts.append(dst)
                    srcs.append(part)
                    roles.append(index_of.sub(".#", name))
            dsts, srcs = _coalesce(dsts, srcs, roles)
            fb_bytes = sum(t.nbytes for _, t in fallbacks)
            chunks.append(
                (wire.sizes[k], dsts, srcs, reshaped, pending, fallbacks, fb_bytes)
            )
        layout = cache[(id(group), src)] = {
            "names": list(names),
            "shapes": shapes,
            "chunks": chunks,
            "need": need,
        }
        # These views live as long as the engine: keep the collector from
        # rescanning ~1e5 of them mid-transfer (full collections cost ~0.3 s).
        gc.collect()
        gc.freeze()
        return layout

    def _receive_planned(
        self, update_info: NCCLWeightTransferUpdateInfo, src: int, plan, group
    ) -> None:
        """Packed receive into reused buffers, then one batched copy per chunk;
        fallback names still go through `load_weights` in arrival order."""
        from vllm.distributed.weight_transfer.packed_tensor import _get_streams

        layout = self._planned_layout(update_info, src, plan, group)
        nbuf = self.packed_num_buffers
        bufs = slot_buffers(nbuf, layout["need"])  # regrows released storage
        streams = _get_streams(torch.accelerator.current_device_index(), nbuf)
        slot_done: list[torch.Event | None] = [None] * nbuf
        prev: torch.Event | None = None
        deferred: list[tuple[str, torch.Tensor]] = []
        joined: dict[str, torch.Tensor] = {}
        for i, (size, dsts, srcs, reshaped, pending, fallbacks, fb_bytes) in enumerate(
            layout["chunks"]
        ):
            s = i % nbuf
            done = slot_done[s]
            if done is not None:
                done.synchronize()  # its previous chunk is copied out
            with torch.cuda.stream(streams[s]):
                group.broadcast(bufs[s][:size], src=src)
                # chunk i's copies run after every earlier chunk's (fallback
                # loaders may keep views across calls, as on the packed path)
                if prev is not None:
                    streams[s].wait_event(prev)
                if dsts:
                    torch._foreach_copy_(dsts, srcs)
                for dst, part in reshaped:
                    dst.copy_(part.reshape(dst.shape))
                # Fallbacks keep their arrival order but load once per call:
                # a copy outlives the slot, and the plan keeps their writes
                # disjoint from every direct write.
                if fb_bytes > (64 << 20):  # too big to copy aside: load now
                    self.model.load_weights([*deferred, *fallbacks])
                    deferred = []
                else:
                    deferred.extend((n, t.clone()) for n, t in fallbacks)
                for (index, rows, shape, dtype, _, _), view, name, ops in pending:
                    full = view if view.dtype == dtype else view.clone().view(dtype)
                    full = full.view(shape)
                    pairs = (
                        None if ops is None or rows is None else _pairs(ops, rows, full)
                    )
                    if pairs is not None:  # a realigned piece of a split tensor
                        for dst, part in pairs:
                            dst.copy_(part.reshape(dst.shape))
                        continue
                    if rows is not None:  # a split tensor needed whole: join it
                        whole = joined.get(name)
                        if whole is None:
                            whole = joined[name] = torch.empty(
                                layout["shapes"][index], dtype=dtype, device=self.device
                            )
                        whole[rows[0] : rows[1]].copy_(full)
                        if rows[1] < whole.shape[0]:
                            continue
                        full = joined.pop(name)
                    if ops is None:
                        deferred.append((name, full))  # its own storage
                    for box, dst in ops or ():
                        dst.copy_(full[box].reshape(dst.shape))
                prev = torch.Event()
                prev.record(streams[s])
                slot_done[s] = prev
        for stream in streams:
            stream.synchronize()
        if deferred:
            self.model.load_weights(deferred)
            torch.accelerator.synchronize()

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
        self.model_update_group: PyNcclCommunicator | None = None
        # This rank's rank in the transfer group, set by trainer_init.
        self._group_rank = 0
        # Metadata the worker already holds; an unchanged round sends none.
        self._sent_meta: list[ParamMeta] | None = None
        # PP-stage groups: opt in by setting `stage_groups` (needs
        # `use_load_plan`); formed at the first update, once its start has
        # built the workers' plan.
        self.stage_groups = False
        self._stages: list[dict] | None = None
        self._stage_split: tuple | None = None  # rounds and tail per slices
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
        if self.stage_groups and self._stages is None:
            self._setup_stages(meta, group_ranks)
        if self._stages:
            self._send_stages(group_ranks, slices)
        else:
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

    def _setup_stages(self, meta: list[ParamMeta], group_ranks: list[int]) -> None:
        """Trainer side of `_stage_handshake`; rank 0 triggers the workers'
        side with an `update_weights` RPC that runs concurrently."""
        group = self.model_update_group
        assert group is not None and self._plan_manifest is not None
        names = list(self._plan_manifest)
        owner = {
            m.name: group_ranks.index(0 if m.owners is None else m.owners[0])
            for m in meta
        }

        size = {m.name: math.prod(m.shape) * m.dtype.itemsize for m in meta}

        def owner_flags(need: list[torch.Tensor]) -> torch.Tensor:
            # Each trainer rank joins one stage's ring only: the one it sends
            # the most bytes to (a rank on two rings halves both). What it
            # owns for other stages goes in a final turn on the main group.
            nbytes = torch.zeros(len(need), len(group_ranks), dtype=torch.float64)
            for c, mask in enumerate(need):
                for i in mask.nonzero().flatten().tolist():
                    if names[i] in owner:
                        nbytes[c, owner[names[i]]] += size[names[i]]
            flags = torch.zeros(len(need), len(group_ranks), dtype=torch.int32)
            for b in range(len(group_ranks)):
                if nbytes[:, b].sum() > 0:
                    flags[int(nbytes[:, b].argmax()), b] = 1
            return flags.to(group.device)

        row = torch.zeros(1 + len(names), dtype=torch.int32, device=group.device)
        row[0] = -1
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = (
                exe.submit(self.client.update_weights, {"stage_setup": True})
                if self.is_sender
                else None
            )
            stages = _stage_handshake(group, row, len(group_ranks), owner_flags)
            if future is not None:
                future.result()
        finally:
            exe.shutdown(wait=False)
        for st in stages:
            st["need"] = {names[i] for i in st["need"].nonzero().flatten().tolist()}
        self._stages = stages

    def _send_stages(
        self, group_ranks: list[int], slices: list[tuple[int, list[ParamMeta]]]
    ) -> None:
        """One RPC; each stage runs its rounds (one per trainer rank on its
        ring) back to back, then every rank joins the tail turns: what a rank
        owns for a stage whose ring it is not on, on the main group."""
        assert self._stages is not None and self.model_update_group is not None
        key = [(b, len(metas)) for b, metas in slices]
        split = self._stage_split
        if split is None or split[0] != key:
            rounds: list[list] = [[] for _ in self._stages]
            tail = []
            for broadcaster, metas in slices:
                b = group_ranks.index(broadcaster)
                home = [c for c, st in enumerate(self._stages) if b in st["members"]]
                if not home:
                    continue
                st = self._stages[home[0]]
                mine = [m for m in metas if m.name in st["need"]]
                rest = [
                    m
                    for m in metas
                    if any(
                        m.name in other["need"]
                        for c, other in enumerate(self._stages)
                        if c != home[0]
                    )
                ]
                if mine:
                    rounds[home[0]].append(
                        (st["comm"], st["members"].index(b), metas, mine)
                    )
                if rest:
                    tail.append((self.model_update_group, b, metas, rest))
            split = (key, rounds, tail, set())
            self._stage_split = split
        _, rounds, tail, sent = split

        def info(src: int, sub: list[ParamMeta]) -> dict:
            if id(sub) in sent:
                return {"src": src}  # the workers hold its metadata
            sent.add(id(sub))
            return {
                "names": [m.name for m in sub],
                "dtype_names": [_dtype_name(m.dtype) for m in sub],
                "shapes": [list(m.shape) for m in sub],
                "src": src,
            }

        payload = {
            "stages": [
                [info(src, sub) for _, src, _, sub in r] or None for r in rounds
            ],
            "tail": [info(src, sub) for _, src, _, sub in tail],
        }
        turns = [t for r in rounds for t in r if t[0] is not None] + tail
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = (
                exe.submit(self.client.update_weights, payload)
                if self.is_sender
                else None
            )
            for comm, src, metas, sub in turns:  # one ring per rank: in order
                if comm.rank != src:
                    self._discard(sub, src, comm)
                    continue
                assert self.source is not None
                keep = {m.name for m in sub}
                packed_nccl_broadcast_producer(
                    iterator=(
                        p
                        for p in self._checked_iter(self.source, metas)
                        if p[0] in keep
                    ),
                    group=comm,
                    src=src,
                    post_iter_func=lambda item: item[1],
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                    layout=self._wire_layout(sub),
                )
            if future is not None:
                future.result()
        finally:
            exe.shutdown(wait=False)

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
        for unused in ("stage_setup", "stages", "tail"):
            if not payload[unused]:
                del payload[unused]
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
                layout=self._wire_layout(meta),
            )
        else:
            stream = torch.cuda.current_stream()
            for _name, tensor in pairs:
                # NCCL sends `numel` elements straight from `data_ptr()`, so a
                # non-contiguous view would ship whatever follows its base
                # pointer. Keep the copy referenced until the broadcast is
                # enqueued. (The packed path linearizes in `pack_tensors`.)
                send = tensor if tensor.is_contiguous() else tensor.contiguous()
                self.model_update_group.broadcast(send, src=src, stream=stream)

    def _wire_layout(self, meta: list[ParamMeta]):
        """How `meta` travels, built once per parameter list."""
        cache = self.__dict__.setdefault("_wire_layouts", {})
        key = (len(meta), meta[0].name if meta else "", meta[-1].name if meta else "")
        layout = cache.get(key)
        if layout is None:
            layout = cache[key] = chunk_layout(
                ((m.shape, m.dtype) for m in meta), self.packed_buffer_size_bytes
            )
        return layout

    def _discard(
        self,
        meta: list[ParamMeta],
        src: int,
        group: "PyNcclCommunicator | None" = None,
    ) -> None:
        """Receive another broadcaster's parameters and drop them."""
        group = group or self.model_update_group
        assert group is not None
        device = group.device
        if self.packed:
            # Nothing is kept: receive each chunk into one buffer, unread. A
            # rank that lags here stalls the whole broadcast ring.
            sizes = self._wire_layout(meta).sizes
            buf = torch.empty(max(sizes, default=0), dtype=torch.uint8, device=device)
            for n in sizes:  # on the current stream
                group.broadcast(buf[:n], src=src)
            torch.accelerator.current_stream().synchronize()
        else:
            stream = torch.cuda.current_stream()
            for m in meta:
                weight = torch.empty(m.shape, dtype=m.dtype, device=device)
                group.broadcast(weight, src=src, stream=stream)

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
