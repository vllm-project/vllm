# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served GLM-5.3 decode kernels built from CuTe DSL: entries, host-wrapper arguments
and compiled launchers.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import importlib.metadata
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .moe_q_shape import Q8Flag, Q8Shape
from .row_table import PACKED_SLOTS
from .w8a8_shape import TIERS, TierShape

if TYPE_CHECKING:
    from cutlass.base_dsl.jit_executor import JitCompiledFunction, JitExecutor

logger = logging.getLogger(__name__)

EXAMPLE_ADDRESS = 1 << 20
PACKAGE_DIR = Path(__file__).resolve().parent
ELEMENT_TYPES = {
    "bfloat16": "BFloat16",
    "float32": "Float32",
    "float8_e4m3": "Float8E4M3FN",
    "uint32": "Uint32",
    "int32": "Int32",
    "uint16": "Uint16",
    "uint8": "Uint8",
    "uint64": "Uint64",
    "int64": "Int64",
}


@dataclass(frozen=True)
class CuteEntry:
    """One served kernel: its family-module class, compile-time knobs and host-read
    sizes.
    """

    module: str
    cls: str
    constants: tuple[tuple[str, int | bool], ...] = ()
    #: Sizes the providers read (peer mailbox bytes, flag words, head shapes).
    host: dict[str, int] = field(default_factory=dict, compare=False)


_SELECTOR = {
    "kMaxContext": 1048576,
    "kShardSelectionBytes": 3020800,
    "kKeyTile": 128,
    "kFlagWords": 6840,
    "kStateWords": 330102,
}
_SELECTOR_BLOCKS = ("kShardSelectionBytes", "kFlagWords", "kStateWords")
SELECTOR_RANKS = 3
#: Selector ranks of the request-parallel ``serve8`` builds (every rank selects).
WIDE_SELECTOR_RANKS = 8
_WORKER = {
    "kFlagWords": 15804,
    "kHandoffWords": 59392,
    "kPartialOBytes": 2097152,
    "kPartialLseBytes": 6144,
}


#: The worker roles: entry name part, class, and whether the role owns an output slice.
_WORKER_ROLES = (
    ("full_solo", "FullWorker", False),
    ("shared", "SharedWorker", True),
    ("shared_seam", "SeamWorker", True),
)
_PUSH_MAILBOX = 1009792
_HANDOFF_BLOCK = 59680


def _sel(
    shard: int,
    cls: str = "SelectorFull",
    knobs: tuple[tuple[str, int | bool], ...] = (),
) -> CuteEntry:
    return CuteEntry("mla_selector", cls, (("SHARD", shard),) + knobs, _SELECTOR)


def _worker(cls: str, **host: int) -> CuteEntry:
    return CuteEntry("mla_worker", cls, host=_WORKER | host)


DEAD_ROW_SHAPES = ((8, 5), (8, 6), (8, 7))


def _q8(seam: bool, dead: tuple[int, int] | None = None) -> CuteEntry:
    return CuteEntry(
        "fused_moe",
        "FusedMoeQ8Seam" if seam else "FusedMoeQ8",
        (("DEAD_STRIDE", dead[0]), ("DEAD_LIVE", dead[1])) if dead else (),
        host={
            "kRows": Q8Shape.ROWS,
            "kMailboxBytes": (
                Q8Shape.SEAM_MAILBOX_BYTES if seam else Q8Shape.MAILBOX_BYTES
            ),
            "kFlagWords": Q8Flag.WORDS,
        },
    )


ENTRIES: dict[str, CuteEntry] = {
    "dense_ffn_q4_vllm": CuteEntry(
        "dense_ffn", "DenseFfn", host={"kMailboxBytes": 917504, "kFlagWords": 3840}
    ),
    "fused_moe_q4_ckpt_shared": CuteEntry(
        "fused_moe", "FusedMoeQ4", host={"kMailboxBytes": 917504, "kFlagWords": 14952}
    ),
    "fused_moe_q4_seam_ckpt_shared": CuteEntry(
        "fused_moe",
        "FusedMoeSeam",
        host={"kMailboxBytes": 1048576, "kFlagWords": 14952},
    ),
    "fused_moe_q8_ckpt_shared": _q8(False),
    "fused_moe_q8_seam_ckpt_shared": _q8(True),
    **{
        f"fused_moe_q8d{rows}x{live}{'_seam' if seam else ''}_ckpt_shared": _q8(
            seam, (rows, live)
        )
        for rows, live in DEAD_ROW_SHAPES
        for seam in (False, True)
    },
    "mla_worker_q4_full_paged_vlin": _worker("FullWorker"),
    "mla_worker_q4_full_push_paged_vlin": _worker(
        "PushWorker", kMailboxBytes=_PUSH_MAILBOX
    ),
    "mla_worker_q4_full_pushed_paged_vlin": _worker("PushedWorker"),
    "mla_worker_q4_shared_paged_vlin": _worker("SharedWorker", kMailboxBytes=917632),
    "mla_worker_q4_shared_seam_paged_vlin": _worker("SeamWorker", kMailboxBytes=917632),
    "prologue_q4": CuteEntry("prologue", "PrologueQ4"),
    "prologue_vllm_q4": CuteEntry("prologue", "PrologueVllm"),
    **{f"mla_selector_q4_shard3r{shard}_paged_vlin": _sel(shard) for shard in range(3)},
}


#: Row capacities (Rmax) with row-tier entries (``launch.TIERS`` without the q4 build).
ROW_TIERS = (8, 16)
#: MTP drafts per request the tier entries' request capacity (Bmax) is derived for.
TIER_DRAFTS = 3
LONE_TIERS = (8,)


def _worker_entry(
    rmax: int,
    role: str,
    cls: str,
    mailbox: bool,
    knobs: tuple[tuple[str, int | bool], ...],
    packed: bool = False,
) -> CuteEntry:
    groups = rmax // 4
    blocks = {name: size * groups for name, size in _WORKER.items()}
    blocks["kHandoffWords"] = _HANDOFF_BLOCK * groups
    if mailbox:
        blocks["kMailboxBytes"] = _PUSH_MAILBOX * groups
    tier = knobs + (("PASS_ROWS", rmax),)
    if role == "full_solo":
        tier += (("REFINE", False),)
    tier += (("PREFETCH_LAST", True),)
    tier += (("WIDE", True), ("ATTN_LOOP", True))
    if groups % 4 == 0:
        tier += (("WIDE_TAIL", 2),)
    if packed:
        tier += (("PACKED", True),)
    return CuteEntry("mla_worker", cls, tier, blocks)


def _row_tier_entries() -> dict[str, CuteEntry]:
    entries: dict[str, CuteEntry] = {}
    for rmax in ROW_TIERS:
        knobs = (("ROWS", rmax), ("REQUESTS", rmax // (TIER_DRAFTS + 1)))
        entries[f"prologue_vllm_r{rmax}"] = CuteEntry("prologue", "PrologueVllm", knobs)
        entries[f"prologue_r{rmax}"] = CuteEntry("prologue", "PrologueQ4", knobs)
        # The dense FFN's mailbox holds 7 peers x 2 parities x 128 slices x rmax rows.
        entries[f"dense_ffn_r{rmax}_vllm"] = CuteEntry(
            "dense_ffn",
            "DenseFfn",
            (("ROWS", rmax),),
            host={"kMailboxBytes": 229376 * rmax, "kFlagWords": 3840},
        )
        groups = rmax // 4
        for role, cls, mailbox in _WORKER_ROLES:
            entries[f"mla_worker_r{rmax}_{role}_paged_vlin"] = _worker_entry(
                rmax, role, cls, mailbox, knobs
            )
        splits = ["serve"] + (["gserve"] if rmax in LONE_TIERS else [])
        blocks = {
            name: -(-_SELECTOR[name] // 4) * 4 * groups for name in _SELECTOR_BLOCKS
        }
        for split in splits:
            for rank in range(WIDE_SELECTOR_RANKS):
                extra: tuple[tuple[str, int | bool], ...] = (
                    ("SHARDS", 1),
                    ("SERVE", WIDE_SELECTOR_RANKS),
                    ("RANK", rank),
                )
                if split == "gserve":
                    extra += (("GROUP_SERVE", True),)
                selector = _sel(0, knobs=knobs + extra)
                name = (
                    f"mla_selector_r{rmax}_{split}{WIDE_SELECTOR_RANKS}r{rank}"
                    "_paged_vlin"
                )
                entries[name] = CuteEntry(
                    selector.module,
                    selector.cls,
                    selector.constants,
                    _SELECTOR | blocks,
                )
    return entries


ENTRIES.update(_row_tier_entries())

PACKED_TIERS = (8, 16)


def _packed_entries() -> dict[str, CuteEntry]:
    entries: dict[str, CuteEntry] = {}
    for rmax in PACKED_TIERS:
        knobs = (("ROWS", rmax), ("REQUESTS", PACKED_SLOTS[rmax]))
        # vLLM's embedding shards (served) and a replicated table.
        for name, cls in (
            ("prologue_vllm", "PrologueVllm"),
            ("prologue", "PrologueQ4"),
        ):
            entries[f"{name}_r{rmax}p"] = CuteEntry(
                "prologue", cls, knobs + (("PACKED", True),)
            )
        for role, cls, mailbox in _WORKER_ROLES:
            entries[f"mla_worker_r{rmax}p_{role}_paged_vlin"] = _worker_entry(
                rmax, role, cls, mailbox, knobs, packed=True
            )
        slots = PACKED_SLOTS[rmax]
        blocks = {
            name: -(-_SELECTOR[name] // 4) * 4 * max(slots, rmax // 4)
            for name in _SELECTOR_BLOCKS
        }
        for rank in range(WIDE_SELECTOR_RANKS):
            extra = (
                ("SHARDS", 1),
                ("SERVE", WIDE_SELECTOR_RANKS),
                ("RANK", rank),
                ("SEGMENTS", True),
                ("SEGMENT_SLOTS", slots),
            )
            selector = _sel(0, knobs=knobs + extra)
            name = f"mla_selector_r{rmax}p_serve{WIDE_SELECTOR_RANKS}r{rank}_paged_vlin"
            entries[name] = CuteEntry(
                selector.module, selector.cls, selector.constants, _SELECTOR | blocks
            )
    return entries


ENTRIES.update(_packed_entries())


def _moe_tier(rmax: int, seam: bool = False) -> CuteEntry:
    shape = TierShape(rmax, seam)
    return CuteEntry(
        "fused_moe_w8a8",
        "FusedMoeW8A8Seam" if seam else "FusedMoeW8A8",
        (("RMAX", rmax),),
        host={
            "kMailboxBytes": shape.mailbox_bytes,
            "kFlagWords": shape.flag_words,
            "kWorkBytes": shape.work_bytes,
            "kRowTableWords": 2 + 3 * rmax,
        },
    )


ENTRIES.update({f"fused_moe_r{r}_ckpt_shared": _moe_tier(r) for r in TIERS})
ENTRIES.update({f"fused_moe_r{r}_seam_ckpt_shared": _moe_tier(r, True) for r in TIERS})


_HEAD = {
    "kMailboxBytes": 1024,
    "kVocabShard": 19360,
    "kHidden": 6144,
    "kPaddedRows": 19362,
}
_DRAFT_WORKER = {"kMailboxBytes": 917504, "kFlagWords": 15804}
_DRAFT_PRE = {"kMailboxBytes": 917504}
DRAFT_ENTRIES: dict[str, CuteEntry] = {
    "lm_head_top1_q1": CuteEntry(
        "lm_head",
        "LmHead",
        (("ROWS", 1),),
        {"kQTokens": 1, **_HEAD, "kDoneWords": 299, "kRecordWords": 4},
    ),
    "lm_head_top1_q4": CuteEntry(
        "lm_head",
        "LmHead",
        (("ROWS", 4),),
        {"kQTokens": 4, **_HEAD, "kDoneWords": 1187, "kRecordWords": 4},
    ),
    "mla_worker_draft_q1_paged": CuteEntry(
        "mla_draft",
        "DraftWorker",
        (("ROWS", 1), ("GROUP_ROWS", 1)),
        {
            **_DRAFT_WORKER,
            "kHandoffWords": 14848,
            "kPartialOBytes": 524288,
            "kPartialLseBytes": 1024,
        },
    ),
    "mla_worker_draft_q4_paged": CuteEntry(
        "mla_draft",
        "DraftWorker",
        (("ROWS", 4),),
        {
            **_DRAFT_WORKER,
            "kHandoffWords": 59392,
            "kPartialOBytes": 2097152,
            "kPartialLseBytes": 4096,
        },
    ),
    "mtp_preprocess_q1": CuteEntry(
        "mtp_preprocess", "MtpPreprocess", (("ROWS", 1),), _DRAFT_PRE
    ),
    "mtp_preprocess_q4": CuteEntry(
        "mtp_preprocess", "MtpPreprocess", (("ROWS", 4),), _DRAFT_PRE
    ),
    "prologue_vllm": CuteEntry("prologue", "PrologueVllm", (("ROWS", 1),)),
    **{
        f"mla_selector_q4_shard3r{shard}_only_row_paged": _sel(shard, "SelectorOnlyRow")
        for shard in range(3)
    },
}
ENTRIES.update(DRAFT_ENTRIES)

DRAFT_REQUESTS = (2, 4)
_DRAFT_ROW = {
    "kHandoffWords": 14848,
    "kPartialOBytes": 524288,
    "kPartialLseBytes": 1024,
}


def _draft_tier_entries() -> dict[str, CuteEntry]:
    entries: dict[str, CuteEntry] = {}
    for requests in DRAFT_REQUESTS:
        rows = 4 * requests
        catch_up = (("ROWS", rows), ("REQUESTS", requests))
        sizes = _SELECTOR | {
            name: -(-_SELECTOR[name] // 4) * 4 * requests for name in _SELECTOR_BLOCKS
        }
        for shard in range(SELECTOR_RANKS):
            selector = _sel(shard, "SelectorOnlyRow", knobs=catch_up)
            entries[
                f"mla_selector_r{rows}_shard{SELECTOR_RANKS}r{shard}_only_row_paged"
            ] = CuteEntry(selector.module, selector.cls, selector.constants, sizes)
        for name, group_rows, part_rows in (
            (f"mla_worker_draft_r{rows}_paged", 4, rows),
            (f"mla_worker_draft_p{requests}_paged", 1, requests),
        ):
            entries[name] = CuteEntry(
                "mla_draft",
                "DraftWorker",
                (
                    ("ROWS", part_rows),
                    ("GROUP_ROWS", group_rows),
                    ("REQUESTS", requests),
                ),
                {
                    "kMailboxBytes": _PUSH_MAILBOX * requests,
                    "kFlagWords": _WORKER["kFlagWords"] * (part_rows // group_rows),
                    **{key: value * part_rows for key, value in _DRAFT_ROW.items()},
                },
            )
        entries[f"prologue_vllm_p{requests}"] = CuteEntry(
            "prologue", "PrologueVllm", (("ROWS", requests), ("REQUESTS", requests))
        )
    return entries


ENTRIES.update(_draft_tier_entries())


def kernel_class(name: str) -> type:
    """Return entry ``name``'s kernel class."""
    entry = ENTRIES[name]
    return getattr(importlib.import_module(f"{__package__}.{entry.module}"), entry.cls)


def kernel(name: str, arch: str = "sm_100a") -> object:
    """Return entry ``name``'s configured kernel object (an ``rt.DecodeKernel``)."""
    cls = kernel_class(name)
    knobs = dict(ENTRIES[name].constants)
    if hasattr(cls, "EXPLICIT_SMEM_BASE"):
        knobs["EXPLICIT_SMEM_BASE"] = arch == "sm_100a"
    return cls(**knobs)


def _arguments(cls: type) -> list[tuple[str, dict, type]]:
    records = getattr(cls, "RECORDS", None) or (cls.Params,)  # type: ignore[attr-defined]
    fields = [
        (f.name, dict(f.metadata), f.type)
        for r in records
        for f in dataclasses.fields(r)
    ]
    return fields + [(n, {}, t) for n, t in cls.HOST_ARGS]  # type: ignore[attr-defined, return-value]


def arguments(name: str) -> tuple[str, ...]:
    """Return entry ``name``'s host-wrapper argument names, in order (the stream
    follows).
    """
    return tuple(n for n, _, _ in _arguments(kernel_class(name)))


def bind(name: str, values: Sequence[int | None]) -> tuple:
    """Return the host wrapper's typed arguments for ``values`` (addresses, None for
    null, scalars).
    """
    import cutlass
    import cutlass.cute as cute

    specs = _arguments(kernel_class(name))
    if len(values) != len(specs):
        raise ValueError(
            f"{name}: {len(values)} arguments for {[n for n, _, _ in specs]}"
        )
    bound = []
    for (_, meta, type_), value in zip(specs, values):
        if "dtype" in meta:
            dtype = getattr(cutlass, ELEMENT_TYPES[meta["dtype"]])
            gmem = cute.AddressSpace.gmem
            bound.append(
                cute.runtime.make_ptr(
                    dtype, value or 0, gmem, assumed_align=meta["align"]
                )
            )
        else:
            bound.append(type_(value))
    return tuple(bound)


def example_arguments(name: str) -> tuple:
    """Return typed host-wrapper arguments to compile entry ``name`` with (the stream
    follows).
    """
    specs = _arguments(kernel_class(name))
    return bind(
        name, [EXAMPLE_ADDRESS if "dtype" in meta else 0 for _, meta, _ in specs]
    )


@functools.cache
def _source_digest() -> str:
    digest = hashlib.sha256(importlib.metadata.version("nvidia-cutlass-dsl").encode())
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


@functools.cache
def compiled(name: str, arch: str) -> JitCompiledFunction:
    """Return entry ``name``'s host wrapper for ``arch``: from the object cache, else
    compiled and cached.
    """
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.runtime import load_module

    import vllm.envs as envs

    constants = repr(ENTRIES[name].constants)
    key = f"{name}/{arch}/{constants}/{_source_digest()}"
    key = hashlib.sha256(key.encode()).hexdigest()[:32]
    path = Path(envs.VLLM_CACHE_ROOT) / "glm53_kernels" / f"{name}.{arch}.{key}.o"
    if path.is_file():
        return getattr(load_module(str(path)), name)
    start = time.perf_counter()
    function = cute.compile(
        kernel(name, arch).launch,  # type: ignore[attr-defined]
        *example_arguments(name),
        CUstream(0),
        options=f"--gpu-arch {arch}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(function.dump_to_object(name))
    temporary.replace(path)
    logger.info(
        "GLM-5.3: compiled CuTe kernel %s for %s in %.2f s.",
        name,
        arch,
        time.perf_counter() - start,
    )
    return function


@functools.cache
def executor(name: str, device_index: int) -> JitExecutor:
    """Return entry ``name`` compiled for device ``device_index`` and bound to it; call
    it with :func:`bind`'s arguments and a ``CUstream`` (capturable into a CUDA
    graph).
    """
    major, minor = torch.cuda.get_device_capability(device_index)
    return compiled(name, f"sm_{major}{minor}a").to(device_index)
