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
_WORKER = {
    "kFlagWords": 15804,
    "kHandoffWords": 59392,
    "kPartialOBytes": 2097152,
    "kPartialLseBytes": 6144,
}


def _sel(shard: int, packed: bool = False, cls: str = "SelectorFull") -> CuteEntry:
    """The selector ``cls`` of key shard ``shard``; ``packed``: for an index-key cache
    with packed UE8M0 scales (``_mx`` entries).
    """
    constants = (("SHARD", shard),) + ((("PACKED_SCALE", True),) if packed else ())
    return CuteEntry("mla_selector", cls, constants, _SELECTOR)


def _worker(cls: str, **host: int) -> CuteEntry:
    return CuteEntry("mla_worker", cls, host=_WORKER | host)


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
    "mla_worker_q4_full_paged_vlin": _worker("FullWorker"),
    "mla_worker_q4_full_push_paged_vlin": _worker("PushWorker", kMailboxBytes=1009792),
    "mla_worker_q4_full_pushed_paged_vlin": _worker("PushedWorker"),
    "mla_worker_q4_shared_paged_vlin": _worker("SharedWorker", kMailboxBytes=917632),
    "mla_worker_q4_shared_seam_paged_vlin": _worker("SeamWorker", kMailboxBytes=917632),
    "prologue_q4": CuteEntry("prologue", "PrologueQ4"),
    "prologue_vllm_q4": CuteEntry("prologue", "PrologueVllm", (("ROWS", 4),)),
    "mla_selector_q4_shard3r0_paged_vlin": _sel(0),
    "mla_selector_q4_shard3r1_paged_vlin": _sel(1),
    "mla_selector_q4_shard3r2_paged_vlin": _sel(2),
    "mla_selector_q4_shard3r0_paged_vlin_mx": _sel(0, True),
    "mla_selector_q4_shard3r1_paged_vlin_mx": _sel(1, True),
    "mla_selector_q4_shard3r2_paged_vlin_mx": _sel(2, True),
}


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
    "mla_worker_draft_q1_paged": CuteEntry(
        "mla_draft",
        "DraftWorker",
        (("ROWS", 1),),
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
        f"mla_selector_q4_shard3r{shard}_only_row_paged{suffix}": _sel(
            shard, bool(suffix), "SelectorOnlyRow"
        )
        for shard in range(3)
        for suffix in ("", "_mx")
    },
}
ENTRIES.update(DRAFT_ENTRIES)


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
    records = getattr(cls, "RECORDS", None) or (cls.Params,)
    fields = [
        (f.name, dict(f.metadata), f.type)
        for r in records
        for f in dataclasses.fields(r)
    ]
    return fields + [(n, {}, t) for n, t in cls.HOST_ARGS]


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

    key = hashlib.sha256(f"{name}/{arch}/{_source_digest()}".encode()).hexdigest()[:32]
    path = Path(envs.VLLM_CACHE_ROOT) / "glm53_kernels" / f"{name}.{arch}.{key}.o"
    if path.is_file():
        return getattr(load_module(str(path)), name)
    start = time.perf_counter()
    function = cute.compile(
        kernel(name, arch).launch,
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
