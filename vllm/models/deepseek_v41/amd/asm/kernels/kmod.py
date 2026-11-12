# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Build, load and launch a generated gfx950 kernel. One `KArgs` spec gives
the asm kernarg offsets, the code-object metadata and the host packing, so the
two sides can't disagree."""

import collections
import ctypes
import os
import struct
from concurrent.futures import Future, ThreadPoolExecutor

RSRC3 = 0x00020000  # buffer resource dword 3: gfx9 raw dword format


class KArgs:
    """Ordered kernargs: ("name", "ptr") 8 B or ("name", "u32") 4 B, natural
    alignment, in declaration order."""

    def __init__(self, *args: tuple[str, str]):
        self.args, self.off, o = list(args), {}, 0
        for name, kind in args:
            size = 8 if kind == "ptr" else 4
            o = (o + size - 1) // size * size
            self.off[name] = o
            o += size
        self.size = (o + 7) // 8 * 8

    def pack(self, **vals) -> bytes:
        buf = bytearray(self.size)
        for name, kind in self.args:
            v = vals[name]
            if kind == "ptr":
                struct.pack_into(
                    "<Q",
                    buf,
                    self.off[name],
                    v.data_ptr() if hasattr(v, "data_ptr") else int(v),
                )
            else:
                struct.pack_into("<I", buf, self.off[name], int(v) & 0xFFFFFFFF)
        return bytes(buf)

    def metadata(self) -> str:
        out = []
        for name, kind in self.args:
            if kind == "ptr":
                out.append(
                    f"      - {{ .name: {name}, .size: 8, .offset: {self.off[name]}, "
                    ".value_kind: global_buffer, .address_space: global }"
                )
            else:
                out.append(
                    f"      - {{ .name: {name}, .size: 4, .offset: {self.off[name]}, .value_kind: by_value }}"
                )
        return "\n".join(out)


def kernel_source(
    name: str,
    body: list[str],
    ka: KArgs,
    *,
    lds: int,
    vgprs: int,
    agprs: int,
    sgprs: int,
    block: int,
) -> str:
    acc_off = (vgprs + 3) // 4 * 4
    total = acc_off + agprs
    assert vgprs <= 256 and agprs <= 256 and total <= 512, (vgprs, agprs)
    return f""".amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl {name}
.p2align 8
.type {name},@function
{name}:
{chr(10).join(body)}
.Lfunc_end:
.size {name}, .Lfunc_end-{name}

.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_group_segment_fixed_size {lds}
  .amdhsa_kernarg_size {ka.size}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr {total}
  .amdhsa_accum_offset {acc_off}
  .amdhsa_next_free_sgpr {sgprs}
  .amdhsa_reserve_vcc 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: {ka.size}
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds}
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: {sgprs}
    .vgpr_count: {total}
    .agpr_count: {agprs}
    .max_flat_workgroup_size: {block}
    .args:
{ka.metadata()}
...
.end_amdgpu_metadata
"""


_builds: dict = {"pool": None, "pending": []}


def wait_builds() -> None:
    """Loads every kernel whose build is still running (call outside graph
    capture; a failed build raises here)."""
    while _builds["pending"]:
        _builds["pending"].pop()._load()


class Kernel:
    """A generated kernel: assembled and linked on a thread pool (clang and
    ld.lld subprocesses, so builds run in parallel), loaded on the first launch
    or at `wait_builds`."""

    def __init__(self, name: str, src: str, ka: KArgs):
        from vllm.models.deepseek_v41.amd.asm.kernels.build import build

        if _builds["pool"] is None:
            _builds["pool"] = ThreadPoolExecutor(min(16, os.cpu_count() or 1))
        self.name, self.ka = name, ka
        self._img_future: Future = _builds["pool"].submit(build, src, name)
        self.fn: ctypes.c_void_p | None = None
        _builds["pending"].append(self)
        # recent launches' kernarg memory (bounded: a server launches forever)
        self._keep: collections.deque = collections.deque(maxlen=4096)

    def _load(self) -> None:
        from vllm.models.deepseek_v41.amd.asm.kernels.build import _check, hip

        if self.fn is not None:
            return
        self._check, self.h = _check, hip()
        img = self._img_future.result()
        self._img = ctypes.create_string_buffer(img, len(img))
        self.mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        _check(
            self.h.hipModuleLoadData(ctypes.byref(self.mod), self._img),
            "hipModuleLoadData",
        )
        _check(
            self.h.hipModuleGetFunction(ctypes.byref(fn), self.mod, self.name.encode()),
            "hipModuleGetFunction",
        )
        self.fn = fn

    def __call__(self, grid: int, block: int, **vals) -> None:
        """KernelParams launch: an array of pointers to each argument value; HIP copies them using the code
        object's argument metadata (at launch, or at capture for a graph node)."""
        import torch

        if self.fn is None:
            self._load()
        cells = []
        for name, kind in self.ka.args:
            v = vals[name]
            if kind == "ptr":
                cells.append(
                    ctypes.c_uint64(v.data_ptr() if hasattr(v, "data_ptr") else int(v))
                )
            else:
                cells.append(ctypes.c_uint32(int(v) & 0xFFFFFFFF))
        params = (ctypes.c_void_p * len(cells))(
            *[ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in cells]
        )
        self._keep.append((cells, params))
        self._check(
            self.h.hipModuleLaunchKernel(
                self.fn,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                torch.cuda.current_stream().cuda_stream,
                params,
                None,
            ),
            "hipModuleLaunchKernel",
        )


def rsrc(
    sbase: int, ptr_sgpr: int, nbytes_sgpr: int | None = None, nbytes: int | None = None
) -> list[str]:
    """Buffer resource in s[sbase:sbase+3] from a 64-bit pointer in
    s[ptr_sgpr:ptr_sgpr+1] and a byte size (SGPR or constant)."""
    n = f"s{nbytes_sgpr}" if nbytes_sgpr is not None else str(nbytes)
    return [
        f"  s_mov_b32 s{sbase}, s{ptr_sgpr}",
        f"  s_and_b32 s{sbase + 1}, s{ptr_sgpr + 1}, 0xffff",
        f"  s_mov_b32 s{sbase + 2}, {n}",
        f"  s_mov_b32 s{sbase + 3}, {RSRC3:#x}",
    ]
