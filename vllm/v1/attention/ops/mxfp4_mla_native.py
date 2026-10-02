# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launcher for the native FP8 x FP4 MXFP4 MLA kernel.

The kernel is HIP rather than Triton for prefill-size speed: the Triton port
(mxfp4_mla_triton) issues the same v_mfma_scale_f32_16x16x128_f8f6f4 and ties
this kernel at decode sizes, but is ~2x slower from 512 rows up. This tree
ships Python only, so the code object is built with ``hipcc --genco`` on first
use, cached by source hash, and launched through ``hipModuleLaunchKernel`` on
torch's current stream, which also makes it capturable in a CUDA graph.

Numerics: Q is quantized to E4M3 with a per-(token, head) absmax scale, and
the softmax weights to E4M3 after folding V's E8M0 scales in.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import pathlib
import subprocess

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HEADS = 16
LATENT = 512
ROW_BYTES = 272
TOK_PER_SPLIT = 128  # tokens per chunk: the PV MFMA's k
E4M3_MAX = 448.0
# Workgroups to aim for before splitting a row further. At decode sizes each
# row is split into one chunk per workgroup for parallelism; at prefill sizes
# splitting only adds partial traffic (16 fp32 partials per row outweighed the
# KV read at 8192 rows), so rows keep all their chunks in one workgroup.
TARGET_WORKGROUPS = 2048
# Row-splits per launch; each needs a 32 KB fp32 partial, so this caps the
# workspace at 128 MB and large batches are processed in slices.
MAX_ROW_SPLITS_PER_LAUNCH = 4096


def choose_splits(nq: int, chunks: int) -> int:
    """One chunk per split, halved while the grid keeps TARGET_WORKGROUPS."""
    splits = chunks
    while splits > 1 and nq * (splits // 2) >= TARGET_WORKGROUPS:
        splits //= 2
    return splits


_SRC = pathlib.Path(__file__).with_suffix(".hip")
_CACHE = pathlib.Path(os.getenv("GLM53_NATIVE_CACHE", "/tmp/glm53_native_kernels"))
_hip = None

# Persistent split partials and quantised Q, one set per device, sized for
# MAX_ROW_SPLITS_PER_LAUNCH. Allocated on the first call, which vLLM makes in its
# warmup/profile pass before any CUDA-graph capture, and reused across layers
# since they run in order on one stream.
_workspace: dict[int, tuple[torch.Tensor, ...]] = {}


def _hip_lib():
    global _hip
    if _hip is None:
        for name in ("libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"):
            try:
                _hip = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _hip is None:
            raise RuntimeError("libamdhip64.so not found")
    return _hip


def _check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"{what} failed with hipError {err}")


def _code_object() -> pathlib.Path:
    src = _SRC.read_bytes()
    out = _CACHE / f"mxfp4_mla_native_{hashlib.sha256(src).hexdigest()[:16]}.hsaco"
    if out.exists():
        return out
    _CACHE.mkdir(parents=True, exist_ok=True)
    # TP workers start together; one compiles, the rest wait on the lock.
    with open(_CACHE / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not out.exists():
            from torch.utils.cpp_extension import ROCM_HOME

            tmp = out.with_suffix(f".tmp{os.getpid()}")
            hipcc = os.path.join(ROCM_HOME or "/opt/rocm", "bin", "hipcc")
            subprocess.run(
                [
                    hipcc,
                    "--genco",
                    "--offload-arch=gfx950",
                    "-O3",
                    str(_SRC),
                    "-o",
                    str(tmp),
                ],
                check=True,
                capture_output=True,
            )
            tmp.rename(out)
            logger.info("built native MXFP4 MLA kernel: %s", out)
    return out


_modules: dict[int, ctypes.c_void_p] = {}
_fn_cache: dict[tuple[int, str], ctypes.c_void_p] = {}


def _function(device: int, name: str) -> ctypes.c_void_p:
    if (device, name) not in _fn_cache:
        hip = _hip_lib()
        if device not in _modules:
            _check(hip.hipSetDevice(ctypes.c_int(device)), "hipSetDevice")
            module = ctypes.c_void_p()
            _check(
                hip.hipModuleLoad(ctypes.byref(module), str(_code_object()).encode()),
                "hipModuleLoad",
            )
            _modules[device] = module
        f = ctypes.c_void_p()
        _check(
            hip.hipModuleGetFunction(ctypes.byref(f), _modules[device], name.encode()),
            f"hipModuleGetFunction({name})",
        )
        _fn_cache[(device, name)] = f
    return _fn_cache[(device, name)]


def _quantize_q_into(
    q: torch.Tensor, q8: torch.Tensor, qs: torch.Tensor, stream: int
) -> None:
    _launch(
        _function(q.device.index, "mxfp4_mla_native_quantize_q"),
        (q.shape[0], 1),
        64,
        [
            q.data_ptr(),
            ctypes.c_int64(q.stride(0)),
            ctypes.c_int64(q.stride(1)),
            q8.data_ptr(),
            qs.data_ptr(),
        ],
        stream,
    )


def device_quantize_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the kernel's Q quantisation and return (q8, qscale), for tests."""
    q8 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
    qs = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    _quantize_q_into(q, q8, qs, torch.cuda.current_stream(q.device).cuda_stream)
    return q8, qs


def _launch(fn, grid, block, args, stream: int) -> None:
    holders = [
        a
        if isinstance(
            a, (ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int64)
        )
        else ctypes.c_void_p(a)
        for a in args
    ]
    params = (ctypes.c_void_p * len(holders))(
        *[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p) for h in holders]
    )
    _check(
        _hip_lib().hipModuleLaunchKernel(
            fn,
            grid[0],
            grid[1],
            1,
            block,
            1,
            1,
            0,
            ctypes.c_void_p(stream),
            params,
            None,
        ),
        "hipModuleLaunchKernel",
    )


def quantize_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 query with a per-(token, head) absmax scale.

    The kernel does this itself; this is the reference definition it must match.
    """
    x = q.float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
    q8 = (x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q8.view(torch.uint8).contiguous(), scale.squeeze(-1).contiguous()


def supported(q: torch.Tensor, kv: torch.Tensor) -> bool:
    return (
        q.dim() == 3
        and q.shape[1] == HEADS
        and q.shape[2] == LATENT
        and kv.dtype == torch.uint8
        and kv.shape[-1] == ROW_BYTES
    )


def mxfp4_mla_native(
    q: torch.Tensor,  # [nq, 16, 512] bf16
    kv: torch.Tensor,  # [num_slots, 272] uint8
    indices: torch.Tensor,  # ragged int32
    indptr: torch.Tensor,  # [nq + 1] int32
    sm_scale: float,
    max_topk: int,
    out: torch.Tensor,  # [nq, 16, 512] bf16, unit stride on the last dim
    num_splits: int | None = None,  # None: choose_splits; tests force it
) -> torch.Tensor:
    assert supported(q, kv) and out.stride(-1) == 1 and q.stride(-1) == 1
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    kv = kv.reshape(-1, ROW_BYTES)
    if indices.dtype != torch.int32 or not indices.is_contiguous():
        indices = indices.to(torch.int32).contiguous()
    if indptr.dtype != torch.int32 or not indptr.is_contiguous():
        indptr = indptr.to(torch.int32).contiguous()
    # The split count depends only on the row count and max_topk, never on the
    # data, so the grid is fixed for a captured batch size; splits beyond a
    # row's real length exit immediately.
    nq = q.shape[0]
    chunks = (max_topk + TOK_PER_SPLIT - 1) // TOK_PER_SPLIT
    splits = num_splits or choose_splits(nq, chunks)
    chunks_per_split = (chunks + splits - 1) // splits
    rows_per_launch = max(1, MAX_ROW_SPLITS_PER_LAUNCH // splits)
    dev = q.device.index
    f_split = _function(dev, "mxfp4_mla_native_split")
    f_reduce = _function(dev, "mxfp4_mla_native_reduce")
    stream = torch.cuda.current_stream(q.device).cuda_stream
    if dev not in _workspace:
        rs = MAX_ROW_SPLITS_PER_LAUNCH
        _workspace[dev] = (
            torch.empty(rs * HEADS * LATENT, dtype=torch.float32, device=q.device),
            torch.empty(rs * 2 * HEADS, dtype=torch.float32, device=q.device),
            torch.empty(rs, HEADS, LATENT, dtype=torch.uint8, device=q.device),
            torch.empty(rs, HEADS, dtype=torch.float32, device=q.device),
        )
    part_acc, part_ml, q8, qscale = _workspace[dev]
    for r0 in range(0, nq, rows_per_launch):
        n = min(rows_per_launch, nq - r0)
        assert n * splits <= MAX_ROW_SPLITS_PER_LAUNCH, "workspace smaller than a slice"
        sub_out = out[r0 : r0 + n]
        _quantize_q_into(q[r0 : r0 + n], q8, qscale, stream)
        _launch(
            f_split,
            (n, splits),
            256,
            [
                q8.data_ptr(),
                qscale.data_ptr(),
                kv.data_ptr(),
                indices.data_ptr(),
                indptr[r0:].data_ptr(),
                part_acc.data_ptr(),
                part_ml.data_ptr(),
                ctypes.c_int(kv.shape[0]),
                ctypes.c_int(splits),
                ctypes.c_int(chunks_per_split),
                ctypes.c_float(sm_scale),
            ],
            stream,
        )
        _launch(
            f_reduce,
            (n, HEADS),
            128,
            [
                part_acc.data_ptr(),
                part_ml.data_ptr(),
                sub_out.data_ptr(),
                ctypes.c_int(splits),
                ctypes.c_int64(sub_out.stride(0)),
                ctypes.c_int64(sub_out.stride(1)),
            ],
            stream,
        )
    return out
