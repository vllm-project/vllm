# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes loader for the transitional Qwen3 RDNA dense backend."""

from __future__ import annotations

import ctypes
import functools
import os

import torch

from vllm.logger import init_logger

log = init_logger(__name__)


class _Layer(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "q",
            "k",
            "v",
            "o",
            "gate",
            "up",
            "down",
            "attn_norm",
            "ffn_norm",
            "q_norm",
            "k_norm",
        )
    ]


def _bind_api(lib):
    p, i, ip = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    lib.mk_vllm_dense_create.restype = p
    lib.mk_vllm_dense_create.argtypes = [
        i,
        ctypes.POINTER(_Layer),
        p,
        p,
        p,
        i,
        ctypes.c_char_p,
        i,
    ]
    lib.mk_vllm_dense_bind.restype = i
    lib.mk_vllm_dense_bind.argtypes = [p, ctypes.POINTER(p), ctypes.POINTER(p), i]
    lib.mk_vllm_dense_launch.restype = i
    lib.mk_vllm_dense_launch.argtypes = [p, p, p, p, p, p]
    lib.mk_vllm_dense_status.restype = i
    lib.mk_vllm_dense_status.argtypes = [p]
    lib.mk_vllm_dense_max_ctx.restype = i
    lib.mk_vllm_dense_free.argtypes = [p]
    lib.mk_vllm_dense_shape.argtypes = [ip] * 6
    return lib


def _lib_dirs() -> list[str]:
    dirs: list[str] = []
    for env in ("VLLM_QWEN3_RDNA_DENSE_LIB_DIR", "RDNA_MK_LIB_DIR"):
        path = os.environ.get(env)
        if path:
            dirs.append(path)
    dirs.append(os.path.join(os.path.dirname(__file__), "lib"))
    return dirs


@functools.cache
def arch() -> str:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(
        ":"
    )[0]


@functools.cache
def load(model: str):
    for lib_dir in _lib_dirs():
        path = os.path.join(lib_dir, f"libmk_{model}_vllm_{arch()}.so")
        if os.path.exists(path):
            return _bind_api(ctypes.CDLL(path))
    tried = [os.path.join(d, f"libmk_{model}_vllm_{arch()}.so") for d in _lib_dirs()]
    log.warning("Qwen3 RDNA dense library not found; tried %s", tried)
    return None


def lib_shape(lib) -> dict[str, int]:
    values = [ctypes.c_int() for _ in range(6)]
    lib.mk_vllm_dense_shape(*[ctypes.byref(x) for x in values])
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "qk_norm",
    )
    return dict(zip(keys, (x.value for x in values)))


def kv_planes(kv: torch.Tensor, kv_heads: int, head_dim: int) -> tuple[int, int, int] | None:
    if not isinstance(kv, torch.Tensor) or kv.dtype != torch.bfloat16:
        return None
    row = kv_heads * head_dim
    if kv.dim() == 5 and kv.shape[0] == 2 and tuple(kv.shape[3:]) == (kv_heads, head_dim):
        kv_dim, blk_dim, block_size = 0, 1, kv.shape[2]
        inner_ok = (
            kv.stride(4) == 1
            and kv.stride(3) == head_dim
            and kv.stride(2) == row
        )
    elif kv.dim() == 4 and kv.shape[1] == 2 and kv.shape[3] == row:
        kv_dim, blk_dim, block_size = 1, 0, kv.shape[2]
        inner_ok = kv.stride(3) == 1 and kv.stride(2) == row
    else:
        return None
    if not inner_ok or kv.stride(blk_dim) != block_size * row:
        return None
    element_size = kv.element_size()
    return (
        kv.data_ptr(),
        kv.data_ptr() + kv.stride(kv_dim) * element_size,
        block_size,
    )
