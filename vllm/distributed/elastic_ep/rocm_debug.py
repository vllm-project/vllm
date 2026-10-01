# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV_PREFIXES = (
    "NCCL_",
    "RCCL_",
    "HSA_",
    "HIP_",
    "AMD_",
    "PYTORCH_HIP",
    "PYTORCH_CUDA",
    "TORCH_NCCL",
    "ROCR_",
    "GPU_",
)

_MALLOC_FLAGS = {"default": 0x0, "finegrained": 0x1, "uncached": 0x3}

_hip = None


def _lib():
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipGetErrorString.restype = ctypes.c_char_p
    return _hip


def _err(code: int) -> str:
    return f"{code} ({_lib().hipGetErrorString(code).decode()})"


def _ipc_probe(size: int) -> dict[str, str]:
    hip = _lib()
    results = {}
    for name, flags in _MALLOC_FLAGS.items():
        ptr = ctypes.c_void_p()
        rc = hip.hipExtMallocWithFlags(
            ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(flags)
        )
        if rc != 0:
            results[name] = f"malloc failed {_err(rc)}"
            hip.hipGetLastError()
            continue
        handle = ctypes.create_string_buffer(64)
        rc = hip.hipIpcGetMemHandle(handle, ptr)
        results[name] = "ok" if rc == 0 else f"ipc failed {_err(rc)}"
        hip.hipGetLastError()
        hip.hipFree(ptr)
    try:
        tensor = torch.empty(size, dtype=torch.uint8, device="cuda")
        tensor.untyped_storage()._share_cuda_()
        results["torch_allocator"] = "ok"
    except Exception as e:
        results["torch_allocator"] = f"failed {type(e).__name__}: {e}"
    return results


def dump(tag: str, groups: dict | None = None) -> None:
    try:
        device = torch.accelerator.current_device_index()
        free, total = torch.accelerator.get_memory_info(device)
        get_settings = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
        info = {
            "pid": os.getpid(),
            "device": device,
            "free_gib": round(free / 2**30, 2),
            "total_gib": round(total / 2**30, 2),
            "allocated_gib": round(
                torch.accelerator.memory_allocated(device) / 2**30, 2
            ),
            "reserved_gib": round(torch.accelerator.memory_reserved(device) / 2**30, 2),
            "allocator_backend": torch.cuda.memory.get_allocator_backend(),
            "allocator_settings": get_settings() if get_settings else None,
        }
        if groups:
            info["groups"] = {
                name: {
                    "rank": g.rank_in_group,
                    "world_size": g.world_size,
                    "ranks": list(getattr(g, "ranks", [])),
                }
                for name, g in groups.items()
                if g is not None
            }
        info["ipc_probe_6MiB"] = _ipc_probe(6 << 20)
        logger.warning("[EEP-DEBUG] %s %s", tag, info)
    except Exception as e:
        logger.warning("[EEP-DEBUG] %s dump failed: %r", tag, e)


def dump_static() -> None:
    try:
        env = {
            k: v for k, v in sorted(os.environ.items()) if k.startswith(_ENV_PREFIXES)
        }
        logger.warning(
            "[EEP-DEBUG] static torch=%s hip=%s rccl=%s env=%s",
            torch.__version__,
            torch.version.hip,
            torch.cuda.nccl.version(),
            env,
        )
    except Exception as e:
        logger.warning("[EEP-DEBUG] static dump failed: %r", e)


def warm_each(groups: dict, tag: str) -> None:
    stream = torch.Stream(device=next(iter(groups.values())).device)
    with stream:
        tensor = torch.zeros(1, dtype=torch.int32, device=stream.device)
        for name, group in groups.items():
            try:
                torch.distributed.all_reduce(tensor, group=group.device_group)
                stream.synchronize()
                logger.warning("[EEP-DEBUG] %s warm %s ok", tag, name)
            except Exception as e:
                logger.warning(
                    "[EEP-DEBUG] %s warm %s FAILED rank=%s world_size=%s: %r",
                    tag,
                    name,
                    group.rank_in_group,
                    group.world_size,
                    e,
                )
                dump(f"{tag} after {name} failure", groups)
                raise
