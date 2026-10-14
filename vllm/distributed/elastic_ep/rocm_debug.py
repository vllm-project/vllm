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

_HANDLE_BYTES = 64
_IPC_LAZY_ENABLE_PEER_ACCESS = 0x1


class _HipIpcMemHandle(ctypes.Structure):
    """c_ubyte, not c_char: ctypes converts a c_char array to bytes and
    truncates it at the first NUL, which silently corrupts the handle."""

    _fields_ = [("reserved", ctypes.c_ubyte * _HANDLE_BYTES)]


_hip = None


def _lib():
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipGetErrorString.restype = ctypes.c_char_p
        # hipIpcOpenMemHandle takes the handle by value, not by pointer.
        _hip.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _HipIpcMemHandle,
            ctypes.c_uint,
        ]
        _hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    return _hip


def _err(code: int) -> str:
    return f"{code} ({_lib().hipGetErrorString(code).decode()})"


def _hip_device() -> int:
    """HIP's own notion of the current device, which torch does not always set."""
    dev = ctypes.c_int(-1)
    _lib().hipGetDevice(ctypes.byref(dev))
    return dev.value


def _hip_device_count() -> int:
    count = ctypes.c_int(0)
    _lib().hipGetDeviceCount(ctypes.byref(count))
    return count.value


def _device_fds() -> dict[str, int]:
    """IPC export needs the device node open; count what this process holds."""
    counts = {"kfd": 0, "render": 0}
    try:
        for fd in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
            if target == "/dev/kfd":
                counts["kfd"] += 1
            elif target.startswith("/dev/dri/render"):
                counts["render"] += 1
    except OSError:
        pass
    return counts


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


def _ipc_probe_plain(size: int) -> str:
    """RCCL exports handles for plain hipMalloc buffers, which the
    hipExtMallocWithFlags probe above never exercises."""
    hip = _lib()
    ptr = ctypes.c_void_p()
    rc = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if rc != 0:
        hip.hipGetLastError()
        return f"malloc failed {_err(rc)}"
    handle = ctypes.create_string_buffer(64)
    rc = hip.hipIpcGetMemHandle(handle, ptr)
    out = "ok" if rc == 0 else f"ipc failed {_err(rc)}"
    hip.hipGetLastError()
    hip.hipFree(ptr)
    return out


def _ipc_probe_burst(size: int, count: int) -> str:
    """Exports many handles at once without freeing, to see whether the export
    path has a per-process limit. Everything is released before returning."""
    hip = _lib()
    ptrs: list[ctypes.c_void_p] = []
    outcome = f"all {count} ok"
    try:
        for i in range(count):
            ptr = ctypes.c_void_p()
            rc = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
            if rc != 0:
                hip.hipGetLastError()
                outcome = f"malloc failed at {i}: {_err(rc)}"
                break
            ptrs.append(ptr)
            handle = ctypes.create_string_buffer(64)
            rc = hip.hipIpcGetMemHandle(handle, ptr)
            if rc != 0:
                hip.hipGetLastError()
                outcome = f"ipc failed at {i}: {_err(rc)}"
                break
    finally:
        for ptr in ptrs:
            hip.hipFree(ptr)
        hip.hipGetLastError()
    return outcome


def _ipc_probe_foreign_device(size: int) -> str:
    """Allocates on the current device, then exports while a different device is
    current. This is the mismatch RCCL would hit if it exports a peer buffer
    without setting the device first."""
    hip = _lib()
    count = _hip_device_count()
    if count < 2:
        return "skipped: single device"
    home = _hip_device()
    ptr = ctypes.c_void_p()
    rc = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if rc != 0:
        hip.hipGetLastError()
        return f"malloc failed {_err(rc)}"
    other = (home + 1) % count
    try:
        rc = hip.hipSetDevice(ctypes.c_int(other))
        if rc != 0:
            hip.hipGetLastError()
            return f"setDevice({other}) failed {_err(rc)}"
        handle = ctypes.create_string_buffer(64)
        rc = hip.hipIpcGetMemHandle(handle, ptr)
        out = f"home={home} exported_from={other} " + (
            "ok" if rc == 0 else f"ipc failed {_err(rc)}"
        )
        hip.hipGetLastError()
    finally:
        hip.hipSetDevice(ctypes.c_int(home))
        hip.hipFree(ptr)
        hip.hipGetLastError()
    return out


def _ipc_mesh_probe(group, size: int, flags: int) -> str:
    """Exports one buffer per rank and attaches every peer's. Handles travel
    over the CPU group, so the device group under test stays untouched.

    The attach side is the one that aborted in CI when
    HSA_ENABLE_IPC_MODE_LEGACY was set, and nothing probed it until now.
    """
    hip = _lib()
    ptr = ctypes.c_void_p()
    rc = hip.hipExtMallocWithFlags(
        ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(flags)
    )
    if rc != 0:
        hip.hipGetLastError()
        return f"malloc failed {_err(rc)}"

    handle = _HipIpcMemHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    hip.hipGetLastError()
    mine = bytes(handle.reserved) if rc == 0 else None
    export = "ok" if rc == 0 else f"failed {_err(rc)}"

    peers: list = [None] * group.world_size
    try:
        torch.distributed.all_gather_object(peers, mine, group=group.cpu_group)
    except Exception as e:
        hip.hipFree(ptr)
        hip.hipGetLastError()
        return f"export {export}, exchange failed {type(e).__name__}: {e}"

    problems = []
    for peer, raw in enumerate(peers):
        if peer == group.rank_in_group:
            continue
        if raw is None:
            problems.append(f"rank{peer} exported nothing")
            continue
        opened = ctypes.c_void_p()
        theirs = _HipIpcMemHandle()
        ctypes.memmove(ctypes.byref(theirs), raw, _HANDLE_BYTES)
        rc = hip.hipIpcOpenMemHandle(
            ctypes.byref(opened), theirs, _IPC_LAZY_ENABLE_PEER_ACCESS
        )
        hip.hipGetLastError()
        if rc != 0:
            problems.append(f"rank{peer} {_err(rc)}")
            continue
        hip.hipIpcCloseMemHandle(opened)
        hip.hipGetLastError()

    hip.hipFree(ptr)
    hip.hipGetLastError()
    attach = "ok" if not problems else "; ".join(problems)
    return f"export {export}, attach {attach}"


def dump(tag: str, groups: dict | None = None, heavy: bool = False) -> None:
    """Heavy adds probes that perturb state, so they stay off at the points
    right before a collective we are trying to observe."""
    try:
        device = torch.accelerator.current_device_index()
        free, total = torch.accelerator.get_memory_info(device)
        get_settings = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
        info = {
            "pid": os.getpid(),
            "device": device,
            "hip_device": _hip_device(),
            "device_fds": _device_fds(),
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
                    "device": str(getattr(g, "device", None)),
                }
                for name, g in groups.items()
                if g is not None
            }
        info["ipc_probe_6MiB"] = _ipc_probe(6 << 20)
        info["ipc_probe_plain_6MiB"] = _ipc_probe_plain(6 << 20)
        if heavy:
            # Burst can exhaust export handles and foreign_device creates a
            # context on a neighbour card, so neither runs near the collective.
            info["ipc_probe_burst_64x6MiB"] = _ipc_probe_burst(6 << 20, 64)
            info["ipc_probe_foreign_device"] = _ipc_probe_foreign_device(6 << 20)
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
                # The stream context does not set HIP's current device, so an
                # export inside RCCL can land on the wrong one.
                logger.warning(
                    "[EEP-DEBUG] %s warm %s ctx hip_device=%s torch_device=%s "
                    "group_device=%s stream_device=%s",
                    tag,
                    name,
                    _hip_device(),
                    torch.accelerator.current_device_index(),
                    getattr(group, "device", None),
                    stream.device,
                )
                # Sizes and flags taken from the two CI failures: 2 MiB with
                # flags 3 aborted on attach, 6 MiB failed on export.
                logger.warning(
                    "[EEP-DEBUG] %s warm %s mesh 2MiB/uncached [%s] 6MiB/default [%s]",
                    tag,
                    name,
                    _ipc_mesh_probe(group, 2 << 20, 0x3),
                    _ipc_mesh_probe(group, 6 << 20, 0x0),
                )
                if os.getenv("VLLM_EEP_DEBUG_SET_DEVICE") == "1":
                    torch.accelerator.set_device_index(group.device.index)
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
                dump(f"{tag} after {name} failure", groups, heavy=True)
                raise
