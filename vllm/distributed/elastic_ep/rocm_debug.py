# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import os
import struct
import subprocess
import sys
import traceback

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


def _exception_chain(exc: BaseException) -> str:
    """While handling a collective failure, torch's c10d logger calls
    dist.get_backend() for its message dict. On an invalidated group that raises
    ValueError, which then replaces the RCCL error everywhere downstream, so the
    CI logs show "Process group <torch..." instead of hipIpcGetMemHandle. Walk
    the chain so the original error survives."""
    parts = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " <- caused by <- ".join(parts)


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
    """IPC export needs the device node open; count what this process holds.

    The total matters too: every live RCCL communicator costs about ten
    descriptors, so a process that keeps resizing groups can approach the limit
    without the device-node counts moving at all.
    """
    counts = {"kfd": 0, "render": 0, "total": 0, "limit": 0}
    try:
        import resource

        counts["limit"] = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:
        pass
    try:
        for fd in os.listdir("/proc/self/fd"):
            counts["total"] += 1
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


class _HipPointerAttributes(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("device", ctypes.c_int),
        ("device_pointer", ctypes.c_void_p),
        ("host_pointer", ctypes.c_void_p),
        ("is_managed", ctypes.c_int),
        ("allocation_flags", ctypes.c_uint),
    ]


_MEMORY_TYPES = {
    0: "unregistered",
    1: "host",
    2: "device",
    3: "managed",
    10: "array",
    11: "unified",
}


def _pointer_facts(ptr: ctypes.c_void_p) -> str:
    """What HIP believes the address is that ROCr just refused.

    ROCr answers the export with HSA_STATUS_ERROR_INVALID_ARGUMENT and the
    argument in question is this pointer, so ask the driver about it directly.
    A type other than device, a device index that is not ours, or a base that
    does not match the pointer would each be an answer rather than a guess.
    """
    hip = _lib()
    facts = []
    attrs = _HipPointerAttributes()
    rc = hip.hipPointerGetAttributes(ctypes.byref(attrs), ptr)
    hip.hipGetLastError()
    if rc != 0:
        facts.append(f"attrs failed {_err(rc)}")
    else:
        facts.append(
            f"type={_MEMORY_TYPES.get(attrs.type, attrs.type)} "
            f"device={attrs.device} managed={attrs.is_managed} "
            f"alloc_flags={attrs.allocation_flags:#x} "
            f"dev_ptr={attrs.device_pointer or 0:#x} "
            f"host_ptr={attrs.host_pointer or 0:#x}"
        )
    base = ctypes.c_void_p()
    psize = ctypes.c_size_t()
    rc = hip.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(psize), ptr)
    hip.hipGetLastError()
    if rc != 0:
        facts.append(f"range failed {_err(rc)}")
    else:
        facts.append(
            f"base={base.value or 0:#x} size={psize.value} "
            f"offset={(ptr.value or 0) - (base.value or 0)}"
        )
    return ", ".join(facts)


_CHILD_EXPORT = r"""
import ctypes, sys

size, device, flags, limit = (int(a) for a in sys.argv[1:5])
hip = ctypes.CDLL("libamdhip64.so")
hip.hipGetErrorString.restype = ctypes.c_char_p


def err(rc):
    return f"{rc} {hip.hipGetErrorString(rc).decode()}"


rc = hip.hipSetDevice(device)
if rc != 0:
    print(f"setDevice failed {err(rc)}")
    raise SystemExit(0)
ptrs = []
held = 0
stopped = f"no ceiling below {limit}"
while held < limit:
    ptr = ctypes.c_void_p()
    rc = hip.hipExtMallocWithFlags(
        ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(flags)
    )
    if rc != 0:
        stopped = f"malloc stopped first: {err(rc)}"
        break
    ptrs.append(ptr)
    handle = ctypes.create_string_buffer(64)
    rc = hip.hipIpcGetMemHandle(handle, ptr)
    if rc != 0:
        stopped = f"export failed: {err(rc)}"
        break
    held += 1
for p in ptrs:
    hip.hipFree(p)
print(f"{held} held, {stopped}")
"""


def _ipc_probe_fresh_process(size: int, limit: int) -> str:
    """Whether the breakage belongs to this process or to the card.

    By the time an export fails here, this process has been through engine
    startup, several scaling transitions and thousands of RCCL exports. A child
    that has done none of that, on the same device, settles the question in one
    line: exporting fine means the state is ours to find, failing the same way
    means the card is in a state no process can use.

    The child walks the same ceiling the parent does rather than doing a single
    export, so the two numbers are directly comparable. Last run the parent was
    down to 0 while the child still managed its one export, which answered the
    question but left open how much the child really had.

    It deliberately does not import torch, so it starts in a fraction of a
    second and costs only a HIP context on the target device. It inherits our
    environment, so HIP_VISIBLE_DEVICES still maps the index to the same card.
    """
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                _CHILD_EXPORT,
                str(size),
                str(_hip_device()),
                str(_MALLOC_FLAGS["uncached"]),
                str(limit),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "child timed out"
    said = out.stdout.strip() or "no output"
    if out.returncode != 0:
        return f"child exited {out.returncode}: {said} {out.stderr.strip()[-300:]}"
    return said


def _ipc_export_mode() -> str:
    """Which of ROCr's two export implementations is actually running.

    Runtime::IPCCreate branches on ipc_dmabuf_supported_, which is simply
    !HSA_ENABLE_IPC_MODE_LEGACY. One side goes through hsaKmtShareMemory, the
    other through hsaKmtExportDMABufHandle, so these are different kernel
    calls, not two spellings of one. The vLLM ROCm images pin the legacy side:

        docker/Dockerfile.rocm:862  ENV HSA_ENABLE_IPC_MODE_LEGACY=1

    Reading the variable back would only tell us what the image set. Decoding
    the handle tells us which branch ran, because the dmabuf side writes the
    pointer into words 0 and 1 and the exporting pid into word 2, while the
    legacy side writes the thunk's own handle there.
    """
    hip = _lib()
    env = os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "unset")
    ptr = ctypes.c_void_p()
    rc = hip.hipExtMallocWithFlags(
        ctypes.byref(ptr), ctypes.c_size_t(6 << 20), ctypes.c_uint(0x3)
    )
    if rc != 0:
        hip.hipGetLastError()
        return f"env={env} malloc failed {_err(rc)}"
    handle = (ctypes.c_ubyte * _HANDLE_BYTES)()
    rc = hip.hipIpcGetMemHandle(handle, ptr)
    if rc != 0:
        hip.hipGetLastError()
        hip.hipFree(ptr)
        return f"env={env} export failed {_err(rc)}"
    words = struct.unpack("<16I", bytes(handle))
    # ctypes types value as int | None, and the allocation succeeded above, so
    # read it once into an int rather than guarding it twice below.
    addr = ptr.value or 0
    hip.hipFree(ptr)
    hip.hipGetLastError()
    dmabuf = (
        words[0] == (addr & 0xFFFFFFFF)
        and words[1] == (addr >> 32)
        and words[2] == os.getpid()
    )
    return f"env={env} effective={'dmabuf' if dmabuf else 'legacy'}"


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
        if rc == 0:
            results[name] = "ok"
        else:
            results[name] = f"ipc failed {_err(rc)} [{_pointer_facts(ptr)}]"
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


def _ipc_export_ceiling(size: int, limit: int) -> str:
    """How many exports fit before the driver refuses one.

    The flags matter and the earlier version of this probe had them wrong. It
    went through plain hipMalloc, while RCCL allocates the buffers it exports
    with hipExtMallocWithFlags and hipDeviceMallocUncached:

        p2p_tmp.cc:284 Cuda Alloc Size 6291456 pointer 0x... flags 3

    so every ceiling measured so far was on a different pool from the one that
    fails. One two-rank group brings up 32 channels and RCCL ends up holding
    roughly a thousand of these per rank, which is the same order as the counts
    this probe walks, so a ceiling here would be a direct explanation.
    """
    hip = _lib()
    ptrs: list[ctypes.c_void_p] = []
    reached = 0
    stopped = f"no ceiling below {limit}"
    try:
        while reached < limit:
            ptr = ctypes.c_void_p()
            rc = hip.hipExtMallocWithFlags(
                ctypes.byref(ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(_MALLOC_FLAGS["uncached"]),
            )
            if rc != 0:
                hip.hipGetLastError()
                stopped = f"malloc stopped first: {_err(rc)}"
                break
            ptrs.append(ptr)
            handle = ctypes.create_string_buffer(_HANDLE_BYTES)
            rc = hip.hipIpcGetMemHandle(handle, ptr)
            if rc != 0:
                hip.hipGetLastError()
                stopped = f"export failed: {_err(rc)}"
                break
            reached += 1
    finally:
        for ptr in ptrs:
            hip.hipFree(ptr)
        hip.hipGetLastError()
    return f"{reached} held, {stopped}"


def _kfd_topology() -> dict:
    """amd-smi on the CI node reports XGMI as N/A for four of six pairs, which
    could mean no link or merely no visibility from inside the VM. The driver's
    own node properties and io_links do not carry that ambiguity."""
    base = "/sys/class/kfd/kfd/topology/nodes"
    wanted = (
        "simd_count",
        "capability",
        "gfx_target_version",
        "location_id",
        "domain",
        "drm_render_minor",
        "num_xcc",
        "hive_id",
    )
    gpus: dict[str, object] = {}
    cpus: list[str] = []
    unreadable: list[str] = []
    try:
        nodes = sorted(os.listdir(base), key=lambda n: int(n) if n.isdigit() else -1)
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}
    for node in nodes:
        props: dict[str, str] = {}
        try:
            with open(f"{base}/{node}/properties") as f:
                for line in f:
                    key, _, value = line.strip().partition(" ")
                    if key in wanted:
                        props[key] = value.strip()
        except OSError as e:
            # The earlier version dropped these silently, which made an absent
            # node indistinguishable from one we simply cannot read. In a VM
            # that difference is the whole question.
            unreadable.append(f"{node} ({type(e).__name__})")
            continue
        if props.get("simd_count", "0") == "0":
            cpus.append(node)
            continue
        links = []
        try:
            for link in sorted(os.listdir(f"{base}/{node}/io_links")):
                fields: dict[str, str] = {}
                with open(f"{base}/{node}/io_links/{link}/properties") as f:
                    for line in f:
                        key, _, value = line.strip().partition(" ")
                        fields[key] = value.strip()
                links.append(f"to={fields.get('node_to')} type={fields.get('type')}")
        except OSError:
            pass
        gpus[node] = {"props": props, "io_links": links}

    # RCCL reads this same table, and in CI it claims XGMI towards nodes that
    # never showed up above. Say so explicitly rather than leaving it to be
    # noticed by eye.
    present = set(nodes)
    for entry in gpus.values():
        missing = sorted(
            {
                link.split("to=", 1)[1].split(" ", 1)[0]
                for link in entry["io_links"]  # type: ignore[index]
            }
            - present
        )
        if missing:
            entry["links_to_absent_nodes"] = missing  # type: ignore[index]
    return {
        "nodes_seen": nodes,
        "cpu_nodes": cpus,
        "unreadable_nodes": unreadable,
        "gpus": gpus,
    }


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


def _ipc_attach_ceiling(group, size: int, flags: int, limit: int) -> str:
    """How many of a peer's exports attach before one is refused.

    The export ceiling says nothing about the import side, and the dmabuf
    implementation breaks there rather than on export, so without this the two
    implementations cannot be compared at the point where each of them actually
    fails. One rank exports and the others attach and hold, which means the
    number the others report can never exceed what that one rank exported, so
    both counts go into the line.
    """
    hip = _lib()
    source = 0
    mine = group.rank_in_group == source
    ptrs: list[ctypes.c_void_p] = []
    raws: list[bytes] = []
    opened: list[ctypes.c_void_p] = []
    stopped = f"no ceiling below {limit}"
    try:
        while mine and len(raws) < limit:
            ptr = ctypes.c_void_p()
            rc = hip.hipExtMallocWithFlags(
                ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(flags)
            )
            if rc != 0:
                hip.hipGetLastError()
                stopped = f"malloc stopped first: {_err(rc)}"
                break
            ptrs.append(ptr)
            handle = _HipIpcMemHandle()
            rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
            hip.hipGetLastError()
            if rc != 0:
                stopped = f"export failed: {_err(rc)}"
                break
            raws.append(bytes(handle.reserved))

        gathered: list = [None] * group.world_size
        try:
            torch.distributed.all_gather_object(
                gathered, raws if mine else None, group=group.cpu_group
            )
        except Exception as e:
            return f"exchange failed {type(e).__name__}: {e}"

        peer = gathered[source] or []
        if mine:
            return f"source, exported {len(raws)}, {stopped}"

        for raw in peer:
            theirs = _HipIpcMemHandle()
            ctypes.memmove(ctypes.byref(theirs), raw, _HANDLE_BYTES)
            got = ctypes.c_void_p()
            rc = hip.hipIpcOpenMemHandle(
                ctypes.byref(got), theirs, _IPC_LAZY_ENABLE_PEER_ACCESS
            )
            hip.hipGetLastError()
            if rc != 0:
                stopped = f"attach failed: {_err(rc)}"
                break
            opened.append(got)
        return f"attached {len(opened)} of {len(peer)}, {stopped}"
    finally:
        for got in opened:
            hip.hipIpcCloseMemHandle(got)
        hip.hipGetLastError()
        # The exporter has to outlive every attach, so nothing is freed until
        # all of them have been closed.
        try:
            torch.distributed.barrier(group=group.cpu_group)
        except Exception:
            logger.warning("[EEP-DEBUG] attach ceiling barrier failed")
        for ptr in ptrs:
            hip.hipFree(ptr)
        hip.hipGetLastError()


def _off() -> bool:
    """Every probe here exports IPC handles, which is the very thing that
    fails, so a pass that only wants to know whether the test itself passes has
    to be able to turn all of this off. An earlier NCCL_P2P_DISABLE=1 pass
    looked like a failure when the only thing that failed was this file.
    """
    return os.getenv("VLLM_EEP_DEBUG_OFF") == "1"


def _warm_only() -> bool:
    """Warm each new group with one collective and measure nothing.

    Across both attempts of build 95060, in legacy mode with the normal
    transport, the scale up completed in all three passes that carried these
    probes and in none of the four that did not. The probes both warm and
    measure, so that split has two possible causes and I cannot tell them
    apart. This leaves only the warm-up: one real collective on each new group
    right after it is swapped in, before the EPLB reshuffle reaches its own
    first collective. It also swallows the failure instead of re-raising it, so
    a pass answers for the product and not for this file.
    """
    return os.getenv("VLLM_EEP_DEBUG_WARM_ONLY") == "1"


def _drain_probe() -> bool:
    """The 2 MiB ceiling is destructive on the agent, so it is opt in."""
    return os.getenv("VLLM_EEP_DEBUG_DRAIN") == "1"


def _attach_probe() -> bool:
    """The attach ceiling has produced a real GPU page fault, reported as
    `Memory access fault by GPU node-N` followed by an abort, so it is opt in
    until the exporter is proven to outlive every peer that mapped it.
    """
    return os.getenv("VLLM_EEP_DEBUG_ATTACH") == "1"


def dump(tag: str, groups: dict | None = None, heavy: bool = False) -> None:
    """Heavy adds probes that perturb state, so they stay off at the points
    right before a collective we are trying to observe."""
    if _off():
        return
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
        # The drop from 1536 in prepare to about 120 at the door happens
        # somewhere in between, and nothing has measured that stretch yet. A
        # limit of 256 holds 1.5 GiB for under a tenth of a second, and the same
        # measurement already runs immediately before a collective that then
        # succeeds, so it is safe to take here too.
        info["ipc_export_ceiling_6MiB_256"] = _ipc_export_ceiling(6 << 20, 256)
        # This one costs what it measures. Two dumps a second apart read 117
        # and then 0 on the agent, because the 2 MiB exports in the first one
        # never give their capacity back, so leaving it on turns every later
        # reading into a reading of this probe. It stays off unless a pass is
        # specifically asking how much a batch of 2 MiB exports consumes.
        if _drain_probe():
            info["ipc_export_ceiling_2MiB_768"] = _ipc_export_ceiling(2 << 20, 768)
        if heavy:
            # Burst can exhaust export handles and foreign_device creates a
            # context on a neighbour card, so neither runs near the collective.
            info["ipc_probe_burst_64x6MiB"] = _ipc_probe_burst(6 << 20, 64)
            info["ipc_probe_foreign_device"] = _ipc_probe_foreign_device(6 << 20)
            # Same size and same limit as the parent's own ceiling below, so
            # the two lines can be read side by side.
            info["ipc_probe_fresh_process"] = _ipc_probe_fresh_process(6 << 20, 1536)
            info["ipc_export_ceiling_2MiB"] = _ipc_export_ceiling(2 << 20, 4096)
            # 6 MiB is the size RCCL exports most of, and it holds about a
            # thousand of them per rank, so walk past that.
            info["ipc_export_ceiling_6MiB"] = _ipc_export_ceiling(6 << 20, 1536)
        logger.warning("[EEP-DEBUG] %s %s", tag, info)
    except Exception as e:
        logger.warning("[EEP-DEBUG] %s dump failed: %r", tag, e)


def dump_static() -> None:
    if _off():
        return
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
        logger.warning("[EEP-DEBUG] static ipc_export_mode=%s", _ipc_export_mode())
        logger.warning("[EEP-DEBUG] static kfd_topology=%s", _kfd_topology())
    except Exception as e:
        logger.warning("[EEP-DEBUG] static dump failed: %r", e)


def warm_each(groups: dict, tag: str) -> None:
    if _off():
        return
    warm_only = _warm_only()
    stream = torch.Stream(device=next(iter(groups.values())).device)
    with stream:
        tensor = torch.zeros(1, dtype=torch.int32, device=stream.device)
        for name, group in groups.items():
            try:
                if warm_only:
                    torch.distributed.all_reduce(tensor, group=group.device_group)
                    stream.synchronize()
                    logger.warning("[EEP-DEBUG] %s warm %s ok, warm only", tag, name)
                    continue
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
                # The nearest ceiling measurement before this point is the one
                # in prepare, which is far enough back that it cannot say
                # whether the headroom was already gone when RCCL started or
                # whether RCCL's own 32 channels consumed the last of it. This
                # measures right at the door. The limit stays small on purpose:
                # walking to 1536 here would hold 9 GiB and would itself be a
                # suspect if the collective then failed.
                # Four runs now show one rank falling to about 120 while the
                # others stay above the limit, so the rank belongs on the line.
                # Without it there is no way to tell whether it is the same rank
                # every time, and if it is, that points at what consumes it.
                #
                # The measurement runs twice because on this machine the first
                # of the two ends in a refusal, and a refused export releasing
                # its slot is not something that can be checked where no ceiling
                # is ever reached. Two equal numbers here mean the probe itself
                # is not consuming anything, which is what makes the number on
                # the far side of the collective readable.
                #
                # This goes before anything that needs a collective of its own.
                # The import ceiling used to run first and its exports on the
                # source rank left that rank reading zero here, and in the
                # dmabuf pass the whole thing stopped at that probe and this
                # line never printed at all.
                #
                # Twice in a row, because on the agent the second reading is
                # what exposes whether the first one cost anything. The 2 MiB
                # size answers the same question far more sharply but consumes
                # the remaining budget outright, so it waits for a pass that
                # asks for it.
                logger.warning(
                    "[EEP-DEBUG] %s warm %s headroom rank=%s/%s"
                    " 6MiB/uncached %s, again %s%s",
                    tag,
                    name,
                    group.rank_in_group,
                    group.world_size,
                    _ipc_export_ceiling(6 << 20, 256),
                    _ipc_export_ceiling(6 << 20, 256),
                    f" | 2MiB/uncached {_ipc_export_ceiling(2 << 20, 768)}"
                    if _drain_probe()
                    else "",
                )
                # The mesh probe attaches one buffer per peer, which answers
                # whether attaching works at all but not how many fit. This one
                # counts, but it has also faulted the GPU outright, so it only
                # runs when a pass asks for it.
                if _attach_probe():
                    logger.warning(
                        "[EEP-DEBUG] %s warm %s attach rank=%s/%s 2MiB/uncached %s",
                        tag,
                        name,
                        group.rank_in_group,
                        group.world_size,
                        _ipc_attach_ceiling(group, 2 << 20, 0x3, 256),
                    )
                if os.getenv("VLLM_EEP_DEBUG_SET_DEVICE") == "1":
                    torch.accelerator.set_device_index(group.device.index)
                torch.distributed.all_reduce(tensor, group=group.device_group)
                stream.synchronize()
                # The same measurement on the far side of the collective. RCCL
                # builds 32 channels in here and keeps their buffers, so if it
                # consumes the resource this is where the number drops. If it
                # comes back unchanged, RCCL only ever reads what was already
                # spent, which is what every run so far suggests.
                logger.warning(
                    "[EEP-DEBUG] %s warm %s headroom after rank=%s/%s 6MiB/uncached %s",
                    tag,
                    name,
                    group.rank_in_group,
                    group.world_size,
                    _ipc_export_ceiling(6 << 20, 256),
                )
                logger.warning("[EEP-DEBUG] %s warm %s ok", tag, name)
            except Exception as e:
                logger.warning(
                    "[EEP-DEBUG] %s warm %s FAILED rank=%s world_size=%s: %s",
                    tag,
                    name,
                    group.rank_in_group,
                    group.world_size,
                    _exception_chain(e),
                )
                logger.warning(
                    "[EEP-DEBUG] %s warm %s traceback:\n%s",
                    tag,
                    name,
                    traceback.format_exc(),
                )
                if not warm_only:
                    dump(f"{tag} after {name} failure", groups, heavy=True)
                    raise
