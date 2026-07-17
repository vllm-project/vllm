# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CRIU imports-snapshot for `vllm serve`.

`vllm snapshot create` freezes a helper process that has imported the serving
envelope but never initialized CUDA; `VLLM_SNAPSHOT=1 vllm serve` restores it
instead of re-paying the import bill. Any miss falls back to a normal cold
start. This module owns the whole mechanism; the CLI shim and the restore hook
in `cli/main.py` are one-liners into it.

Module-level imports stay light (stdlib plus vllm.logger, which cli/main.py
has already paid) so the restore hook can run before the serving-graph
imports; every torch/heavy-vllm import is function-local (helper mode and the
restored handler).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# ---- protocol / policy constants -------------------------------------------

PAYLOAD_FD = 9
DESIGN_VERSION = 1
CRIU_MIN = (3, 17)
CAP_CHECKPOINT_RESTORE = 40

READY_TIMEOUT = 30.0
PIDFILE_TIMEOUT = 30.0
ACK_TIMEOUT = 10.0
REAP_DEADLINE = 10.0
MAX_FRAME = 4 * 1024 * 1024

# The serve process's import envelope plus the EngineCore target graph its
# child re-imports under spawn; snapshotting the union covers both bills.
UNION_IMPORTS = (
    "vllm.entrypoints.cli.main",
    "vllm.entrypoints.openai.api_server",
    "vllm.v1.engine.async_llm",
    "vllm.v1.engine.core",
)

# Scrubbed from the canonical env on BOTH sides and exempt from the compare, so
# the opt-in flag never self-invalidates the snapshot it enables.
CONTROL_ENV = frozenset(
    {"VLLM_SNAPSHOT", "VLLM_SNAPSHOT_ROOT", "VLLM_SNAPSHOT_RESTORED"}
)
# Pins the helper is created under: default env leaves an OMP pool per nproc and
# device fds; these reach 0 device fds with one residual libcuda driver thread.
CREATION_PINS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES": "",
}
# v1 allowlist = the creation pins only. A restored process keeps its create-time
# thread-pool state regardless of the live values, a documented property.
ENV_ALLOWLIST = frozenset(CREATION_PINS)
# Recorded for diagnostics; the authoritative restore check is default-deny over
# the whole env minus the allowlist, not this subset.
IMPORT_AFFECTING_ENV = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "VLLM_PLUGINS",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_WORKER_MULTIPROC_METHOD",
    }
)


class SnapshotKeyError(RuntimeError):
    """The compatibility key cannot be computed (editable / RECORD-less dist)."""


class _RestoreInterrupted(BaseException):
    """A terminating signal arrived mid-restore.

    BaseException on purpose: an operator's SIGTERM must terminate the process
    after cleanup, never turn into an ordinary miss that cold-starts a server
    nobody asked for.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


# ---- small helpers ---------------------------------------------------------


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def canonical_env(env: dict[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if env is None else env)
    for name in CONTROL_ENV:
        source.pop(name, None)
    return dict(sorted(source.items()))


def creation_env(env: dict[str, str] | None = None) -> dict[str, str]:
    values = canonical_env(env)
    # Create runs with PYTHONHASHSEED unset: the interpreter's hash secret is
    # baked at helper start and cannot be refreshed post-restore.
    values.pop("PYTHONHASHSEED", None)
    values.update(CREATION_PINS)
    return dict(sorted(values.items()))


# Import-time code mutates process state in place (cv2's loader shim appends
# to LD_LIBRARY_PATH, torch and vllm.env_override set cache and worker vars,
# sys.path can grow). `snapshot create` dispatches AFTER the CLI's eager
# imports while the restore hook runs BEFORE them, so both sides must key the
# state as it was at CLI entry or they key different worlds and every restore
# misses. maybe_restore_serve() captures this on every CLI start.
_entry_state: dict[str, Any] = {}


def _capture_entry_state() -> None:
    if not _entry_state:
        _entry_state["env"] = canonical_env()
        _entry_state["sys_path"] = list(sys.path)


def _entry_env() -> dict[str, str]:
    return dict(_entry_state["env"]) if _entry_state else canonical_env()


def _entry_sys_path() -> list[str]:
    return list(_entry_state["sys_path"]) if _entry_state else list(sys.path)


def environment_digest(env: dict[str, str]) -> str:
    stable = {name: value for name, value in env.items() if name not in ENV_ALLOWLIST}
    return sha256_json(stable)


def environment_record(env: dict[str, str]) -> dict[str, Any]:
    return {
        "values": dict(sorted(env.items())),
        "import_affecting": {
            name: env[name] for name in sorted(IMPORT_AFFECTING_ENV) if name in env
        },
        "allowlist": sorted(ENV_ALLOWLIST),
        "digest": environment_digest(env),
    }


def environment_miss(
    recorded: dict[str, str], live: dict[str, str], allowlist: set[str] | frozenset[str]
) -> str | None:
    for name in sorted(set(recorded) | set(live)):
        if recorded.get(name) != live.get(name) and name not in allowlist:
            return f"env.{name}"
    return None


def fd_identity_record(raw: str) -> dict[str, str]:
    # For a regular file criu names the fd by its path; the root-relative form is
    # the documented one. For a socket the raw `socket:[ino]` is the identity.
    return {"raw": raw, "root_relative": raw.lstrip("/")}


# ---- length-prefixed framing over the payload socket -----------------------


def _write_bytes(channel: socket.socket | int, data: bytes) -> None:
    if isinstance(channel, socket.socket):
        channel.sendall(data)
        return
    view = memoryview(data)
    while view:
        view = view[os.write(channel, view) :]


def _read_exact(channel: socket.socket | int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        if isinstance(channel, socket.socket):
            chunk = channel.recv(remaining)
        else:
            chunk = os.read(channel, remaining)
        if not chunk:
            raise EOFError("payload channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_frame(channel: socket.socket | int, payload: dict[str, Any]) -> None:
    body = json_bytes(payload)
    if len(body) > MAX_FRAME:
        raise ValueError("payload frame is too large")
    _write_bytes(channel, len(body).to_bytes(4, "big") + body)


def read_frame(channel: socket.socket | int) -> dict[str, Any]:
    length = int.from_bytes(_read_exact(channel, 4), "big")
    if length <= 0 or length > MAX_FRAME:
        raise ValueError(f"invalid payload frame length: {length}")
    value = json.loads(_read_exact(channel, length).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("payload frame must be a JSON object")
    return value


def read_ack_or_frame(channel: socket.socket) -> tuple[str, dict[str, Any] | None]:
    first = _read_exact(channel, 1)
    if first == b"A":
        return "ack", None
    length = int.from_bytes(first + _read_exact(channel, 3), "big")
    if length <= 0 or length > MAX_FRAME:
        raise ValueError(f"invalid response frame length: {length}")
    value = json.loads(_read_exact(channel, length).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("response frame must be a JSON object")
    return "frame", value


# ---- manifest io -----------------------------------------------------------


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # 0600: manifests carry the recorded environment, credentials included
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest(snap_dir: Path) -> dict[str, Any]:
    value = read_json(snap_dir / "MANIFEST.json")
    if not isinstance(value, dict):
        raise ValueError(f"invalid manifest object: {snap_dir}")
    return value


# ---- compatibility key -----------------------------------------------------


def criu_version() -> str:
    try:
        result = subprocess.run(
            ["criu", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "missing"
    return (result.stdout or result.stderr).strip() or "missing"


def _criu_at_least(minimum: tuple[int, int]) -> bool:
    digits: list[int] = []
    for token in criu_version().replace(".", " ").split():
        if token.isdigit():
            digits.append(int(token))
        elif digits:
            break
    if len(digits) < 2:
        return False
    return (digits[0], digits[1]) >= minimum


def _cpuinfo_flags_digest() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unavailable"
    for line in text.splitlines():
        if line.startswith("flags"):
            return hashlib.sha256(line.split(":", 1)[1].strip().encode()).hexdigest()
    return "unavailable"


def _is_editable(dist: importlib.metadata.Distribution) -> bool:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return False
    try:
        info = json.loads(raw)
    except ValueError:
        return False
    return bool(isinstance(info, dict) and info.get("dir_info", {}).get("editable"))


def _distributions_digest() -> str:
    triples: list[list[str]] = []
    offenders: list[str] = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] or dist.name or "?"
        if _is_editable(dist):
            offenders.append(f"{name} (editable)")
            continue
        record = dist.read_text("RECORD")
        if not record:
            offenders.append(f"{name} (no RECORD)")
            continue
        digest = hashlib.sha256(record.encode()).hexdigest()
        triples.append([name, dist.version, digest])
    if offenders:
        raise SnapshotKeyError(
            "editable or RECORD-less distributions refuse snapshot: "
            + ", ".join(sorted(set(offenders)))
        )
    return sha256_json(sorted(triples))


def lookup_key(env: dict[str, str]) -> dict[str, Any]:
    """Pre-import-computable identity. Raises SnapshotKeyError if unkeyable."""
    return {
        "kernel": platform.release(),
        "python": {
            "version": platform.python_version(),
            "prefix": sys.prefix,
            "executable": sys.executable,
            "build": list(platform.python_build()),
            "sys_path_digest": sha256_json(_entry_sys_path()),
        },
        "env_digest": environment_digest(env),
        "dists_digest": _distributions_digest(),
        "cpu": {
            "machine": platform.machine(),
            "flags_digest": _cpuinfo_flags_digest(),
        },
        "criu": criu_version(),
        "design": DESIGN_VERSION,
    }


def key_from(key_obj: dict[str, Any]) -> str:
    return hashlib.sha256(json_bytes(key_obj)).hexdigest()[:16]


# ---- mapped shared-object identities (layer 2) -----------------------------


def _find_build_id(notes: bytes, endian: str) -> str | None:
    offset = 0
    while offset + 12 <= len(notes):
        namesz, descsz, ntype = struct.unpack_from(endian + "III", notes, offset)
        offset += 12
        name = notes[offset : offset + namesz]
        offset += (namesz + 3) & ~3
        desc = notes[offset : offset + descsz]
        offset += (descsz + 3) & ~3
        if ntype == 3 and name.rstrip(b"\x00") == b"GNU":  # NT_GNU_BUILD_ID
            return desc.hex()
    return None


def _elf_build_id(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
            if head[:4] != b"\x7fELF":
                return None
            is64 = head[4] == 2
            endian = "<" if head[5] == 1 else ">"
            if is64:
                phoff = struct.unpack_from(endian + "Q", head, 32)[0]
                phentsize, phnum = struct.unpack_from(endian + "HH", head, 54)
            else:
                phoff = struct.unpack_from(endian + "I", head, 28)[0]
                phentsize, phnum = struct.unpack_from(endian + "HH", head, 42)
            handle.seek(phoff)
            program_headers = handle.read(phentsize * phnum)
            for index in range(phnum):
                base = index * phentsize
                p_type = struct.unpack_from(endian + "I", program_headers, base)[0]
                if p_type != 4:  # PT_NOTE
                    continue
                if is64:
                    offsets = struct.unpack_from(
                        endian + "Q", program_headers, base + 8
                    )
                    sizes = struct.unpack_from(endian + "Q", program_headers, base + 32)
                else:
                    offsets = struct.unpack_from(
                        endian + "I", program_headers, base + 4
                    )
                    sizes = struct.unpack_from(endian + "I", program_headers, base + 16)
                p_offset, p_filesz = offsets[0], sizes[0]
                handle.seek(p_offset)
                build_id = _find_build_id(handle.read(p_filesz), endian)
                if build_id:
                    return build_id
    except (OSError, struct.error, IndexError):
        return None
    return None


def object_identity(path: str) -> str | None:
    build_id = _elf_build_id(path)
    if build_id:
        return f"buildid:{build_id}"
    try:
        with open(path, "rb") as handle:
            return "sha256:" + hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _distribution_owned_libraries() -> set[str]:
    # Exact dist-file membership, not path prefixes: a prefix rule would let an
    # unmanaged native module on PYTHONPATH change in place without changing
    # the key, and a system-python prefix would swallow /usr/lib/**/libcuda.so.
    owned: set[str] = set()
    for dist in importlib.metadata.distributions():
        for pkg_path in dist.files or []:
            if ".so" not in pkg_path.name:
                continue
            with contextlib.suppress(OSError, ValueError):
                owned.add(os.path.realpath(str(pkg_path.locate())))
    return owned


def collect_shared_objects(pid: int) -> list[dict[str, str]]:
    """(path, identity) for every mapped .so no distribution owns.

    Distribution-owned .so files are already covered by dists_digest; the
    remainder is system and driver libraries. A driver bump changes libcuda's
    build-id here, which is what invalidates a GPU-created snapshot into a
    clean re-create.
    """
    try:
        maps = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    owned = _distribution_owned_libraries()
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if not path.startswith("/") or ".so" not in os.path.basename(path):
            continue
        real = os.path.realpath(path)
        if real in seen or real in owned:
            seen.add(real)
            continue
        seen.add(real)
        identity = object_identity(real)
        if identity is not None:
            records.append({"path": real, "id": identity})
    return sorted(records, key=lambda record: record["path"])


# ---- process reap (descendant walk + dual-condition settle) ----------------


def process_table() -> dict[int, tuple[int, int, int, str]]:
    # pid -> (ppid, pgid, starttime, state); state 'Z' marks a reaped-pending
    # corpse, not a leak.
    if sys.platform != "linux":
        return {}
    table: dict[int, tuple[int, int, int, str]] = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            pid = int(path.parent.name)
            # comm (field 2) can contain spaces/parens; the tail after the last
            # ')' is pure ASCII: state ppid pgrp ... starttime(idx 19).
            tail = path.read_bytes().rsplit(b")", 1)[1].split()
            table[pid] = (int(tail[1]), int(tail[2]), int(tail[19]), tail[0].decode())
        except (OSError, IndexError, ValueError):
            continue
    return table


def descendant_identities(root_pid: int) -> dict[int, tuple[int, int]]:
    # pid -> (starttime, pgid) for root_pid and everything descended from it;
    # starttime guards against pid reuse across the kill/settle window.
    identities: dict[int, tuple[int, int]] = {}
    known = {root_pid}
    for _ in range(3):
        table = process_table()
        changed = True
        while changed:
            changed = False
            for pid, (ppid, pgid, starttime, _state) in table.items():
                if pid in known or ppid in known:
                    if pid not in known:
                        known.add(pid)
                        changed = True
                    identities.setdefault(pid, (starttime, pgid))
    return identities


def _leakers(root_pid: int, targets: dict[int, tuple[int, int]]) -> list[int]:
    table = process_table()
    leaked: list[int] = []
    for pid, (starttime, _pgid) in targets.items():
        row = table.get(pid)
        if row is not None and row[2] == starttime and row[3] != "Z":
            leaked.append(pid)
    for pid, (_ppid, pgid, _starttime, state) in table.items():
        if pgid == root_pid and state != "Z" and pid not in leaked:
            leaked.append(pid)
    return leaked


def pgid_empty(pgid: int) -> bool:
    result = subprocess.run(
        ["pgrep", "-g", str(pgid), "."],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return not result.stdout.strip()
    if result.returncode == 1:
        return True
    raise RuntimeError(f"pgrep -g {pgid} failed rc={result.returncode}")


def kill_restored_group(pid: int, deadline_s: float = REAP_DEADLINE) -> list[int]:
    targets = descendant_identities(pid)
    pgids = {pid} | {pgid for _, pgid in targets.values()}
    pgids.discard(os.getpgrp())  # never SIGKILL the wrapper's own group
    for pgid in pgids:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
    for target_pid in targets:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(target_pid, signal.SIGKILL)
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        leaked = _leakers(pid, targets)
        # settle needs BOTH no live leakers AND the group drained: zombies are
        # not re-killable and not leaks, but the group is not settled until
        # their reaper collects them.
        if not leaked and pgid_empty(pid):
            break
        for target_pid in leaked:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(target_pid, signal.SIGKILL)
        time.sleep(0.05)
    residual = _leakers(pid, targets)
    if residual:
        logger.warning("snapshot reap left processes behind: %s", residual)
    return residual


# ---- host / privilege ------------------------------------------------------


def _has_dump_privilege() -> bool:
    if os.geteuid() == 0:
        return True
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapEff:"):
                return bool(int(line.split()[1], 16) & (1 << CAP_CHECKPOINT_RESTORE))
    except (OSError, ValueError):
        pass
    return False


def require_dump_host() -> None:
    if sys.platform != "linux":
        raise RuntimeError("vllm snapshot create requires linux")
    if shutil.which("criu") is None:
        raise RuntimeError("vllm snapshot create requires criu on PATH")
    if not _criu_at_least(CRIU_MIN):
        raise RuntimeError(
            f"vllm snapshot create requires criu >= {CRIU_MIN[0]}.{CRIU_MIN[1]}"
        )
    if not _has_dump_privilege():
        raise RuntimeError(
            "vllm snapshot create requires root or CAP_CHECKPOINT_RESTORE"
        )
    import torch  # already imported by main()'s eager serve import

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "vllm snapshot create requires CUDA uninitialized in this process"
        )


def _snapshot_root() -> Path:
    from vllm import envs

    # resolved: criu requires absolute paths for its own arguments
    return Path(envs.VLLM_SNAPSHOT_ROOT).expanduser().resolve()


def _acquire_lock(root: Path, key: str) -> _KeyLock:
    # One stable per-key lock OUTSIDE the replaceable dir, so a concurrent
    # --force rmtree cannot invalidate what a restorer is reading.
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    handle = (root / f"{key}.lock").open("w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return _KeyLock(handle)


class _KeyLock:
    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def release(self) -> None:
        if self.handle is None:
            return
        import fcntl

        with contextlib.suppress(OSError):
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


# ---- helper mode (the dumped process) --------------------------------------


def _adopt_payload_fd(src: int) -> None:
    if src != PAYLOAD_FD:
        os.dup2(src, PAYLOAD_FD)
        os.close(src)


def _sweep_inherited_fds() -> None:
    # close_fds/pass_fds handles python-created fds; descriptors inheritable
    # before python started would otherwise land inside the criu image.
    keep = {0, 1, 2, PAYLOAD_FD}
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        return
    for entry in list(fd_dir.iterdir()):
        try:
            fd = int(entry.name)
        except ValueError:
            continue
        if fd not in keep:
            with contextlib.suppress(OSError):
                os.close(fd)


def _set_stdio_distinct(work_dir: Path) -> None:
    # Distinct files give each stdio fd its own --inherit-fd identity; a shared
    # /dev/null would give all three the same id.
    for target, name in ((0, "stdin.null"), (1, "stdout.null"), (2, "stderr.null")):
        fd = os.open(str(work_dir / name), os.O_RDWR | os.O_CREAT, 0o600)
        os.dup2(fd, target)
        if fd > 2:
            os.close(fd)


def _join_threads() -> None:
    current = threading.current_thread()
    for thread in threading.enumerate():
        if thread is not current:
            thread.join(timeout=2.0)


def task_inventory() -> dict[str, str]:
    inventory: dict[str, str] = {}
    task_dir = Path("/proc/self/task")
    for entry in sorted(task_dir.iterdir(), key=lambda value: value.name):
        try:
            inventory[entry.name] = (entry / "comm").read_text().strip()
        except OSError:
            inventory[entry.name] = "?"
    return inventory


def audit_dump_state() -> dict[str, Any]:
    """Dump-time precondition check (lenient form; PR-1 restores via spawn).

    HARD-fail: any /dev/nvidia* fd, any device mapping, or any extra task that
    is not a libcuda driver thread (comm cuda*). RECORDED, not fatal: driver
    library .so maps and the one residual driver thread.
    """
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("linux required: /proc is unavailable")
    tasks = task_inventory()
    own = str(os.getpid())
    extra = {tid: comm for tid, comm in tasks.items() if tid != own}
    foreign = {tid: comm for tid, comm in extra.items() if not comm.startswith("cuda")}
    if foreign:
        raise RuntimeError(f"non-driver extra tasks at dump point: {foreign}")
    fd_offenders = [
        f"{entry.name}:{os.readlink(entry)}"
        for entry in Path("/proc/self/fd").iterdir()
        if _links_device(entry)
    ]
    device_maps = [
        line
        for line in maps.read_text(errors="replace").splitlines()
        if "/dev/nvidia" in line.lower()
    ]
    if fd_offenders or device_maps:
        raise RuntimeError(
            f"device-state audit failed: fds={fd_offenders} maps={device_maps}"
        )
    return {
        "tasks": tasks,
        "driver_threads": {
            tid: comm for tid, comm in extra.items() if comm.startswith("cuda")
        },
    }


def _links_device(entry: Path) -> bool:
    try:
        return os.readlink(entry).startswith("/dev/nvidia")
    except OSError:
        return False


def _read_own_fd_identities() -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for number in (0, 1, 2, PAYLOAD_FD):
        values[str(number)] = fd_identity_record(os.readlink(f"/proc/self/fd/{number}"))
    return values


def _post_restore_refresh() -> None:
    # create-time RNG/authkey/uuid state is baked into the image; refresh what
    # must differ per restore (PYTHONHASHSEED cannot be refreshed, documented).
    import random
    import uuid

    random.seed()
    # best-effort refresh, never fatal
    with contextlib.suppress(Exception):
        from multiprocessing import current_process

        current_process().authkey = os.urandom(32)
    with contextlib.suppress(Exception):
        uuid._node = None  # type: ignore[attr-defined]


def _dispatch_serve() -> int:
    from vllm.entrypoints.cli.main import main

    main()
    return 0


def _helper_resume(payload: dict[str, Any], manifest_path: Path) -> int:
    manifest = read_manifest(manifest_path.parent)
    recorded = {
        str(k): str(v) for k, v in manifest.get("env", {}).get("values", {}).items()
    }
    allowlist = set(manifest.get("env", {}).get("allowlist", ENV_ALLOWLIST))
    live = {str(k): str(v) for k, v in payload.get("env", {}).items()}
    miss = environment_miss(recorded, live, allowlist)
    if miss:
        write_frame(PAYLOAD_FD, {"miss": miss})
        return 3
    os.environ.clear()
    os.environ.update(live)
    os.environ["VLLM_SNAPSHOT_RESTORED"] = "1"
    cwd = payload.get("cwd")
    if isinstance(cwd, str):
        try:
            os.chdir(cwd)
        except OSError as error:
            # committing a server that runs from the snapshot work dir would
            # be worse than a cold start; refuse pre-ack
            write_frame(PAYLOAD_FD, {"miss": f"cwd: {error}"})
            return 3
    argv = payload.get("argv")
    if isinstance(argv, list) and argv:
        sys.argv = [str(value) for value in argv]
    else:
        sys.argv = sys.argv[:1]
    _post_restore_refresh()
    os.write(PAYLOAD_FD, b"A")
    code = 1
    try:
        code = _dispatch_serve()
        return code
    except SystemExit as error:
        if error.code is None:
            code = 0
        else:
            code = error.code if isinstance(error.code, int) else 1
        raise
    finally:
        # criu's non-detached rc is 0 regardless of the task's exit, so the real
        # status travels back as a final frame the wrapper prefers.
        with contextlib.suppress(OSError):
            write_frame(PAYLOAD_FD, {"exit_code": code})


def _helper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", action="store_true", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sys-path", required=True)
    parser.add_argument("--payload-fd", type=int, default=PAYLOAD_FD)
    return parser.parse_args()


def _helper_main() -> int:
    args = _helper_args()
    # own our session so pre-commit kill is a defined killpg
    with contextlib.suppress(PermissionError):
        os.setsid()
    _adopt_payload_fd(args.payload_fd)
    _sweep_inherited_fds()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    _set_stdio_distinct(work_dir)
    keyed = json.loads(Path(args.sys_path).read_text())
    sys.path[:] = keyed
    effective_sys_path = list(sys.path)
    for name in UNION_IMPORTS:
        importlib.import_module(name)
    _join_threads()
    audit = audit_dump_state()
    state = {
        "pid": os.getpid(),
        "sys_path": effective_sys_path,
        "fd_identities": _read_own_fd_identities(),
        "shared_objects": collect_shared_objects(os.getpid()),
        "driver_threads": audit["driver_threads"],
        "work_assets": ["stdin.null", "stdout.null", "stderr.null"],
    }
    resumed = {"value": False}

    def _wake(_signum: int, _frame: Any) -> None:
        resumed["value"] = True

    signal.signal(signal.SIGUSR2, _wake)
    write_json_atomic(Path(args.ready), state)
    signal.pause()
    if not resumed["value"]:
        raise RuntimeError("snapshot resumed without SIGUSR2")
    payload = read_frame(PAYLOAD_FD)
    return _helper_resume(payload, Path(args.manifest))


# ---- create ----------------------------------------------------------------


def _prepare_directory(directory: Path, force: bool) -> None:
    if directory.exists():
        if (directory / "MANIFEST.json").exists() and not force:
            raise RuntimeError(f"snapshot already exists, pass --force: {directory}")
        shutil.rmtree(directory)
    (directory / "work").mkdir(parents=True)
    (directory / "imgs").mkdir()
    (directory / "dump.log").touch()
    # 0700/0600: the criu images and logs hold full process memory, the
    # recorded environment included
    for entry in (directory, directory / "work", directory / "imgs"):
        os.chmod(entry, 0o700)
    os.chmod(directory / "dump.log", 0o600)


def _spawn_helper(
    directory: Path, env: dict[str, str], subject_fd: int
) -> subprocess.Popen[bytes]:
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.snapshot",
        "--helper",
        "--work-dir",
        str(directory / "work"),
        "--ready",
        str(directory / "work" / "ready.json"),
        "--manifest",
        str(directory / "MANIFEST.json"),
        "--sys-path",
        str(directory / "work" / "sys_path.json"),
        "--payload-fd",
        str(subject_fd),
    ]
    return subprocess.Popen(
        argv,
        cwd=str(directory / "work"),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=(subject_fd,),
        start_new_session=True,
    )


def _wait_for_ready(process: subprocess.Popen[bytes], ready: Path) -> dict[str, Any]:
    deadline = time.monotonic() + READY_TIMEOUT
    while not ready.exists():
        if process.poll() is not None:
            raise RuntimeError(
                f"helper exited before ready (status {process.returncode})"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for helper ready")
        time.sleep(0.05)
    return read_json(ready)


def _dump_argv(directory: Path, pid: int, socket_ino: str) -> list[str]:
    # Absolute log path: criu resolves a relative --log-file against the images
    # dir. A connected socketpair half needs `--external unix[ino]` explicitly;
    # `--ext-unix-sk` alone fails "Can't dump half of stream unix connection".
    return [
        "criu",
        "dump",
        "--tree",
        str(pid),
        "--images-dir",
        str(directory / "imgs"),
        "--log-file",
        str(directory / "dump.log"),
        "--shell-job",
        "--ext-unix-sk",
        "--external",
        f"unix[{socket_ino}]",
    ]


def _payload_socket_ino(fd_identities: dict[str, Any]) -> str:
    raw = fd_identities[str(PAYLOAD_FD)]["raw"]
    if not raw.startswith("socket:[") or not raw.endswith("]"):
        raise RuntimeError(f"payload fd is not a socket: {raw}")
    return raw[len("socket:[") : -1]


def _print_dry_run(key: str, directory: Path) -> None:
    template = [
        "criu",
        "dump",
        "--tree",
        "<pid>",
        "--images-dir",
        str(directory / "imgs"),
        "--log-file",
        str(directory / "dump.log"),
        "--shell-job",
        "--ext-unix-sk",
        "--external",
        "unix[<socket-ino>]",
    ]
    print(f"snapshot key: {key}")
    print(f"target dir:   {directory}")
    print(f"dump argv:    {shlex.join(template)}")


def _run_create(
    key: str, directory: Path, create_env: dict[str, Any], key_obj: dict[str, Any]
) -> None:
    write_json_atomic(directory / "work" / "sys_path.json", _entry_sys_path())
    subject, peer = socket.socketpair()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = _spawn_helper(directory, create_env, subject.fileno())
        subject.close()
        subject = None  # type: ignore[assignment]
        state = _wait_for_ready(process, directory / "work" / "ready.json")
        if state.get("sys_path") != _entry_sys_path():
            raise RuntimeError("helper sys.path diverged from the keyed sys.path")
        dump_argv = _dump_argv(
            directory, process.pid, _payload_socket_ino(state["fd_identities"])
        )
        result = subprocess.run(dump_argv, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"criu dump failed (status {result.returncode})")
        try:
            process.wait(timeout=READY_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("helper still alive after criu dump") from error
        manifest = {
            "design_version": DESIGN_VERSION,
            "snapshot_key": key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "subject_pid": state["pid"],
            "lookup_key": key_obj,
            "env": environment_record(create_env),
            "fd_identities": state["fd_identities"],
            "dump_argv": dump_argv,
            "sys_path": state["sys_path"],
            "shared_objects": state["shared_objects"],
            "driver_threads": state["driver_threads"],
            "work_assets": state["work_assets"],
        }
        write_json_atomic(directory / "MANIFEST.json", manifest)
        process = None
    except BaseException:
        if process is not None and process.poll() is None:
            kill_restored_group(process.pid)
        # dump.log is inside the dir about to be removed; surface it first
        for label, text in (
            ("helper error", _read_helper_error(directory)),
            ("criu dump log tail", _log_tail(directory / "dump.log")),
        ):
            if text:
                logger.error("snapshot create failed; %s:\n%s", label, text)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            peer.close()


def _read_helper_error(directory: Path) -> str | None:
    try:
        return (directory / "work" / "helper.error").read_text()
    except OSError:
        return None


def _log_tail(path: Path, lines: int = 30) -> str | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    tail = text.splitlines()[-lines:]
    return "\n".join(tail) if tail else None


def create_snapshot(force: bool = False, dry_run: bool = False) -> None:
    require_dump_host()
    create_env = creation_env(_entry_env())
    key_obj = lookup_key(create_env)  # raises SnapshotKeyError on editable/RECORD-less
    key = key_from(key_obj)
    root = _snapshot_root()
    directory = root / key
    if dry_run:
        _print_dry_run(key, directory)
        return
    lock = _acquire_lock(root, key)
    try:
        _prepare_directory(directory, force)
        _run_create(key, directory, create_env, key_obj)
    finally:
        lock.release()
    logger.info("snapshot created key=%s dir=%s", key, directory)


# ---- restore hook (the `vllm serve` wrapper) -------------------------------


def _first_diff(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    for field in sorted(set(a) | set(b)):
        av, bv = a.get(field), b.get(field)
        if isinstance(av, dict) and isinstance(bv, dict):
            nested = _first_diff(av, bv)
            if nested:
                return f"{field}.{nested}"
        elif av != bv:
            return field
    return None


def _diagnose_miss(
    root: Path, live_key_obj: dict[str, Any], key: str, live_env: dict[str, str]
) -> str:
    try:
        others = [
            entry
            for entry in root.iterdir()
            if entry.is_dir()
            and entry.name != key
            and (entry / "MANIFEST.json").is_file()
        ]
    except OSError:
        return "no snapshot"
    if not others:
        return "no snapshot"
    newest = max(others, key=lambda entry: (entry / "MANIFEST.json").stat().st_mtime)
    try:
        manifest = read_manifest(newest)
    except (OSError, ValueError):
        return "no snapshot"
    field = _first_diff(manifest.get("lookup_key", {}), live_key_obj)
    if field == "env_digest":
        # name the actual variable, not the digest
        recorded = {
            str(k): str(v) for k, v in manifest.get("env", {}).get("values", {}).items()
        }
        allowlist = set(manifest.get("env", {}).get("allowlist", ENV_ALLOWLIST))
        var = environment_miss(recorded, live_env, allowlist)
        if var:
            field = var
    return f"no snapshot (nearest differs at {field})" if field else "no snapshot"


def _validate_layer2(
    manifest: dict[str, Any], directory: Path, live_env: dict[str, str]
) -> str | None:
    for record in manifest.get("shared_objects", []):
        current = object_identity(record["path"])
        if current is None or current != record["id"]:
            return f"so.{os.path.basename(record['path'])}"
    work = directory / "work"
    for name in manifest.get("work_assets", []):
        if not (work / name).exists():
            return f"work.{name}"
    recorded = {
        str(k): str(v) for k, v in manifest.get("env", {}).get("values", {}).items()
    }
    allowlist = set(manifest.get("env", {}).get("allowlist", ENV_ALLOWLIST))
    return environment_miss(recorded, live_env, allowlist)


def _restore_argv(
    directory: Path,
    manifest: dict[str, Any],
    socket_fd: int,
    stdio_fds: tuple[int, int, int],
    run_dir: Path,
) -> list[str]:
    fd_identities = manifest["fd_identities"]
    argv = [
        "criu",
        "restore",
        "--images-dir",
        str(directory / "imgs"),
        "--log-file",
        str(run_dir / "restore.log"),
        "--shell-job",
        "--ext-unix-sk",
        "--pidfile",
        str(run_dir / "pid"),
        "--inherit-fd",
        f"fd[{socket_fd}]:{fd_identities[str(PAYLOAD_FD)]['raw']}",
    ]
    for image_fd, live_fd in zip(("0", "1", "2"), stdio_fds):
        identity = fd_identities[image_fd].get("root_relative")
        if not identity:
            # never invoke criu with a missing inherit-fd identity
            raise RuntimeError(f"missing inherit-fd identity fd[{image_fd}]")
        argv.extend(["--inherit-fd", f"fd[{live_fd}]:{identity}"])
    return argv


def _read_pidfile(path: Path) -> int | None:
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def _wait_pidfile(process: subprocess.Popen[bytes], path: Path, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while True:
        pid = _read_pidfile(path)
        if pid is not None:
            return pid
        if process.poll() is not None:
            raise RuntimeError(
                f"criu restore exited before pidfile (status {process.returncode})"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for restore pidfile")
        time.sleep(0.05)


def _wait_bounded(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _send_signal(pid: int, signum: int) -> None:
    pidfd = None
    try:
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send = getattr(signal, "pidfd_send_signal", None)
        if pidfd_open and pidfd_send:
            pidfd = pidfd_open(pid)
            pidfd_send(pidfd, signum)
            return
        os.kill(pid, signum)
    except (OSError, ProcessLookupError):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signum)
    finally:
        if pidfd is not None:
            os.close(pidfd)


def _read_status_frame(sock: socket.socket) -> int:
    # Post-commit EOF with no well-formed frame = abnormal termination (fatal
    # signal / native crash / OOM), fixed exit 254; criu's rc is never consulted.
    try:
        sock.settimeout(2.0)
        code = read_frame(sock).get("exit_code")
        if isinstance(code, int):
            return code
    except (OSError, EOFError, ValueError):
        pass
    return 254


def _run_criu_restore(
    directory: Path,
    manifest: dict[str, Any],
    live_env: dict[str, str],
    lock: _KeyLock,
) -> None:
    """Return on a pre-commit miss (cold fallback).

    Raises SystemExit with the restored server's status after a commit, or
    with 128+signum when a terminating signal lands mid-restore.
    """
    run_dir = Path(tempfile.mkdtemp(prefix="vllm-snapshot-restore-"))
    parent_sock, child_sock = socket.socketpair()
    stdio_fds: tuple[int, ...] = (os.dup(0), os.dup(1), os.dup(2))
    process: subprocess.Popen[bytes] | None = None
    restored_pid: int | None = None
    committed = False
    old_handlers: dict[int, Any] = {}
    launch = time.monotonic()
    try:
        argv = _restore_argv(
            directory, manifest, child_sock.fileno(), stdio_fds, run_dir
        )

        # installed BEFORE launch so a mid-restore signal cleans the paused tree
        # instead of stranding it.
        def _interrupt(signum: int, _frame: Any) -> None:
            raise _RestoreInterrupted(signum)

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _interrupt)
        process = subprocess.Popen(
            argv,
            close_fds=True,
            pass_fds=(child_sock.fileno(), *stdio_fds),
            env=canonical_env(live_env),
        )
        child_sock.close()
        child_sock = None  # type: ignore[assignment]
        for fd in stdio_fds:
            os.close(fd)
        stdio_fds = ()
        parent_sock.settimeout(ACK_TIMEOUT)
        write_frame(
            parent_sock,
            {"argv": list(sys.argv), "env": live_env, "cwd": os.getcwd()},
        )
        restored_pid = _wait_pidfile(process, run_dir / "pid", PIDFILE_TIMEOUT)
        os.kill(restored_pid, signal.SIGUSR2)
        kind, response = read_ack_or_frame(parent_sock)
        if kind != "ack":
            reason = str((response or {}).get("miss", "protocol"))
            kill_restored_group(restored_pid)
            _wait_bounded(process)
            logger.info("snapshot restore miss (%s)", reason)
            return
        committed = True
        lock.release()  # restore critical section ends at commit ack
        logger.info(
            "snapshot restore hit key=%s restore_ms=%d",
            directory.name,
            int((time.monotonic() - launch) * 1000),
        )
        parent_sock.settimeout(None)

        def _forward(signum: int, _frame: Any) -> None:
            _send_signal(restored_pid, signum)  # type: ignore[arg-type]

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, _forward)
        process.wait()
        raise SystemExit(_read_status_frame(parent_sock))
    except BaseException as error:
        if not committed:
            pid = restored_pid
            if pid is None:
                pid = _read_pidfile(run_dir / "pid")
            if pid is not None:
                kill_restored_group(pid)
            elif process is not None:
                # criu died or never wrote the pidfile: reap whatever part of
                # the restored tree it materialized, not just criu itself
                kill_restored_group(process.pid)
            if process is not None:
                _wait_bounded(process)
            if isinstance(error, _RestoreInterrupted):
                raise SystemExit(128 + error.signum) from None
            if isinstance(error, Exception):
                reason = str(error)
                tail = _log_tail(run_dir / "restore.log", lines=5)
                if tail:
                    # one INFO line: fold a sanitized criu-log tail into it
                    reason = f"{reason}; criu log: {' | '.join(tail.splitlines())}"
                logger.info("snapshot restore miss (%s)", reason)
                return
        raise
    finally:
        if child_sock is not None:
            child_sock.close()
        parent_sock.close()
        for fd in stdio_fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        shutil.rmtree(run_dir, ignore_errors=True)


def _restore_serve() -> None:
    live_env = _entry_env()
    key_obj = lookup_key(live_env)  # raises SnapshotKeyError -> miss
    key = key_from(key_obj)
    root = _snapshot_root()
    directory = root / key
    if not (directory / "MANIFEST.json").is_file():
        logger.info(
            "snapshot restore miss (%s)",
            _diagnose_miss(root, key_obj, key, live_env),
        )
        return
    lock = _acquire_lock(root, key)
    try:
        manifest = read_manifest(directory)
        miss = _validate_layer2(manifest, directory, live_env)
        if miss:
            logger.info("snapshot restore miss (%s)", miss)
            return
        _run_criu_restore(directory, manifest, live_env, lock)
    finally:
        lock.release()


def maybe_restore_serve() -> None:
    """First statement of the vllm CLI main(): restore a serve snapshot if one
    matches, else return so the normal cold path runs. Never returns once a
    restore commits (the process becomes the restored server)."""
    try:
        enabled = bool(int(os.getenv("VLLM_SNAPSHOT", "0")))
    except ValueError:
        enabled = False
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if not (command == "snapshot" or (enabled and command == "serve")):
        return  # gate closed: no work at all on any other path
    _capture_entry_state()  # before the eager imports mutate env/sys.path
    if command == "snapshot":
        return  # capture only; create keys off it
    if os.environ.get("VLLM_SNAPSHOT_RESTORED"):
        return  # already inside a restored process; no recursive restore
    if sys.platform != "linux":
        return
    try:
        _restore_serve()
    except SystemExit:
        raise  # a committed restore propagates its exit status
    except Exception as error:  # noqa: BLE001 - any failure is a cold fallback
        logger.info("snapshot restore miss (%s)", error)


if __name__ == "__main__":
    try:
        raise SystemExit(_helper_main())
    except SystemExit:
        raise
    except BaseException:
        import traceback

        with contextlib.suppress(OSError):
            Path("helper.error").write_text(traceback.format_exc())
        raise
