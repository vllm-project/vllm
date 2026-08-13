# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CacheConfig fingerprinting and socket protocol for the weight cache daemon.

The protocol uses pickle over a Unix domain socket and is only intended for
communication between trusted local processes owned by the same user. The
sockets live in a per-user private directory (mode 0700) and the daemon
restricts the socket file permissions to the owner (0600). Both the daemon and
the engine verify that the directory and socket are owned by the current user
and are not group/world accessible before trusting them, so a different local
user cannot pre-plant a malicious socket at a predictable path.
"""

import json
import os
import pickle
import socket
import stat
import struct
import tempfile
from dataclasses import dataclass, fields
from typing import Any

import torch

import vllm.version
from vllm.config import ModelConfig
from vllm.utils.hashing import safe_hash

SOCKET_NAME_TEMPLATE = "vllm_weight_cache_gpu{gpu_id}.sock"
SOCKET_DIR_TEMPLATE = "vllm_weight_cache_{uid}"

_LEN_STRUCT = struct.Struct("!Q")
# Sanity bound for a single message. IPC handles are tiny; only small
# non-CUDA tensors are ever shipped by value.
MAX_MSG_SIZE = 1 << 34


def _current_uid() -> int:
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else -1


class WeightCacheUnavailableError(Exception):
    """Raised when no weight cache daemon is reachable or usable."""


class CacheConfigMismatchError(Exception):
    """Raised when the daemon's cached weights don't match the engine."""


def get_physical_device_id(device_index: int) -> int | None:
    """Map a local CUDA device index to the physical GPU id.

    Returns None if CUDA_VISIBLE_DEVICES contains non-integer entries
    (e.g. GPU UUIDs), in which case an explicit socket path is required.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return device_index
    entries = visible.split(",")
    try:
        return int(entries[device_index])
    except (IndexError, ValueError):
        return None


def get_socket_dir(socket_dir: str | None = None) -> str:
    """Return the directory that holds the daemon sockets.

    When no explicit directory is given, use a per-user private directory
    under the system temp dir so its path is unpredictable to other users and
    can be locked down to mode 0700.
    """
    if socket_dir is not None:
        return socket_dir
    return os.path.join(
        tempfile.gettempdir(), SOCKET_DIR_TEMPLATE.format(uid=_current_uid())
    )


def get_socket_path(gpu_id: int, socket_dir: str | None = None) -> str:
    directory = get_socket_dir(socket_dir)
    return os.path.join(directory, SOCKET_NAME_TEMPLATE.format(gpu_id=gpu_id))


def ensure_private_socket_dir(directory: str, strict_perms: bool = True) -> None:
    """Create the socket directory (if needed) locked down to the owner.

    Called by the daemon before binding. Existing directories are re-checked
    and, for the auto-derived path, tightened so a pre-existing world-writable
    directory is rejected.
    """
    os.makedirs(directory, mode=0o700, exist_ok=True)
    if strict_perms:
        os.chmod(directory, 0o700)
    verify_private_dir(directory, strict_perms=strict_perms)


def verify_private_dir(directory: str, strict_perms: bool = True) -> None:
    """Verify a directory is a real dir owned by us and not world/group readable.

    When ``strict_perms`` is False the group/world permission bits are not
    checked; this is used for directories the operator explicitly configured
    (they own the trust decision), while the auto-derived per-user directory is
    always checked strictly.
    """
    info = os.lstat(directory)
    if stat.S_ISLNK(info.st_mode):
        raise WeightCacheUnavailableError(
            f"Refusing to use symlinked socket directory {directory}"
        )
    if not stat.S_ISDIR(info.st_mode):
        raise WeightCacheUnavailableError(f"{directory} is not a directory")
    uid = _current_uid()
    if uid != -1 and info.st_uid != uid:
        raise WeightCacheUnavailableError(
            f"Socket directory {directory} is not owned by the current user"
        )
    if strict_perms and info.st_mode & 0o077:
        raise WeightCacheUnavailableError(
            f"Socket directory {directory} is group/world accessible"
        )


def verify_socket_owner(socket_path: str, strict_perms: bool = True) -> None:
    """Verify the socket lives in a private dir and is owned by the current user.

    Called by the engine before connecting so it never talks to a socket a
    different user could have planted.
    """
    verify_private_dir(os.path.dirname(socket_path), strict_perms=strict_perms)
    info = os.lstat(socket_path)
    if stat.S_ISLNK(info.st_mode):
        raise WeightCacheUnavailableError(
            f"Refusing to connect to symlinked socket {socket_path}"
        )
    uid = _current_uid()
    if uid != -1 and info.st_uid != uid:
        raise WeightCacheUnavailableError(
            f"Socket {socket_path} is not owned by the current user"
        )


def verify_peer_is_owner(conn: socket.socket) -> None:
    """Best-effort check that the connecting peer runs as the current user.

    Uses SO_PEERCRED where available (Linux). Silently returns on platforms
    that do not expose peer credentials.
    """
    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is None:
        return
    try:
        creds = conn.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
        _, peer_uid, _ = struct.unpack("3i", creds)
    except OSError:
        return
    uid = _current_uid()
    if uid != -1 and peer_uid != uid:
        raise PermissionError(f"Rejecting weight cache connection from uid {peer_uid}")


def _hash_quant_config(quant_config: Any) -> str:
    if quant_config is None:
        return ""
    if hasattr(quant_config, "to_dict"):
        quant_config = quant_config.to_dict()
    payload = json.dumps(quant_config, sort_keys=True, default=str)
    return safe_hash(payload.encode(), usedforsecurity=False).hexdigest()


@dataclass(frozen=True)
class CacheConfig:
    """Fingerprint of the cached weights.

    Any mismatch between the daemon's and the engine's fingerprint means the
    cached weights cannot be reused and the engine must load from disk.
    """

    model: str
    model_arch: str
    tp_size: int
    tp_rank: int
    dtype: str
    quantization: str | None
    quant_config_hash: str
    revision: str | None
    vllm_version: str

    @classmethod
    def from_model_config(
        cls, model_config: ModelConfig, tp_size: int, tp_rank: int
    ) -> "CacheConfig":
        """Build the fingerprint for a model configuration.

        Must be called before weight loading: process_weights_after_loading
        may mutate hf_config.quantization_config, which would change the hash
        between the daemon and the engine.
        """
        hf_config = model_config.hf_config
        arch = ",".join(getattr(hf_config, "architectures", None) or [])
        quant_config = getattr(hf_config, "quantization_config", None)
        return cls(
            model=model_config.model,
            model_arch=arch,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dtype=str(model_config.dtype),
            quantization=model_config.quantization,
            quant_config_hash=_hash_quant_config(quant_config),
            revision=model_config.revision,
            vllm_version=vllm.version.__version__,
        )

    def mismatched_fields(self, other: "CacheConfig") -> list[str]:
        return [
            f.name
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        ]


@dataclass
class TensorEntry:
    """A single cached tensor.

    CUDA tensors are exported as `torch.multiprocessing` reduction args
    (CUDA IPC handles); non-CUDA tensors are shipped by value.
    """

    kind: str
    """Either "param" or "buffer"."""
    ipc_args: tuple | None = None
    cpu_tensor: torch.Tensor | None = None

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, kind: str) -> "TensorEntry":
        from torch.multiprocessing.reductions import reduce_tensor

        tensor = tensor.detach()
        if tensor.is_cuda:
            _, ipc_args = reduce_tensor(tensor)
            return cls(kind=kind, ipc_args=ipc_args)
        return cls(kind=kind, cpu_tensor=tensor.cpu())

    def rebuild(self, device_index: int) -> torch.Tensor:
        if self.ipc_args is None:
            assert self.cpu_tensor is not None
            return self.cpu_tensor
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        args = list(self.ipc_args)
        # Index 6 of the args from reduce_tensor is the device index. It must
        # be retargeted to the local index since the daemon and the engine may
        # have different CUDA_VISIBLE_DEVICES mappings.
        args[6] = device_index
        return rebuild_cuda_tensor(*args)


def send_msg(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_LEN_STRUCT.pack(len(payload)))
    sock.sendall(payload)


def recv_msg(sock: socket.socket) -> Any:
    (length,) = _LEN_STRUCT.unpack(_recv_exact(sock, _LEN_STRUCT.size))
    if length > MAX_MSG_SIZE:
        raise ValueError(f"Message size {length} exceeds limit {MAX_MSG_SIZE}")
    return pickle.loads(_recv_exact(sock, length))


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving message")
        buf.extend(chunk)
    return bytes(buf)
