# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WeightCacheKey fingerprinting and socket protocol for the weight cache daemon.

The protocol uses pickle over a Unix domain socket and is only intended for
communication between trusted local processes owned by the same user. The
sockets live in a per-user private directory (mode 0700) and the daemon
restricts the socket file permissions to the owner (0600). Both the daemon and
the engine verify that the directory and socket are owned by the current user
and are not group/world accessible before trusting them, so a different local
user cannot pre-plant a malicious socket at a predictable path.
"""

import glob
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
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

import vllm.version
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
)
from vllm.platforms import current_platform
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)

SOCKET_NAME_TEMPLATE = "vllm_weight_cache_{gpu_uuid}.sock"
SOCKET_DIR_TEMPLATE = "vllm_weight_cache_{uid}"

_LEN_STRUCT = struct.Struct("!Q")
# Sanity bound for a single pickled message. IPC handles are tiny; tensor
# data itself is streamed separately via send_tensor/recv_tensor_into.
MAX_MSG_SIZE = 1 << 34
# Sanity bound for a single streamed tensor payload (send_tensor/
# recv_tensor_into). Individual weights can be multi-GB (e.g. embedding
# tables), so this is much larger than MAX_MSG_SIZE.
MAX_TENSOR_SIZE = 1 << 40


def _current_uid() -> int:
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else -1


class WeightCacheUnavailableError(Exception):
    """Raised when no weight cache daemon is reachable or usable."""


class CacheConfigMismatchError(Exception):
    """Raised when the daemon's cached weights don't match the engine."""


class UnsupportedQuantForIPCError(Exception):
    """Raised when a quantization method is not verified for IPC weight sharing."""


class UnsupportedPlatformForIPCError(Exception):
    """Raised when the current platform cannot share CUDA IPC handles."""


def check_ipc_platform_support() -> None:
    """Hard-error unless the current platform can use the weight cache.

    CUDA/ROCm tensors get a real IPC handle from ``TensorEntry``; XPU has no
    such handle yet and ships tensors by value instead. Other platforms are
    rejected outright.

    Raises:
        UnsupportedPlatformForIPCError: If the current platform is not
            CUDA, ROCm, or XPU.
    """
    if current_platform.is_cuda_alike() or current_platform.is_xpu():
        return
    raise UnsupportedPlatformForIPCError(
        f"platform {current_platform.device_name!r} does not support the "
        "weight cache; only CUDA, ROCm and XPU are supported. Use the "
        "default --load-format for this platform."
    )


# The daemon transfers post processed weights directly
def check_ipc_quant_support(model: torch.nn.Module) -> None:
    """Hard-error unless every quant method supports pre-processed weights.

    Args:
        model: The model to inspect (weights need not be loaded).

    Raises:
        UnsupportedQuantForIPCError: If any quant method does not declare
            ``supports_pre_processed_weights``.
    """
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if (
            isinstance(quant_method, QuantizeMethodBase)
            and not quant_method.supports_pre_processed_weights
        ):
            raise UnsupportedQuantForIPCError(
                f"layer {name or '<root>'}: {type(quant_method).__name__} "
                "does not support loading from pre-processed weights."
            )


def get_current_device_uuid() -> str:
    """UUID of the physical GPU backing the current accelerator device."""
    return current_platform.get_device_uuid(torch.accelerator.current_device_index())


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


def get_socket_path(gpu_uuid: str, socket_dir: str | None = None) -> str:
    directory = get_socket_dir(socket_dir)
    return os.path.join(directory, SOCKET_NAME_TEMPLATE.format(gpu_uuid=gpu_uuid))


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


def _safetensors_header(path: str) -> bytes:
    """Return the raw safetensors header (length prefix + JSON) of a file.

    The header carries tensor names, dtypes, shapes and byte offsets, so it is
    a content fingerprint of the shard without reading any weight bytes.
    """
    with open(path, "rb") as f:
        size_bytes = f.read(8)
        (header_len,) = struct.unpack("<Q", size_bytes)
        return size_bytes + f.read(header_len)


def hash_checkpoint(model: str) -> str | None:
    """Fingerprint checkpoint content from local safetensors metadata.

    Hashes each shard's safetensors header so a daemon and an engine pointing
    at identical weights in different directories produce the same key. Returns
    None when local safetensors files can't be located (e.g. an undownloaded
    Hugging Face repo id), leaving the caller to fall back to the model path.
    """
    if not os.path.isdir(model):
        return None
    files = glob.glob(os.path.join(model, "*.safetensors"))
    if os.path.isfile(os.path.join(model, SAFE_WEIGHTS_INDEX_NAME)):
        files = filter_duplicate_safetensors_files(
            files, model, SAFE_WEIGHTS_INDEX_NAME
        )
    if not files:
        return None
    hasher = safe_hash(b"", usedforsecurity=False)
    for path in sorted(files, key=os.path.basename):
        hasher.update(os.path.basename(path).encode())
        hasher.update(_safetensors_header(path))
    return hasher.hexdigest()


@dataclass(frozen=True)
class WeightCacheKey:
    """Fingerprint of the cached weights.

    Any mismatch between the daemon's and the engine's fingerprint means the
    cached weights cannot be reused and the engine must load from disk.
    """

    checkpoint: str
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
    ) -> "WeightCacheKey":
        """Build the fingerprint for a model configuration.

        Must be called before weight loading: process_weights_after_loading
        may mutate hf_config.quantization_config, which would change the hash
        between the daemon and the engine.

        The checkpoint is identified by a hash of its safetensors metadata when
        the weights are available locally, so a daemon and engine referencing
        identical weights in different directories still match; otherwise it
        falls back to the model path.
        """
        hf_config = model_config.hf_config
        arch = ",".join(getattr(hf_config, "architectures", None) or [])
        quant_config = getattr(hf_config, "quantization_config", None)
        checkpoint = hash_checkpoint(model_config.model) or model_config.model
        return cls(
            checkpoint=checkpoint,
            model_arch=arch,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dtype=str(model_config.dtype),
            quantization=model_config.quantization,
            quant_config_hash=_hash_quant_config(quant_config),
            revision=model_config.revision,
            vllm_version=vllm.version.__version__,
        )

    def mismatched_fields(self, other: "WeightCacheKey") -> list[str]:
        return [
            f.name
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        ]


def _try_export_ipc(tensor: torch.Tensor) -> tuple | None:
    """Best-effort export of a real device IPC handle for ``tensor``.

    CUDA/ROCm always get one via ``reduce_tensor``; a failure there is a real
    bug and propagates. Other accelerators have no such handle yet, so
    ``reduce_tensor`` silently falls back to a non-IPC rebuild function
    instead -- both that and an outright exception mean "no IPC support",
    returning ``None`` so the caller ships the tensor by value.
    """
    if tensor.device.type == "cpu":
        return None
    if tensor.is_cuda:
        _, args = reduce_tensor(tensor)
        return args
    try:
        rebuild_func, args = reduce_tensor(tensor)
    except Exception:
        logger.debug(
            "Device IPC export not supported for %s tensors; shipping by value instead",
            tensor.device.type,
        )
        return None
    if rebuild_func is not rebuild_cuda_tensor:
        return None
    return args


@dataclass
class TensorEntry:
    """A single cached tensor.

    Tensors with a real device IPC handle (CUDA/ROCm today) are exported as
    `torch.multiprocessing` reduction args. Everything else is shipped by
    value instead: metadata is sent inline via
    ``stream_shape``/``stream_dtype``, and the raw bytes follow separately
    via ``send_tensor``/``recv_tensor_into``.
    """

    kind: str
    """Either "param" or "buffer"."""
    ipc_args: tuple | None = None
    cpu_tensor: torch.Tensor | None = None
    stream_shape: torch.Size | None = None
    stream_dtype: torch.dtype | None = None

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, kind: str
    ) -> tuple["TensorEntry", torch.Tensor | None]:
        """Build the entry for ``tensor``.

        Returns the entry, plus the raw CPU tensor to stream with
        ``send_tensor`` right after it when there's no IPC handle for
        ``tensor`` (``None`` otherwise).
        """
        tensor = tensor.detach()
        ipc_args = _try_export_ipc(tensor)
        if ipc_args is not None:
            return cls(kind=kind, ipc_args=ipc_args), None
        cpu_tensor = tensor.cpu()
        entry = cls(
            kind=kind, stream_shape=cpu_tensor.shape, stream_dtype=cpu_tensor.dtype
        )
        return entry, cpu_tensor

    def rebuild(self, device_index: int) -> torch.Tensor:
        if self.ipc_args is None:
            assert self.cpu_tensor is not None
            device = torch.device(current_platform.device_type, device_index)
            if device.type == "cpu":
                return self.cpu_tensor
            # No IPC handle: copy in via a pinned staging buffer for faster
            # H2D bandwidth. A no-op since cpu_tensor is already pinned.
            return self.cpu_tensor.pin_memory().to(device, non_blocking=True)
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


def send_tensor(sock: socket.socket, tensor: torch.Tensor) -> None:
    """Send a CPU tensor's raw bytes, with no pickling of the data.

    Pairs with a ``recv_tensor_into`` call for a tensor of the same
    shape/dtype on the other end, right after the message that describes it.
    """
    view = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
    sock.sendall(_LEN_STRUCT.pack(view.nbytes))
    sock.sendall(memoryview(view))


def recv_tensor_into(sock: socket.socket, buffer: torch.Tensor) -> None:
    """Receive raw tensor bytes sent by ``send_tensor`` straight into ``buffer``.

    ``buffer`` must be contiguous and already sized to match what was sent
    (e.g. a pinned staging tensor allocated from the sender's advertised
    shape/dtype); reading directly into it combines the receive with the
    host-memory pinning into a single step.
    """
    (length,) = _LEN_STRUCT.unpack(_recv_exact(sock, _LEN_STRUCT.size))
    if length > MAX_TENSOR_SIZE:
        raise ValueError(
            f"Tensor payload size {length} exceeds limit {MAX_TENSOR_SIZE}"
        )
    view = buffer.reshape(-1).view(torch.uint8).numpy()
    if length != view.nbytes:
        raise ValueError(
            f"Tensor payload size {length} does not match expected buffer "
            f"size {view.nbytes}"
        )
    mv = memoryview(view)
    total = 0
    while total < length:
        n = sock.recv_into(mv[total:], length - total)
        if n == 0:
            raise ConnectionError("Socket closed while receiving tensor payload")
        total += n


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving message")
        buf.extend(chunk)
    return bytes(buf)
