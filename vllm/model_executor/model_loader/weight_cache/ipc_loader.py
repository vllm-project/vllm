# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC model loader: maps post-quantized weights from a local weight cache
daemon via CUDA IPC instead of loading from disk."""

import dataclasses
import socket
from copy import copy

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfig,
    CacheConfigMismatchError,
    TensorEntry,
    WeightCacheUnavailableError,
    get_physical_device_id,
    get_socket_path,
    recv_msg,
    send_msg,
)
from vllm.tracing import instrument
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

_CONNECT_TIMEOUT_S = 5.0
_STATE_TIMEOUT_S = 300.0


class IpcModelLoader(BaseModelLoader):
    """Loads a model by mapping the weight cache daemon's tensors via CUDA IPC.

    The model is initialized on the meta device and every parameter/buffer is
    replaced by the daemon's post-quantized tensor, so
    process_weights_after_loading is skipped entirely. In "zero_copy" mode the
    engine shares the daemon's GPU memory; in "copy" mode the tensors are
    cloned into engine-owned memory and the daemon is asked to release its
    cache afterwards.

    Extra config keys (via --model-loader-extra-config):

    - socket_path: explicit daemon socket path. Defaults to a per-GPU path
      derived from the physical GPU id.
    - socket_dir: directory containing the daemon sockets.
    - mode: "zero_copy" (default) or "copy".
    - fallback: fall back to disk loading when the daemon is unavailable or
      the fingerprints mismatch (default: True).

    Note: in zero-copy mode the weights live in the daemon's CUDA IPC
    allocations, so sleep mode (CuMemAllocator weight offloading) must not be
    used with this loader.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = copy(load_config.model_loader_extra_config or {})
        self.socket_path: str | None = extra_config.pop("socket_path", None)
        self.socket_dir: str | None = extra_config.pop("socket_dir", None)
        self.mode: str = extra_config.pop("mode", "zero_copy")
        self.fallback: bool = extra_config.pop("fallback", True)
        if self.mode not in ("zero_copy", "copy"):
            raise ValueError(
                f"Invalid weight cache mode {self.mode!r}, "
                "expected 'zero_copy' or 'copy'"
            )
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: {sorted(extra_config)}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        DefaultModelLoader(self._fallback_load_config()).download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Best-effort in-place reload for an already-initialized model.

        Copies daemon tensors into matching parameters/buffers. The model is
        expected to already be in the post-quantized layout (e.g. previously
        loaded through this loader).
        """
        device_index = torch.cuda.current_device()
        entries = self._fetch_entries(model_config)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        for name, entry in entries.items():
            target = params.get(name, buffers.get(name))
            source = entry.rebuild(device_index)
            if target is None or target.shape != source.shape:
                logger.warning("Skipping mismatched cached tensor %s", name)
                continue
            target.data.copy_(source)

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        try:
            entries = self._fetch_entries(model_config)
            return self._build_model(vllm_config, model_config, prefix, entries)
        except (WeightCacheUnavailableError, CacheConfigMismatchError) as e:
            if not self.fallback:
                raise
            logger.warning(
                "Weight cache unusable (%s); falling back to disk loading", e
            )
        except Exception:
            if not self.fallback:
                raise
            logger.exception(
                "Weight cache IPC loading failed; falling back to disk loading"
            )
            torch.cuda.empty_cache()
        return self._fallback_load(vllm_config, model_config, prefix)

    def _build_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str,
        entries: dict[str, TensorEntry],
    ) -> nn.Module:
        device_config = vllm_config.device_config
        load_device = (
            device_config.device
            if self.load_config.device is None
            else self.load_config.device
        )
        target_device = torch.device(load_device)
        device_index = (
            target_device.index
            if target_device.index is not None
            else torch.cuda.current_device()
        )
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            self._apply_entries(model, entries, device_index)
            _materialize_remaining_meta_tensors(
                model, torch.device(target_device.type, device_index)
            )
        if self.mode == "copy":
            self._send_release()
        logger.info(
            "Mapped %d tensors from the weight cache daemon (%s mode)",
            len(entries),
            self.mode,
        )
        # process_weights_after_loading is intentionally skipped: the daemon
        # exports the already-processed state.
        return model.eval()

    def _apply_entries(
        self,
        model: nn.Module,
        entries: dict[str, TensorEntry],
        device_index: int,
    ) -> None:
        modules = dict(model.named_modules())
        for name, entry in entries.items():
            module_name, _, leaf = name.rpartition(".")
            module = modules.get(module_name)
            if module is None:
                raise RuntimeError(f"Cached tensor {name} has no matching module")
            tensor = entry.rebuild(device_index)
            if self.mode == "copy":
                tensor = tensor.clone()
            # Replace via registration rather than param.data assignment,
            # which fails for meta tensors. Entries may also introduce
            # post-quantization tensors absent from the meta model.
            module._parameters.pop(leaf, None)
            module._buffers.pop(leaf, None)
            if entry.kind == "param":
                module.register_parameter(
                    leaf, nn.Parameter(tensor, requires_grad=False)
                )
            else:
                module.register_buffer(leaf, tensor)

    def _fetch_entries(self, model_config: ModelConfig) -> dict[str, TensorEntry]:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        cache_config = CacheConfig.from_model_config(
            model_config,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
        )
        return self._request_state(cache_config)

    def _request_state(self, cache_config: CacheConfig) -> dict[str, TensorEntry]:
        with self._connect(_STATE_TIMEOUT_S) as conn:
            send_msg(conn, {"cmd": "get_state", "cache_config": cache_config})
            response = recv_msg(conn)
        status = response.get("status")
        if status == "mismatch":
            raise CacheConfigMismatchError(
                f"CacheConfig mismatch on fields: {response.get('fields')}"
            )
        if status != "ok":
            raise WeightCacheUnavailableError(
                f"Weight cache daemon error: {response.get('message')}"
            )
        self._check_gpu_uuid(response.get("gpu_uuid"))
        return response["entries"]

    def _connect(self, timeout: float) -> socket.socket:
        socket_path = self._resolve_socket_path()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(socket_path)
        except OSError as e:
            sock.close()
            raise WeightCacheUnavailableError(
                f"Cannot connect to weight cache daemon at {socket_path}: {e}"
            ) from e
        return sock

    def _resolve_socket_path(self) -> str:
        if self.socket_path is not None:
            return self.socket_path
        device_index = torch.cuda.current_device()
        gpu_id = get_physical_device_id(device_index)
        if gpu_id is None:
            raise WeightCacheUnavailableError(
                "Cannot infer the physical GPU id from CUDA_VISIBLE_DEVICES; "
                "pass socket_path via --model-loader-extra-config"
            )
        return get_socket_path(gpu_id, self.socket_dir)

    def _check_gpu_uuid(self, daemon_uuid: str | None) -> None:
        if daemon_uuid is None:
            return
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        local_uuid = str(props.uuid)
        if daemon_uuid != local_uuid:
            raise CacheConfigMismatchError(
                f"Daemon GPU {daemon_uuid} != engine GPU {local_uuid}; "
                "check the socket path / GPU mapping"
            )

    def _send_release(self) -> None:
        try:
            with self._connect(_CONNECT_TIMEOUT_S) as conn:
                send_msg(conn, {"cmd": "release"})
                recv_msg(conn)
        except (WeightCacheUnavailableError, ConnectionError, OSError):
            logger.warning("Failed to ask the weight cache daemon to release")

    def _fallback_load_config(self) -> LoadConfig:
        # DefaultModelLoader must not see load_format="ipc_cache" or the ipc
        # extra config keys.
        return dataclasses.replace(
            self.load_config, load_format="auto", model_loader_extra_config={}
        )

    def _fallback_load(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str
    ) -> nn.Module:
        loader = DefaultModelLoader(self._fallback_load_config())
        return loader.load_model(
            vllm_config=vllm_config, model_config=model_config, prefix=prefix
        )


def _materialize_remaining_meta_tensors(model: nn.Module, device: torch.device) -> None:
    """Allocate any tensors the daemon did not provide.

    These are typically parameters removed on the daemon side by
    process_weights_after_loading; they are not expected to be read at
    runtime, so they are left uninitialized.
    """
    for module_name, module in model.named_modules():
        for leaf, param in list(module._parameters.items()):
            if param is not None and param.device.type == "meta":
                logger.warning(
                    "Materializing empty parameter %s.%s missing from the weight cache",
                    module_name,
                    leaf,
                )
                module._parameters[leaf] = nn.Parameter(
                    torch.empty_like(param, device=device), requires_grad=False
                )
        for leaf, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[leaf] = torch.empty_like(buffer, device=device)
