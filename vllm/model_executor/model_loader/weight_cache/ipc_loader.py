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
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfigMismatchError,
    TensorEntry,
    UnsupportedQuantForIPCError,
    WeightCacheKey,
    WeightCacheUnavailableError,
    check_ipc_quant_support,
    get_physical_device_id,
    get_socket_path,
    recv_msg,
    send_msg,
    verify_socket_owner,
)
from vllm.model_executor.utils import weights_already_processed
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
    - connect_timeout_s: socket connect timeout (default: 5.0).
    - state_timeout_s: timeout for the weight-transfer request (default: 300.0).

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
        self.connect_timeout_s: float = float(
            extra_config.pop("connect_timeout_s", _CONNECT_TIMEOUT_S)
        )
        self.state_timeout_s: float = float(
            extra_config.pop("state_timeout_s", _STATE_TIMEOUT_S)
        )
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
        device_index = torch.accelerator.current_device_index()
        entries, _ = self._fetch_entries(model_config)
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
        # Unsupported quantization is a permanent misconfiguration rather than a
        # transient daemon outage, so it is raised even when fallback is on.
        self._check_supported(vllm_config, model_config)
        state_fetched = False
        try:
            entries, aliases = self._fetch_entries(model_config)
            state_fetched = True
            return self._build_model(
                vllm_config, model_config, prefix, entries, aliases
            )
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
            # _build_model failed after fetching state without reaching its
            # copy-mode release, so the daemon still holds the full cache;
            # release it so the disk fallback does not OOM against it.
            if state_fetched and self.mode == "copy":
                self._send_release()
            torch.accelerator.empty_cache()
        return self._fallback_load(vllm_config, model_config, prefix)

    def _build_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str,
        entries: dict[str, TensorEntry],
        aliases: dict[str, str],
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
            else torch.accelerator.current_device_index()
        )
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            self._apply_entries(model, entries, aliases, device_index)
            # The daemon exports tensors that already went through
            # process_weights_after_loading; re-run it in pre-processed mode
            # so quant methods only rebuild Python-side state (e.g. the MoE
            # kernel). Leftovers are materialized afterwards so that
            # placeholders the daemon-side post-processing consumed are
            # dropped rather than filled with uninitialized memory.
            with weights_already_processed():
                process_weights_after_loading(model, model_config, target_device)
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
        return model.eval()

    @staticmethod
    def _check_supported(vllm_config: VllmConfig, model_config: ModelConfig) -> None:
        check_ipc_quant_support(model_config, where="engine")
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype != "auto" and not str(cache_dtype).startswith("fp8"):
            # BaseKVCacheMethod.process_weights_after_loading turns the loaded
            # k/v scale parameters into plain float attributes. For fp8 cache
            # dtypes those are rebuilt from the exported scale buffers when
            # process_weights_after_loading runs in pre-processed mode; other
            # quantized cache dtypes are not verified.
            raise UnsupportedQuantForIPCError(
                f"[weight_cache:engine] kv cache dtype {cache_dtype!r} is not "
                "supported by the weight cache; use --kv-cache-dtype auto."
            )

    def _apply_entries(
        self,
        model: nn.Module,
        entries: dict[str, TensorEntry],
        aliases: dict[str, str],
        device_index: int,
    ) -> None:
        # remove_duplicate=False keeps tied module aliases reachable by name:
        # a tied lm_head *is* the embedding module, so the deduplicated view
        # would not contain "lm_head" at all.
        modules = dict(model.named_modules(remove_duplicate=False))
        registered: dict[str, torch.Tensor] = {}

        def _register(name: str, tensor: torch.Tensor, is_param: bool) -> None:
            module_name, _, leaf = name.rpartition(".")
            module = modules.get(module_name)
            if module is None:
                raise RuntimeError(f"Cached tensor {name} has no matching module")
            # Replace via registration rather than param.data assignment,
            # which fails for meta tensors. Entries may also introduce
            # post-quantization tensors absent from the meta model.
            module._parameters.pop(leaf, None)
            module._buffers.pop(leaf, None)
            if is_param:
                obj: torch.Tensor = (
                    tensor
                    if isinstance(tensor, nn.Parameter)
                    else nn.Parameter(tensor, requires_grad=False)
                )
                module.register_parameter(leaf, obj)
            else:
                obj = tensor
                module.register_buffer(leaf, obj)
            registered[name] = obj

        for name, entry in entries.items():
            tensor = entry.rebuild(device_index)
            if self.mode == "copy":
                tensor = tensor.clone()
            _register(name, tensor, entry.kind == "param")

        # Re-establish tied-weight aliases by registering the *same* object the
        # canonical name resolved to, so parameter identity (and the tie) is
        # preserved instead of allocating uninitialized memory.
        for alias_name, canonical_name in aliases.items():
            obj = registered.get(canonical_name)
            if obj is None:
                logger.warning(
                    "Cached alias %s references missing canonical tensor %s",
                    alias_name,
                    canonical_name,
                )
                continue
            _register(alias_name, obj, isinstance(obj, nn.Parameter))

    def _fetch_entries(
        self, model_config: ModelConfig
    ) -> tuple[dict[str, TensorEntry], dict[str, str]]:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        cache_config = WeightCacheKey.from_model_config(
            model_config,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
        )
        return self._request_state(cache_config)

    def _request_state(
        self, cache_config: WeightCacheKey
    ) -> tuple[dict[str, TensorEntry], dict[str, str]]:
        with self._connect(self.state_timeout_s) as conn:
            send_msg(conn, {"cmd": "get_state", "cache_config": cache_config})
            response = recv_msg(conn)
        status = response.get("status")
        if status == "mismatch":
            raise CacheConfigMismatchError(
                f"WeightCacheKey mismatch on fields: {response.get('fields')}"
            )
        if status != "ok":
            raise WeightCacheUnavailableError(
                f"Weight cache daemon error: {response.get('message')}"
            )
        self._check_gpu_uuid(response.get("gpu_uuid"))
        return response["entries"], response.get("aliases", {})

    def _connect(self, timeout: float) -> socket.socket:
        socket_path = self._resolve_socket_path()
        # The auto-derived per-user directory is locked to 0700 and checked
        # strictly. When the operator explicitly configures a path they own the
        # trust decision, so only ownership/symlink safety is enforced.
        strict_perms = self.socket_path is None and self.socket_dir is None
        try:
            verify_socket_owner(socket_path, strict_perms=strict_perms)
        except OSError as e:
            raise WeightCacheUnavailableError(
                f"Weight cache socket {socket_path} is unavailable: {e}"
            ) from e
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
        device_index = torch.accelerator.current_device_index()
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
        props = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        )
        local_uuid = str(props.uuid)
        if daemon_uuid != local_uuid:
            raise CacheConfigMismatchError(
                f"Daemon GPU {daemon_uuid} != engine GPU {local_uuid}; "
                "check the socket path / GPU mapping"
            )

    def _send_release(self) -> None:
        try:
            with self._connect(self.connect_timeout_s) as conn:
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
