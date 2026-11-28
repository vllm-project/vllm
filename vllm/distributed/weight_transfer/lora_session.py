# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side LoRA weight-update session.

Supports updating one or more LoRA adapters inside vLLM's unified
`start_weight_update` -> `update_weights` -> `finish_weight_update`
lifecycle from:
1. Weight-transfer backends (NCCL / CUDA IPC, packed or unpacked) via
   `_LoRAWeightSink` (which implements `nn.Module.load_weights`).
2. Disk directories (`lora_path`), committed on `finish_weight_update`.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import set_current_vllm_config
from vllm.lora.request import LoRARequest, _TensorLoRARequest


@dataclass(frozen=True)
class LoRAAdapterUpdateSpec:
    """Normalized specification for one LoRA adapter in a weight-update session."""

    lora_name: str
    lora_int_id: int
    lora_path: str
    peft_config: dict[str, Any] | None = None
    base_model_name: str | None = None
    is_3d_lora_weight: bool = False

    def __post_init__(self) -> None:
        if not self.lora_name or not self.lora_name.strip():
            raise ValueError("lora_name must be a non-empty string")
        if self.lora_int_id < 1:
            raise ValueError(f"lora_int_id must be >= 1, got {self.lora_int_id}")
        if not self.lora_path or not self.lora_path.strip():
            raise ValueError("lora_path must be a non-empty string")
        if self.peft_config is not None and not isinstance(self.peft_config, dict):
            raise ValueError("peft_config must be a dict when provided")

    @property
    def expects_tensors(self) -> bool:
        """True if tensors will be streamed rather than read from disk."""
        return self.peft_config is not None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoRAAdapterUpdateSpec":
        return cls(
            lora_name=str(data["lora_name"]),
            lora_int_id=int(data["lora_int_id"]),
            lora_path=str(data["lora_path"]),
            peft_config=data.get("peft_config"),
            base_model_name=data.get("base_model_name"),
            is_3d_lora_weight=bool(data.get("is_3d_lora_weight", False)),
        )


class _LoRAWeightSink(torch.nn.Module):
    """Minimal `nn.Module` target for `WeightTransferEngine.set_weight_update_target`.

    Collects streamed LoRA tensors into per-adapter CPU dicts. Because it has
    no parameters or quantized submodules, `start_reload` / `finish_reload` are
    no-ops, while both `IPCWeightTransferEngine` and `NCCLWeightTransferEngine`
    deliver unpacked tensors directly into `load_weights`.
    """

    def __init__(self, specs: Iterable[LoRAAdapterUpdateSpec]) -> None:
        super().__init__()
        self._all_adapters: dict[str, LoRAAdapterUpdateSpec] = {}
        self._tensor_adapters: dict[str, LoRAAdapterUpdateSpec] = {}
        self.stashed: dict[str, dict[str, torch.Tensor]] = {}
        for spec in specs:
            self._all_adapters[spec.lora_name] = spec
            if spec.expects_tensors:
                self._tensor_adapters[spec.lora_name] = spec
                self.stashed[spec.lora_name] = {}

    def _resolve_target(
        self,
        weight_name: str,
        default_lora_name: str | None = None,
    ) -> tuple[str, str]:
        if default_lora_name is not None:
            if default_lora_name not in self._tensor_adapters:
                if default_lora_name in self._all_adapters:
                    raise ValueError(
                        f"LoRA adapter {default_lora_name!r} was not configured "
                        "with peft_config for tensor updates"
                    )
                raise ValueError(
                    f"Unknown LoRA adapter {default_lora_name!r}; expected one of "
                    f"{sorted(self._tensor_adapters)}"
                )
            prefix = f"{default_lora_name}:"
            if weight_name.startswith(prefix):
                return default_lora_name, weight_name[len(prefix) :]
            return default_lora_name, weight_name

        if ":" in weight_name:
            prefix, rest = weight_name.split(":", 1)
            if prefix in self._tensor_adapters:
                return prefix, rest
            if prefix in self._all_adapters:
                raise ValueError(
                    f"LoRA adapter {prefix!r} was not configured with peft_config "
                    "for tensor updates"
                )
            if len(self._tensor_adapters) == 1:
                only_name = next(iter(self._tensor_adapters))
                return only_name, weight_name
            raise ValueError(
                f"Unknown LoRA adapter prefix {prefix!r} in weight {weight_name!r}; "
                f"expected one of {sorted(self._tensor_adapters)}"
            )

        if len(self._tensor_adapters) == 1:
            only_name = next(iter(self._tensor_adapters))
            return only_name, weight_name

        raise ValueError(
            f"Weight {weight_name!r} must be prefixed with '<lora_name>:' when "
            f"updating multiple LoRA adapters ({sorted(self._tensor_adapters)})"
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        default_lora_name: str | None = None,
    ) -> set[str]:
        loaded: set[str] = set()
        for name, tensor in weights:
            adapter_name, tensor_key = self._resolve_target(
                name, default_lora_name=default_lora_name
            )
            # Detach and copy to CPU immediately: IPC handles and packed NCCL
            # staging buffers may be reused as soon as load_weights returns.
            self.stashed[adapter_name][tensor_key] = tensor.detach().to(
                device="cpu", copy=True
            )
            loaded.add(name)
        return loaded


class LoRAWeightUpdateSession:
    """State and lifecycle driver for an active worker LoRA weight-update session."""

    def __init__(self, specs: list[LoRAAdapterUpdateSpec]) -> None:
        if not specs:
            raise ValueError("lora_adapters must not be empty")
        seen: set[str] = set()
        for spec in specs:
            if spec.lora_name in seen:
                raise ValueError(
                    f"Duplicate lora_name {spec.lora_name!r} in lora_adapters"
                )
            seen.add(spec.lora_name)
        self.specs = specs
        self.sink = _LoRAWeightSink(specs)
        self.uses_transfer_engine = False

    @classmethod
    def from_dicts(
        cls, adapters: Iterable[Mapping[str, Any]]
    ) -> "LoRAWeightUpdateSession":
        return cls([LoRAAdapterUpdateSpec.from_dict(item) for item in adapters])

    @property
    def has_tensor_adapters(self) -> bool:
        return any(spec.expects_tensors for spec in self.specs)

    @classmethod
    def start(
        cls, worker: Any, lora_adapters: list[dict[str, Any]]
    ) -> "LoRAWeightUpdateSession":
        """Start a LoRA weight-update session on `worker`."""
        if worker.lora_config is None:
            raise RuntimeError(
                "LoRA is not enabled on this worker. Start vLLM with --enable-lora."
            )
        if worker._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while a weight update is already "
                "active. Call finish_weight_update first."
            )
        session = cls.from_dicts(lora_adapters)
        worker._weight_update_lora_session = session
        worker._weight_update_active = True
        worker._weight_update_is_draft = False
        return session

    def _select_local_update_info(
        self, worker: Any, update_info: dict | list[dict]
    ) -> dict[str, Any]:
        if isinstance(update_info, list):
            parallel_config = worker.vllm_config.parallel_config
            return update_info[
                parallel_config.data_parallel_index * parallel_config.world_size
                + worker.rank
            ]
        return update_info

    def _end(self, worker: Any) -> None:
        worker._weight_update_lora_session = None
        worker._weight_update_active = False
        if self.uses_transfer_engine and worker.weight_transfer_engine is not None:
            worker.weight_transfer_engine.reset_weight_update_target()
        self.sink.stashed.clear()

    def update(self, worker: Any, update_info: dict | list[dict]) -> None:
        """Receive one LoRA weight update chunk on `worker`."""
        with set_current_vllm_config(worker.vllm_config):
            try:
                local_update_info = self._select_local_update_info(worker, update_info)
                worker._check_weight_transfer_engine()
                assert worker.weight_transfer_engine is not None
                if not self.has_tensor_adapters:
                    raise ValueError(
                        "Received weight-transfer tensors, but no adapters in "
                        "this session were configured with peft_config"
                    )
                if not self.uses_transfer_engine:
                    worker.weight_transfer_engine.set_weight_update_target(
                        self.sink, worker.model_config
                    )
                    worker.weight_transfer_engine.start_weight_update()
                    self.uses_transfer_engine = True
                worker.weight_transfer_engine.update_weights(local_update_info)
            except BaseException:
                self._end(worker)
                raise

    def finish(self, worker: Any) -> None:
        """Finalize transfer (if any) and atomically load all session adapters."""
        with set_current_vllm_config(worker.vllm_config):
            try:
                if self.uses_transfer_engine:
                    assert worker.weight_transfer_engine is not None
                    worker.weight_transfer_engine.finish_weight_update()
                for lora_request in self.build_lora_requests():
                    if not worker.model_runner.add_lora(lora_request):
                        raise RuntimeError(
                            f"Failed to load LoRA adapter {lora_request.lora_name!r}"
                        )
            finally:
                self._end(worker)

    def build_lora_requests(self) -> list[LoRARequest]:
        """Build worker-local `LoRARequest` / `_TensorLoRARequest` objects to commit."""
        requests: list[LoRARequest] = []
        for spec in self.specs:
            if spec.expects_tensors:
                tensors = self.sink.stashed.get(spec.lora_name)
                if not tensors:
                    raise ValueError(
                        f"No tensors received for LoRA adapter {spec.lora_name!r} "
                        "before finish_weight_update"
                    )
                requests.append(
                    _TensorLoRARequest(
                        lora_name=spec.lora_name,
                        lora_int_id=spec.lora_int_id,
                        lora_path=spec.lora_path,
                        base_model_name=spec.base_model_name,
                        load_inplace=True,
                        is_3d_lora_weight=spec.is_3d_lora_weight,
                        _lora_tensors=tensors,
                        _peft_config=spec.peft_config,
                    )
                )
            else:
                requests.append(
                    LoRARequest(
                        lora_name=spec.lora_name,
                        lora_int_id=spec.lora_int_id,
                        lora_path=spec.lora_path,
                        base_model_name=spec.base_model_name,
                        load_inplace=True,
                        is_3d_lora_weight=spec.is_3d_lora_weight,
                    )
                )
        return requests
