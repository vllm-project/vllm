# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal `nn.Module` sink and HTTP safetensors codec for streaming one LoRA."""

from collections.abc import Iterable, Mapping
from typing import Any

import pybase64 as base64
import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from vllm.lora.request import _TensorLoRARequest


def encode_lora_tensors_http(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Encode a mapping of LoRA tensors as a JSON-safe HTTP `update_info` dict.

    Uses `safetensors` serialization (safe, no pickle) + base64.
    """
    if not tensors:
        raise ValueError("tensors must not be empty")

    # safetensors requires contiguous CPU tensors with non-aliased storage.
    cpu_tensors: dict[str, torch.Tensor] = {}
    seen_data_ptrs: set[int] = set()
    for name, tensor in tensors.items():
        t = tensor.detach().to(device="cpu").contiguous()
        if t.data_ptr() in seen_data_ptrs:
            t = t.clone()
        seen_data_ptrs.add(t.data_ptr())
        cpu_tensors[name] = t

    raw = safetensors_save(cpu_tensors)
    return {"safetensors_b64": base64.b64encode(raw).decode("ascii")}


def decode_lora_tensors_http(
    update_info: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Decode an HTTP `update_info` payload produced by `encode_lora_tensors_http`."""
    raw_b64 = update_info["safetensors_b64"]
    if not isinstance(raw_b64, str) or not raw_b64:
        raise ValueError("'safetensors_b64' must be a non-empty base64 string")
    return safetensors_load(base64.b64decode(raw_b64))


class LoRAWeightSink(torch.nn.Module):
    """Collects streamed LoRA tensors for `set_weight_update_target`.

    Because this module has no parameters or quantized submodules, `start_reload`
    and `finish_reload` are no-ops, while `IPCWeightTransferEngine` and
    `NCCLWeightTransferEngine` deliver tensors straight into `load_weights`.
    """

    def __init__(
        self,
        *,
        lora_name: str,
        lora_int_id: int,
        lora_path: str,
        peft_config: dict[str, Any],
        base_model_name: str | None = None,
        is_3d_lora_weight: bool = False,
    ) -> None:
        super().__init__()
        if not lora_name or not lora_name.strip():
            raise ValueError("lora_name must be a non-empty string")
        if lora_int_id < 1:
            raise ValueError(f"lora_int_id must be >= 1, got {lora_int_id}")
        if not lora_path or not lora_path.strip():
            raise ValueError("lora_path must be a non-empty string")
        if not isinstance(peft_config, dict):
            raise ValueError("peft_config must be a dict")
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path
        self.peft_config = peft_config
        self.base_model_name = base_model_name
        self.is_3d_lora_weight = is_3d_lora_weight
        self.tensors: dict[str, torch.Tensor] = {}
        self.uses_transfer_engine = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoRAWeightSink":
        return cls(
            lora_name=str(data["lora_name"]),
            lora_int_id=int(data["lora_int_id"]),
            lora_path=str(data["lora_path"]),
            peft_config=data["peft_config"],
            base_model_name=data.get("base_model_name"),
            is_3d_lora_weight=bool(data.get("is_3d_lora_weight", False)),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        for name, tensor in weights:
            # Copy to CPU immediately: IPC handles and packed NCCL staging
            # buffers may be reused as soon as load_weights returns.
            self.tensors[name] = tensor.detach().to(device="cpu", copy=True)
            loaded.add(name)
        return loaded

    def load_http_tensors(self, update_info: Mapping[str, Any]) -> bool:
        """If `update_info` carries HTTP safetensors, decode and stash them."""
        if "safetensors_b64" not in update_info:
            return False
        self.load_weights(decode_lora_tensors_http(update_info).items())
        return True

    def to_lora_request(self) -> _TensorLoRARequest:
        if not self.tensors:
            raise ValueError(
                f"No tensors received for LoRA adapter {self.lora_name!r} "
                "before finish_weight_update"
            )
        return _TensorLoRARequest(
            lora_name=self.lora_name,
            lora_int_id=self.lora_int_id,
            lora_path=self.lora_path,
            base_model_name=self.base_model_name,
            load_inplace=True,
            is_3d_lora_weight=self.is_3d_lora_weight,
            _lora_tensors=self.tensors,
            _peft_config=self.peft_config,
        )
