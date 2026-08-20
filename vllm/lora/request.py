# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import TypeAlias

import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    Request for a LoRA adapter.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.

    load_inplace: If True, forces reloading the adapter even if one
        with the same lora_int_id already exists in the cache. This replaces
        the existing adapter in-place. If False (default), only loads if the
        adapter is not already loaded.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
    """Whether this adapter's MoE weights are stored in the 3D fused
    `gate_up_proj` / `down_proj` layout (one fused tensor per layer) or the
    2D per-expert split layout (separate `gate_proj` / `up_proj` / `down_proj`
    tensors per expert). Only consulted when the engine is started with
    `enable_mixed_moe_lora_format=True`; otherwise it is ignored and the
    on-disk format is inferred from the base model."""

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        # Ensure lora_path is not empty
        assert self.lora_path, "lora_path cannot be empty"

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare LoRARequest
        instances based on lora_name. This allows for identification
        and comparison lora adapter across engines.
        """
        return isinstance(value, self.__class__) and self.lora_name == value.lora_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash LoRARequest instances
        based on lora_name. This ensures that LoRARequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.lora_name)


class LoRARoutingRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """Internal request for fixed-weight routed mixtures of LoRA adapters."""

    routing_name: str
    routing_int_id: int
    lora_requests: tuple[LoRARequest, ...]
    lora_weights: tuple[float, ...]
    routing_strategy: str = "fixed"

    def __post_init__(self):
        self.lora_requests = tuple(self.lora_requests)
        self.lora_weights = tuple(self.lora_weights)

        if self.routing_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.routing_int_id}")
        if not self.routing_name:
            raise ValueError("routing_name cannot be empty")
        if self.routing_strategy != "fixed":
            raise ValueError(
                "Only fixed-weight routed LoRA requests are supported for now."
            )
        if not self.lora_requests:
            raise ValueError("lora_requests cannot be empty")
        if len(self.lora_requests) != len(self.lora_weights):
            raise ValueError("lora_requests and lora_weights must have the same length")

        lora_ids = [request.lora_int_id for request in self.lora_requests]
        if len(lora_ids) != len(set(lora_ids)):
            raise ValueError("lora_requests cannot contain duplicate lora_int_id")

        if any(not math.isfinite(weight) for weight in self.lora_weights):
            raise ValueError("lora_weights must be finite")
        if any(weight < 0 for weight in self.lora_weights):
            raise ValueError("lora_weights must be non-negative")
        if sum(self.lora_weights) <= 0:
            raise ValueError("At least one lora weight must be positive")

    @property
    def adapter_id(self):
        return self.routing_int_id

    @property
    def name(self):
        return self.routing_name

    @property
    def lora_name(self):
        return self.routing_name

    @property
    def top_k(self):
        return len(self.lora_requests)

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, self.__class__)
            and self.routing_name == value.routing_name
        )

    def __hash__(self) -> int:
        return hash(self.routing_name)


def is_routed_lora_request(
    lora_request: LoRARequest | LoRARoutingRequest | None,
) -> bool:
    return isinstance(lora_request, LoRARoutingRequest)


def iter_lora_requests(
    lora_request: LoRARequest | LoRARoutingRequest | None,
) -> tuple[LoRARequest, ...]:
    if lora_request is None:
        return ()
    if isinstance(lora_request, LoRARoutingRequest):
        return lora_request.lora_requests
    return (lora_request,)


def iter_lora_int_ids(
    lora_request: LoRARequest | LoRARoutingRequest | None,
) -> tuple[int, ...]:
    return tuple(request.lora_int_id for request in iter_lora_requests(lora_request))


LoRARequestLike: TypeAlias = LoRARequest | LoRARoutingRequest
