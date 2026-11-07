# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from typing import Protocol

import msgspec
import torch

from vllm.lora.layers.utils import LoRAMapping, LoRARouteMapping
from vllm.lora.request import (
    LoRARequest,
    LoRARequestLike,
    LoRARoutingRequest,
    is_routed_lora_request,
    iter_lora_requests,
)
from vllm.utils.platform_utils import is_pin_memory_available

NO_LORA_ID = 0


@dataclass(frozen=True)
class DistributedLoRAPlan:
    """Batch-local identities shared by ranks, independent of physical slots."""

    requests: tuple[LoRARequest, ...]

    @classmethod
    def from_rank_requests(
        cls, ranks: Iterable[tuple[int, tuple[LoRARequest, ...]]]
    ) -> "DistributedLoRAPlan":
        requests: dict[int, LoRARequest] = {}
        descriptors: dict[int, bytes] = {}
        names: dict[str, int] = {}
        capacities = []
        for capacity, rank_requests in ranks:
            capacities.append(capacity)
            for request in rank_requests:
                if request.load_inplace:
                    raise ValueError("Distributed routed LoRA rejects load_inplace")
                adapter_id = request.lora_int_id
                descriptor = msgspec.json.encode(request)
                if descriptors.setdefault(adapter_id, descriptor) != descriptor:
                    raise ValueError(
                        f"Conflicting LoRA descriptors for ID {adapter_id}"
                    )
                if names.setdefault(request.lora_name, adapter_id) != adapter_id:
                    raise ValueError(f"Conflicting LoRA IDs for {request.lora_name!r}")
                descriptors[adapter_id] = descriptor
                requests[adapter_id] = request
        if not capacities or min(capacities) < len(requests):
            raise ValueError(
                f"Distributed LoRA union ({len(requests)} adapters) exceeds "
                f"participating rank capacities {capacities}"
            )
        return cls(tuple(requests[key] for key in sorted(requests)))

    def slot_tables(self, slot_ids: list[int | None]) -> tuple[list[int], list[int]]:
        """Return slot-to-batch and batch-to-slot tables; batch index 0 is base."""
        indices = {
            request.lora_int_id: i + 1 for i, request in enumerate(self.requests)
        }
        slots = {adapter_id: i for i, adapter_id in enumerate(slot_ids)}
        missing = indices.keys() - slots.keys()
        if missing:
            raise ValueError(
                f"Distributed LoRA adapters are not active: {sorted(missing)}"
            )
        return (
            [
                indices.get(adapter_id, 0) if adapter_id is not None else 0
                for adapter_id in slot_ids
            ],
            [-1] + [slots[request.lora_int_id] for request in self.requests],
        )


class DistributedLoRATensors:
    """Stable dispatch identities and destination lookup, staged before replay."""

    def __init__(
        self, tokens: int, routes: int, slots: int, device: torch.device | str
    ):
        self.indices = torch.zeros((tokens, routes), dtype=torch.int32, device=device)
        self.weights = torch.zeros((tokens, routes), dtype=torch.float32, device=device)
        self.local_slots = torch.full(
            (slots + 1,), -1, dtype=torch.int32, device=device
        )

    def update(
        self,
        plan: DistributedLoRAPlan,
        mapping: LoRAMapping | LoRARouteMapping,
        slot_ids: list[int | None],
    ) -> None:
        _, local_slots = plan.slot_tables(slot_ids)
        identities = {r.lora_int_id: i + 1 for i, r in enumerate(plan.requests)}
        if isinstance(mapping, LoRARouteMapping):
            token_ids, weights = mapping.token_lora_ids, mapping.token_lora_weights
        else:
            token_ids = tuple((i,) for i in mapping.index_mapping)
            weights = tuple((float(i > 0),) for i in mapping.index_mapping)
        if len(token_ids) > self.indices.shape[0]:
            raise ValueError("Distributed LoRA token buffer capacity exceeded")
        indices_cpu = torch.zeros(self.indices.shape, dtype=torch.int32, device="cpu")
        weights_cpu = torch.zeros(self.weights.shape, dtype=torch.float32, device="cpu")
        if token_ids:
            width = len(token_ids[0])
            if width > self.indices.shape[1]:
                raise ValueError("Distributed LoRA route buffer capacity exceeded")
            missing = {i for ids in token_ids for i in ids if i > 0} - identities.keys()
            if missing:
                raise ValueError(
                    f"LoRA IDs absent from distributed plan: {sorted(missing)}"
                )
            indices_cpu[: len(token_ids), :width] = torch.tensor(
                [[identities.get(i, 0) for i in ids] for ids in token_ids],
                dtype=torch.int32,
                device="cpu",
            )
            weights_cpu[: len(token_ids), :width] = torch.tensor(
                weights, dtype=torch.float32, device="cpu"
            )
            weights_cpu.masked_fill_(indices_cpu == 0, 0)
        slots_cpu = torch.full(
            self.local_slots.shape, -1, dtype=torch.int32, device="cpu"
        )
        slots_cpu[: len(local_slots)] = torch.tensor(
            local_slots, dtype=torch.int32, device="cpu"
        )
        for destination, source in (
            (self.indices, indices_cpu),
            (self.weights, weights_cpu),
            (self.local_slots, slots_cpu),
        ):
            if destination.device.type != "cpu" and is_pin_memory_available():
                # A fresh snapshot cannot be overwritten by the next CPU step.
                source = source.pin_memory()
            destination.copy_(source, non_blocking=True)


class LoRARequestMapping(Protocol):
    def __setitem__(self, index: int, value: int, /) -> None: ...


def has_routed_lora(
    lora_requests: Iterable[LoRARequestLike | None],
) -> bool:
    return any(is_routed_lora_request(lora_request) for lora_request in lora_requests)


def get_lora_route(
    lora_request: LoRARequestLike | None,
    route_k: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if route_k < 1:
        raise ValueError("route_k must be greater than 0")

    if isinstance(lora_request, LoRARoutingRequest):
        lora_ids = [request.lora_int_id for request in lora_request.lora_requests]
        lora_weights = list(lora_request.lora_weights)
    else:
        concrete_lora_requests = iter_lora_requests(lora_request)
        lora_ids = [request.lora_int_id for request in concrete_lora_requests]
        lora_weights = [1.0] * len(lora_ids)

    padding = route_k - len(lora_ids)
    if padding < 0:
        raise ValueError("route_k cannot be smaller than the route length")

    lora_ids.extend([NO_LORA_ID] * padding)
    lora_weights.extend([0.0] * padding)
    return tuple(lora_ids), tuple(lora_weights)


def get_lora_route_k(
    lora_requests: Iterable[LoRARequestLike | None],
) -> int:
    route_k = 1
    for lora_request in lora_requests:
        route_k = max(route_k, len(iter_lora_requests(lora_request)))
    return route_k


def make_lora_route_mapping(
    lora_requests: Iterable[LoRARequestLike | None],
    num_scheduled_tokens: Iterable[int],
    prompt_mapping: tuple[int, ...],
    is_prefill: bool = True,
) -> LoRARouteMapping | None:
    lora_requests = tuple(lora_requests)
    num_scheduled_tokens = tuple(int(num_tokens) for num_tokens in num_scheduled_tokens)
    if len(lora_requests) != len(num_scheduled_tokens):
        raise ValueError(
            "lora_requests and num_scheduled_tokens must have the same length"
        )
    if sum(num_scheduled_tokens) == 0:
        return None

    route_k = get_lora_route_k(lora_requests)
    token_lora_ids: list[tuple[int, ...]] = []
    token_lora_weights: list[tuple[float, ...]] = []

    for lora_request, num_tokens in zip(lora_requests, num_scheduled_tokens):
        lora_ids, lora_weights = get_lora_route(lora_request, route_k)
        token_lora_ids.extend([lora_ids] * num_tokens)
        token_lora_weights.extend([lora_weights] * num_tokens)

    return LoRARouteMapping(
        token_lora_ids=tuple(token_lora_ids),
        token_lora_weights=tuple(token_lora_weights),
        prompt_mapping=prompt_mapping,
        is_prefill=is_prefill,
    )


def add_lora_request(
    request_lora_mapping: LoRARequestMapping,
    request_lora_requests: MutableMapping[int, LoRARequestLike],
    lora_id_to_request_ids: MutableMapping[int, set[str]],
    lora_id_to_lora_request: MutableMapping[int, LoRARequest],
    req_index: int,
    req_id: str,
    lora_request: LoRARequestLike | None,
) -> None:
    if lora_request is None:
        request_lora_mapping[req_index] = NO_LORA_ID
        return

    request_lora_requests[req_index] = lora_request
    if isinstance(lora_request, LoRARequest):
        request_lora_mapping[req_index] = lora_request.lora_int_id
    else:
        request_lora_mapping[req_index] = NO_LORA_ID

    for concrete_lora_request in iter_lora_requests(lora_request):
        lora_id = concrete_lora_request.lora_int_id
        lora_req_ids = lora_id_to_request_ids.setdefault(lora_id, set())
        lora_req_ids.add(req_id)
        lora_id_to_lora_request[lora_id] = concrete_lora_request


def remove_lora_request(
    request_lora_mapping: LoRARequestMapping,
    request_lora_requests: MutableMapping[int, LoRARequestLike],
    lora_id_to_request_ids: MutableMapping[int, set[str]],
    lora_id_to_lora_request: MutableMapping[int, LoRARequest],
    req_index: int,
    req_id: str,
) -> None:
    lora_request = request_lora_requests.pop(req_index, None)
    if lora_request is None:
        request_lora_mapping[req_index] = NO_LORA_ID
        return

    for concrete_lora_request in iter_lora_requests(lora_request):
        lora_id = concrete_lora_request.lora_int_id
        lora_req_ids = lora_id_to_request_ids[lora_id]
        lora_req_ids.discard(req_id)
        if not lora_req_ids:
            del lora_id_to_request_ids[lora_id]
            del lora_id_to_lora_request[lora_id]

    request_lora_mapping[req_index] = NO_LORA_ID
