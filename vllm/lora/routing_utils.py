# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, MutableMapping
from typing import Protocol

from vllm.lora.layers.utils import LoRARouteMapping
from vllm.lora.request import (
    LoRARequest,
    LoRARequestLike,
    LoRARoutingRequest,
    is_routed_lora_request,
    iter_lora_requests,
)

NO_LORA_ID = 0


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
