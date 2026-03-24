# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import torch

_hidden_states_cumsum_cache: dict[
    int, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]
] = {}


def get_hidden_states_cumsum(hidden_states: torch.Tensor) -> torch.Tensor:
    key = id(hidden_states)
    cached = _hidden_states_cumsum_cache.get(key)
    if cached is not None:
        ref, cumsum = cached
        if ref() is hidden_states:
            return cumsum

    cumsum = torch.cumsum(hidden_states, dim=0, dtype=torch.float32)

    def _cleanup(_ref: weakref.ReferenceType[torch.Tensor]) -> None:
        _hidden_states_cumsum_cache.pop(key, None)

    _hidden_states_cumsum_cache[key] = (
        weakref.ref(hidden_states, _cleanup),
        cumsum,
    )
    return cumsum
