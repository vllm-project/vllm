# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AMD dispatch for the Kimi-K3 MLA output gate."""

import functools
from collections.abc import Callable

import torch

_AITER_GATE_MODULE = "aiter.ops.flydsl.kimi_k3_mla_gate"
_AiterGate = Callable[..., torch.Tensor]
_AiterGateSupport = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    bool,
]


@functools.lru_cache(maxsize=1)
def _get_aiter_mla_gate() -> tuple[_AiterGate, _AiterGateSupport] | None:
    try:
        from aiter.ops.flydsl.kimi_k3_mla_gate import (
            kimi_k3_mla_gate,
            supports_kimi_k3_mla_gate,
        )
    except ModuleNotFoundError as error:
        if not _AITER_GATE_MODULE.startswith(error.name or ""):
            raise
        return None
    return kimi_k3_mla_gate, supports_kimi_k3_mla_gate


def kimi_k3_mla_output_gate(
    hidden_states: torch.Tensor,
    attention_output: torch.Tensor,
    gate_projection: torch.nn.Module,
) -> torch.Tensor:
    """Use the fused gfx950 gate when supported, otherwise preserve PyTorch."""

    aiter_gate = _get_aiter_mla_gate()
    gate_weight = getattr(gate_projection, "weight", None)
    if aiter_gate is not None and gate_weight is not None:
        gate, supports_gate = aiter_gate
        if supports_gate(hidden_states, gate_weight, attention_output):
            return gate(
                hidden_states,
                gate_weight,
                attention_output,
                out=attention_output,
            )

    projected_gate = gate_projection(hidden_states)[0]
    return attention_output * projected_gate.sigmoid()
