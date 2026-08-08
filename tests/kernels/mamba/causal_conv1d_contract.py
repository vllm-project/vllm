# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@dataclass(frozen=True)
class ContractCase:
    name: str
    lengths: tuple[int, ...]
    dim: int = 256
    width: int = 4
    dtype: torch.dtype = torch.bfloat16
    activation: str | None = "silu"
    has_bias: bool = True
    channel_contiguous: bool = True
    initial_mask: tuple[bool, ...] | None = None
    padded_sequences: int = 0
    family: str = "generic"

    @property
    def active_sequences(self) -> int:
        return len(self.lengths) - self.padded_sequences


CONTRACT_CASES = (
    ContractCase("main_bf16_w4", (1024,), initial_mask=(False,), family="fast"),
    ContractCase("main_bf16_w4_initial", (1024,), initial_mask=(True,), family="fast"),
    ContractCase("biasless", (127,), has_bias=False, initial_mask=(True,)),
    ContractCase("activation_none", (127,), activation=None, initial_mask=(True,)),
    ContractCase("width2", (127,), width=2, initial_mask=(True,)),
    ContractCase("width3", (127,), width=3, initial_mask=(True,)),
    ContractCase("width5", (127,), width=5, initial_mask=(True,)),
    ContractCase("fp32", (127,), dtype=torch.float32, initial_mask=(True,)),
    ContractCase("shorter_than_state", (2,), width=5, initial_mask=(True,)),
    ContractCase(
        "token_contiguous",
        (127,),
        channel_contiguous=False,
        initial_mask=(True,),
    ),
    ContractCase(
        "varlen",
        (17, 31, 9),
        initial_mask=(True, False, True),
        family="varlen",
    ),
    ContractCase(
        "varlen_with_null_padding",
        (17, 31, 9, 13),
        initial_mask=(True, False, True, False),
        padded_sequences=1,
        family="varlen",
    ),
)

REQUIRED_FEATURE_FAMILIES = {
    "fast",
    "generic",
    "varlen",
    "continuous_batching",
    "pad_null",
    "apc_prefix_cache",
}


def build_case(case: ContractCase, device: torch.device) -> dict[str, object]:
    torch.manual_seed(0)
    total_tokens = sum(case.lengths)
    if case.channel_contiguous:
        backing = torch.randn(
            total_tokens,
            case.dim + 64,
            dtype=case.dtype,
            device=device,
        )
        x = backing[:, : case.dim].transpose(0, 1)
    else:
        x = torch.randn(
            case.dim, total_tokens, dtype=case.dtype, device=device
        ).contiguous()

    weight = torch.randn(case.dim, case.width, dtype=case.dtype, device=device)
    bias = (
        torch.randn(case.dim, dtype=case.dtype, device=device)
        if case.has_bias
        else None
    )
    pool_size = case.active_sequences + 4
    states = torch.randn(
        pool_size,
        case.width - 1,
        case.dim,
        dtype=case.dtype,
        device=device,
    ).transpose(1, 2)
    active_indices = torch.arange(
        1, case.active_sequences + 1, dtype=torch.int32, device=device
    )
    if case.padded_sequences:
        padding = torch.full(
            (case.padded_sequences,),
            NULL_BLOCK_ID,
            dtype=torch.int32,
            device=device,
        )
        indices = torch.cat((active_indices, padding))
    else:
        indices = active_indices

    initial_mask = case.initial_mask or tuple(False for _ in case.lengths)
    has_initial_state = torch.tensor(initial_mask, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor(
        (0, *torch.tensor(case.lengths).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "conv_states": states,
        "query_start_loc": query_start_loc,
        "cache_indices": indices,
        "has_initial_state": has_initial_state,
        "activation": case.activation,
    }


def reference_output_and_state(
    case: ContractCase, inputs: dict[str, object]
) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs["x"]
    weight = inputs["weight"]
    bias = inputs["bias"]
    states = inputs["conv_states"]
    indices = inputs["cache_indices"]
    has_initial_state = inputs["has_initial_state"]
    assert isinstance(x, torch.Tensor)
    assert isinstance(weight, torch.Tensor)
    assert bias is None or isinstance(bias, torch.Tensor)
    assert isinstance(states, torch.Tensor)
    states = states.clone()
    assert isinstance(indices, torch.Tensor)
    assert isinstance(has_initial_state, torch.Tensor)

    outputs: list[torch.Tensor] = []
    token_offset = 0
    state_len = case.width - 1
    for sequence, length in enumerate(case.lengths):
        sequence_x = x[:, token_offset : token_offset + length]
        token_offset += length
        state_index = int(indices[sequence].item())
        if state_index == NULL_BLOCK_ID:
            continue
        initial = (
            states[state_index]
            if bool(has_initial_state[sequence].item())
            else torch.zeros_like(states[state_index])
        )
        full_input = torch.cat((initial, sequence_x), dim=1)
        output = F.conv1d(
            full_input.unsqueeze(0).float(),
            weight.unsqueeze(1).float(),
            bias.float() if bias is not None else None,
            groups=case.dim,
        ).squeeze(0)
        if case.activation in ("silu", "swish"):
            output = F.silu(output)
        outputs.append(output.to(case.dtype))
        states[state_index] = full_input[:, -state_len:]

    return torch.cat(outputs, dim=1), states


def active_output(case: ContractCase, output: torch.Tensor) -> torch.Tensor:
    if case.padded_sequences == 0:
        return output
    active_tokens = sum(case.lengths[: case.active_sequences])
    return output[:, :active_tokens]
