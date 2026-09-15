# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


class TrainableTokensBuffer(torch.nn.Module):
    """Fixed-size adapter buffers containing absolute replacement token rows."""

    def __init__(
        self,
        max_loras: int,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "token_ids",
            torch.full((max_loras, max_tokens), -1, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "weights",
            torch.zeros(
                (max_loras, max_tokens, hidden_size), dtype=dtype, device=device
            ),
            persistent=False,
        )

    def reset(self, index: int) -> None:
        self.token_ids[index].fill_(-1)

    def set(
        self, index: int, token_indices: torch.Tensor, weights: torch.Tensor
    ) -> None:
        num_tokens = token_indices.numel()
        if num_tokens > self.token_ids.shape[1]:
            raise ValueError("Trainable token count exceeds max_lora_trainable_tokens.")
        if token_indices.ndim != 1 or weights.shape != (
            num_tokens,
            self.weights.shape[2],
        ):
            raise ValueError(
                "Trainable token IDs and replacement rows have invalid shapes."
            )
        self.reset(index)
        self.weights[index, :num_tokens].copy_(weights, non_blocking=True)
        self.token_ids[index, :num_tokens].copy_(token_indices, non_blocking=True)


@triton.jit
def _replace_embeddings_kernel(
    output,
    tokens,
    adapter_indices,
    token_ids,
    weights,
    output_stride,
    output_hidden_stride,
    MAX_LORAS: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    token_idx = tl.program_id(0)
    adapter_idx = tl.load(adapter_indices + token_idx)
    if adapter_idx < 0 or adapter_idx >= MAX_LORAS:
        return
    token = tl.load(tokens + token_idx)
    offsets = tl.arange(0, BLOCK_TOKENS)
    ids = tl.load(
        token_ids + adapter_idx * MAX_TOKENS + offsets,
        offsets < MAX_TOKENS,
        other=-1,
    )
    row = tl.max(tl.where((ids == token) & (offsets < MAX_TOKENS), offsets, -1), 0)
    if row < 0:
        return
    hidden = tl.arange(0, BLOCK_HIDDEN)
    values = tl.load(
        weights + (adapter_idx * MAX_TOKENS + row) * HIDDEN_SIZE + hidden,
        hidden < HIDDEN_SIZE,
        other=0,
    )
    tl.store(
        output + token_idx * output_stride + hidden * output_hidden_stride,
        values,
        hidden < HIDDEN_SIZE,
    )


@triton.jit
def _replace_logits_kernel(
    output,
    hidden_states,
    adapter_indices,
    token_ids,
    weights,
    bias,
    output_stride,
    output_vocab_stride,
    hidden_stride,
    hidden_dim_stride,
    MAX_LORAS: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    token_idx = tl.program_id(0)
    replacement_idx = tl.program_id(1)
    adapter_idx = tl.load(adapter_indices + token_idx)
    if adapter_idx < 0 or adapter_idx >= MAX_LORAS:
        return
    replacement = adapter_idx * MAX_TOKENS + replacement_idx
    token = tl.load(token_ids + replacement)
    if token < 0:
        return
    hidden = tl.arange(0, BLOCK_HIDDEN)
    x = tl.load(
        hidden_states + token_idx * hidden_stride + hidden * hidden_dim_stride,
        hidden < HIDDEN_SIZE,
        other=0,
    ).to(tl.float32)
    w = tl.load(
        weights + replacement * HIDDEN_SIZE + hidden,
        hidden < HIDDEN_SIZE,
        other=0,
    ).to(tl.float32)
    value = tl.sum(x * w, 0)
    if HAS_BIAS:
        value += tl.load(bias + token).to(tl.float32)
    tl.store(output + token_idx * output_stride + token * output_vocab_stride, value)


def _replace_embeddings(
    output: torch.Tensor,
    tokens: torch.Tensor,
    adapter_indices: torch.Tensor,
    token_ids: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    if output.shape[0] == 0:
        return
    if output.is_cuda:
        _replace_embeddings_kernel[(output.shape[0],)](
            output,
            tokens,
            adapter_indices,
            token_ids,
            weights,
            output.stride(0),
            output.stride(1),
            token_ids.shape[0],
            token_ids.shape[1],
            output.shape[1],
            triton.next_power_of_2(token_ids.shape[1]),
            triton.next_power_of_2(output.shape[1]),
        )
        return
    for adapter_idx in range(token_ids.shape[0]):
        matches = (
            (tokens[:, None] == token_ids[adapter_idx][None, :])
            & (token_ids[adapter_idx][None, :] >= 0)
            & (adapter_indices[:, None] == adapter_idx)
        )
        token_rows, replacement_rows = matches.nonzero(as_tuple=True)
        output[token_rows] = weights[adapter_idx, replacement_rows].to(output.dtype)


def _replace_logits(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    adapter_indices: torch.Tensor,
    token_ids: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    if output.shape[0] == 0:
        return
    if output.is_cuda:
        _replace_logits_kernel[(output.shape[0], token_ids.shape[1])](
            output,
            hidden_states,
            adapter_indices,
            token_ids,
            weights,
            bias,
            output.stride(0),
            output.stride(1),
            hidden_states.stride(0),
            hidden_states.stride(1),
            token_ids.shape[0],
            token_ids.shape[1],
            hidden_states.shape[1],
            bias is not None,
            triton.next_power_of_2(hidden_states.shape[1]),
        )
        return
    for adapter_idx in range(token_ids.shape[0]):
        rows = (adapter_indices == adapter_idx).nonzero(as_tuple=True)[0]
        valid = token_ids[adapter_idx] >= 0
        ids = token_ids[adapter_idx, valid]
        values = F.linear(
            hidden_states[rows].float(),
            weights[adapter_idx, valid].float(),
            bias[ids].float() if bias is not None else None,
        )
        output[rows[:, None], ids[None, :]] = values.to(output.dtype)


direct_register_custom_op(
    op_name="lora_replace_token_embeddings",
    op_func=_replace_embeddings,
    mutates_args=["output"],
    fake_impl=lambda *args, **kwargs: None,
    dispatch_key="CompositeExplicitAutograd",
)
replace_token_embeddings = torch.ops.vllm.lora_replace_token_embeddings

direct_register_custom_op(
    op_name="lora_replace_token_logits",
    op_func=_replace_logits,
    mutates_args=["output"],
    fake_impl=lambda *args, **kwargs: None,
    dispatch_key="CompositeExplicitAutograd",
)
replace_token_logits = torch.ops.vllm.lora_replace_token_logits
