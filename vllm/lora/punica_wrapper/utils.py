# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping, LoRARouteMapping


def compute_meta(
    token_lora_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, bool]:
    """Get the information required for the sgmv kernel. With the  features:
    1. If consecutive requests in the batch use the same LoRA, this function
    will combine them into a single request, improving sgmv kernel inference
    performance.
    2. At the beginning of each prefill stage inference, recalculations are
    needed based on the input, but only once.
    """
    lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(
        token_lora_tensor, return_counts=True
    )
    cum_result = torch.cumsum(seq_length_tensor, dim=0)
    b_seq_start_tensor = torch.zeros_like(seq_length_tensor)
    b_seq_start_tensor[1:].copy_(cum_result[:-1])
    max_length = seq_length_tensor.max().item()
    token_nums = seq_length_tensor.sum().item()
    batch_size = lora_indices_tensor.size(0)
    no_lora = False
    # -1 means no lora should be applied. Use `no_lora` to determine whether
    # the current step requires LoRA. If LoRA is not needed, the prefill stage
    # does not need to launch the triton kernel, which can improve performance
    if batch_size == 1 and lora_indices_tensor == -1:
        no_lora = True
    return (
        b_seq_start_tensor,
        seq_length_tensor,
        lora_indices_tensor,
        batch_size,
        max_length,
        token_nums,
        no_lora,
    )


# TODO see if this can be vectorized
def convert_mapping(
    mapping: "LoRAMapping",
    lora_index_to_id: list[int | None],
    max_loras: int,
    vocab_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Converts LoRAMapping to index tensors.

    Args:
        mapping: LoRAMapping mapping rows in a batch to LoRA ids.
        lora_index_to_id: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        device: Device the returned index tensors are created on.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                LoRA indices for sampler. For generation, this will be the
                same as base_indices. For prefill, this will map requests
                to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with padding.
                Same as sampler_indices, but -1 is replaced with
                max_loras.
            embeddings_indices: Tensor of shape [batch_size] mapping
                requests to the row offset of their LoRA.lora_a embeddings.
            indices_len: List of lengths of the above tensors. It contains
                (base_indices, sampler_indices, sampler_indices_padded,
                embeddings_indices).

    """
    device = torch.device(device)
    index_mapping_indices: list[int] = list(mapping.index_mapping).copy()
    embedding_indices = index_mapping_indices.copy()
    lora_indices = index_mapping_indices.copy()

    # Build a reverse lookup (LoRA id -> index) once instead of repeatedly
    # calling list.index(), which is an O(num_loras) linear scan performed for
    # every prompt token and every batch token on each step.
    lora_id_to_index = {
        lora_id: index
        for index, lora_id in enumerate(lora_index_to_id)
        if lora_id is not None
    }

    prompt_mapping: list[int] = [
        lora_id_to_index[x] if x > 0 else -1 for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(index_mapping_indices)):
        lora_idx = (
            lora_id_to_index[index_mapping_indices[i]]
            if index_mapping_indices[i] > 0
            else -1
        )
        embedding_indices[i] = lora_idx if index_mapping_indices[i] > 0 else 0
        lora_indices[i] = lora_idx

    indices_list: list[list[int] | torch.Tensor] = [
        index_mapping_indices,
        lora_indices,
        embedding_indices,
    ]

    if device.type == "cpu":
        indices = torch.tensor(indices_list, dtype=torch.long, device=device)
        prompt_mapping_tensor = torch.tensor(
            prompt_mapping, dtype=torch.long, device=device
        )
    else:
        indices = async_tensor_h2d(indices_list, dtype=torch.long, device=device)
        prompt_mapping_tensor = async_tensor_h2d(
            prompt_mapping, dtype=torch.long, device=device
        )
    embeddings_indices = indices[2] * vocab_size
    base_indices = indices[1]
    sampler_indices = prompt_mapping_tensor
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded = torch.where(
        sampler_indices_padded == -1, max_loras - 1, sampler_indices_padded
    )
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), device=device, dtype=torch.long
    ) + (sampler_indices_padded * len(sampler_indices_padded))

    # Contain length of indices tensors. Used to index into each tensor.
    indices_len = [
        base_indices.shape[-1],
        sampler_indices.shape[-1],
        sampler_indices_padded.shape[-1],
        embeddings_indices.shape[-1],
    ]

    return (
        base_indices,
        sampler_indices,
        sampler_indices_padded,
        embeddings_indices,
        indices_len,
    )


class RoutedLoRATensors(NamedTuple):
    """Per-step routed LoRA metadata.

    Tokens that use the same resident slot on every route form a plan. Fused
    kernels process each plan's tokens together, so all routes of a token are
    applied in one pass.
    """

    # [T, K] resident slot per token and route; -1 when the route is absent.
    token_lora_indices: torch.Tensor
    # [T, K] FP32 mixture coefficients.
    token_lora_weights: torch.Tensor
    # [P, K] resident slot per plan and route; -1 when the route is absent.
    plan_lora_indices: torch.Tensor
    # [N] indices of tokens with at least one route, grouped by plan.
    plan_token_indices: torch.Tensor
    # [P + 1] offsets into plan_token_indices.
    plan_token_start_loc: torch.Tensor

    @staticmethod
    def make(
        max_tokens: int, max_plans: int, max_routes: int, device: torch.device | str
    ) -> "RoutedLoRATensors":
        return RoutedLoRATensors(
            torch.full((max_tokens, max_routes), -1, dtype=torch.long, device=device),
            torch.zeros((max_tokens, max_routes), dtype=torch.float32, device=device),
            torch.full((max_plans, max_routes), -1, dtype=torch.int32, device=device),
            torch.zeros(max_tokens, dtype=torch.int32, device=device),
            torch.zeros(max_plans + 1, dtype=torch.int32, device=device),
        )

    def copy_(self, source: "RoutedLoRATensors") -> None:
        """Update captured buffers, clearing routes and plans from the last step."""
        tokens, routes = source.token_lora_indices.shape
        plans = source.plan_lora_indices.size(0)
        active_tokens = source.plan_token_indices.numel()
        if (
            tokens > self.token_lora_indices.size(0)
            or routes > self.token_lora_indices.size(1)
            or plans > self.plan_lora_indices.size(0)
        ):
            raise ValueError("Routed LoRA metadata exceeds its reserved capacity")
        self.token_lora_indices.fill_(-1)
        self.token_lora_weights.zero_()
        self.plan_lora_indices.fill_(-1)
        self.plan_token_start_loc.fill_(active_tokens)
        self.token_lora_indices[:tokens, :routes].copy_(source.token_lora_indices)
        self.token_lora_weights[:tokens, :routes].copy_(source.token_lora_weights)
        self.plan_lora_indices[:plans, :routes].copy_(source.plan_lora_indices)
        self.plan_token_indices[:active_tokens].copy_(source.plan_token_indices)
        self.plan_token_start_loc[: plans + 1].copy_(source.plan_token_start_loc)


def make_route_plans(
    token_lora_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Groups tokens by their slot tuple; tokens without any route are dropped.

    Returns:
        Plan slots [P, K], grouped token indices [N] and offsets [P + 1].

    """
    active_tokens = np.flatnonzero((token_lora_indices >= 0).any(axis=1))
    plans, inverse = np.unique(
        token_lora_indices[active_tokens], axis=0, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(plans))
    starts = np.concatenate(([0], np.cumsum(counts)))
    return (
        plans.astype(np.int32),
        active_tokens[order].astype(np.int32),
        starts.astype(np.int32),
    )


def convert_route_mapping(
    mapping: "LoRARouteMapping",
    lora_index_to_id: list[int | None],
    device: torch.device | str,
) -> RoutedLoRATensors:
    """Converts routed LoRA ids to slot indices and per-plan token groups."""
    device = torch.device(device)
    lora_id_to_index = {
        lora_id: index
        for index, lora_id in enumerate(lora_index_to_id)
        if lora_id is not None
    }
    missing_lora_ids = sorted(
        {
            lora_id
            for token_lora_ids in mapping.token_lora_ids
            for lora_id in token_lora_ids
            if lora_id > 0 and lora_id not in lora_id_to_index
        }
    )
    if missing_lora_ids:
        raise ValueError(
            f"Routed LoRA mapping references unloaded LoRA ids: {missing_lora_ids}"
        )

    token_lora_indices = np.array(
        [
            [
                lora_id_to_index[lora_id] if lora_id > 0 else -1
                for lora_id in token_lora_ids
            ]
            for token_lora_ids in mapping.token_lora_ids
        ],
        dtype=np.int64,
    )
    token_lora_weights = np.array(mapping.token_lora_weights, dtype=np.float32)
    plans = make_route_plans(token_lora_indices)

    def to_device(array: np.ndarray) -> torch.Tensor:
        if device.type == "cpu":
            return torch.from_numpy(array)
        return async_tensor_h2d(array, device=device)

    return RoutedLoRATensors(
        to_device(token_lora_indices),
        to_device(token_lora_weights),
        *(to_device(array) for array in plans),
    )
