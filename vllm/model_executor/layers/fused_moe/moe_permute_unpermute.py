# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import torch

from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


@dataclass
class MoEPermuteScratch:
    """Reused metadata buffers for repeated grouped-MoE permutes.

    Capacity follows the largest input seen, like the shared workspace: it grows
    during warmup and is fixed once the workspace manager is locked.
    """

    num_experts: int
    num_local_experts: int
    device: torch.device
    max_expanded_rows: int = field(init=False, default=0)
    token_expert_indices: torch.Tensor = field(init=False)
    expert_first_token_offset: torch.Tensor = field(init=False)
    permuted_idx: torch.Tensor = field(init=False)
    inv_permuted_idx: torch.Tensor = field(init=False)
    sort_workspace: torch.Tensor = field(init=False)
    permuted_experts_id: torch.Tensor = field(init=False)
    sorted_row_idx: torch.Tensor = field(init=False)
    topk_ids_int32: torch.Tensor = field(init=False)
    topk_ids_for_sort: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        assert self.num_experts > 0
        assert self.num_local_experts > 0
        self.expert_first_token_offset = torch.empty(
            self.num_local_experts + 1, dtype=torch.int64, device=self.device
        )
        # torch.device("cuda") in config, after initialized,
        # will be changed to cuda:{index}, so we need to refresh here.
        self.device = self.expert_first_token_offset.device

    def reserve(self, expanded_rows: int) -> None:
        """Ensure capacity for ``expanded_rows`` (num_tokens * topk) rows."""
        if expanded_rows <= self.max_expanded_rows:
            return
        if (
            is_workspace_manager_initialized()
            and current_workspace_manager().is_locked()
        ):
            raise AssertionError(
                f"Workspace is locked but MoE permute scratch requires "
                f"{expanded_rows} rows, current capacity is "
                f"{self.max_expanded_rows} rows. Scratch growth is not allowed "
                "after locking."
            )

        def empty_rows() -> torch.Tensor:
            return torch.empty(expanded_rows, dtype=torch.int32, device=self.device)

        self.token_expert_indices = torch.arange(
            expanded_rows, dtype=torch.int32, device=self.device
        )
        self.permuted_idx = empty_rows()
        self.inv_permuted_idx = empty_rows()
        self.permuted_experts_id = empty_rows()
        self.sorted_row_idx = empty_rows()
        self.topk_ids_int32 = empty_rows()
        self.topk_ids_for_sort = empty_rows()
        sorter_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            expanded_rows, self.num_experts
        )
        self.sort_workspace = torch.empty(
            sorter_size, dtype=torch.int8, device=self.device
        )
        self.max_expanded_rows = expanded_rows

    def prepare_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if topk_ids.dtype == torch.int32:
            return topk_ids
        numel = topk_ids.numel()
        topk_ids_int32 = self.topk_ids_int32[:numel].view_as(topk_ids)
        topk_ids_int32.copy_(topk_ids)
        return topk_ids_int32


def moe_prepare_scatter(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    scratch: MoEPermuteScratch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate expert offsets and shared scatter/unpermute destination indices."""
    assert topk_ids.device == scratch.device
    n_token, topk = topk_ids.shape
    expanded_rows = topk_ids.numel()
    scratch.reserve(expanded_rows)
    inverse = scratch.inv_permuted_idx[:expanded_rows].view(n_token, topk)
    if expanded_rows == 0:
        scratch.expert_first_token_offset.zero_()
        return scratch.expert_first_token_offset, inverse
    torch.ops._moe_C.moe_prepare_scatter(
        scratch.prepare_topk_ids(topk_ids),
        scratch.token_expert_indices[:expanded_rows].view(n_token, topk),
        expert_map,
        scratch.num_experts,
        scratch.num_local_experts,
        scratch.expert_first_token_offset,
        inverse,
        scratch.sort_workspace,
        scratch.permuted_experts_id[:expanded_rows].view(n_token, topk),
        scratch.sorted_row_idx[:expanded_rows].view(n_token, topk),
        scratch.topk_ids_for_sort[:expanded_rows].view(n_token, topk),
    )
    return scratch.expert_first_token_offset, inverse


def get_moe_permute_scratch(
    *,
    num_experts: int,
    num_local_experts: int,
    device: torch.device,
) -> MoEPermuteScratch:
    """Share scratch across sequential layers in the current ubatch and lane.

    Without a workspace manager, allocate new scratch for each call.
    """

    def create_scratch() -> MoEPermuteScratch:
        return MoEPermuteScratch(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            device=device,
        )

    if not is_workspace_manager_initialized():
        return create_scratch()

    key = (MoEPermuteScratch, num_experts, num_local_experts, device)
    return current_workspace_manager().get_persistent_resource(key, create_scratch)


def moe_permute(
    hidden_states: torch.Tensor,
    a1q_scale: torch.Tensor | None,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int = -1,
    expert_map: torch.Tensor | None = None,
    permuted_hidden_states: torch.Tensor | None = None,
    scratch: MoEPermuteScratch | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """This function expands and permutes activation to gather uncontinuous tokens
      for each expert.

    Args:
        hidden_states (torch.Tensor): The input tensor to the MoE layer.
        a1q_scale (Optional[torch.Tensor]): quant scale for hidden_states
        topk_ids (torch.Tensor): topk expert route id for each token.
        n_expert (int): The number of expert.
        n_local_expert (int): The number of expert in current EP rank.
        expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
            from the global expert space to the local expert space of the expert
            parallel shard.
        permuted_hidden_states (Optional[torch.Tensor]): Optional output tensor.
            If None, the output tensor will be created in this function.
        scratch (Optional[MoEPermuteScratch]): Optional reusable metadata
            buffers. Otherwise they are allocated in this function.

    Returns:
    - permuted_hidden_states (torch.Tensor): permuted activation.
    - a1q_scale (Optional[torch.Tensor]): permuted quant scale for hidden_states
        if original scale not per-tensor scaling
    - expert_first_token_offset (torch.Tensor): offset of the first token
       of each expert for standard grouped gemm.
    - inv_permuted_idx (torch.Tensor): idx map for moe_unpermute.
    - permuted_idx (torch.Tensor): idx map from hidden to permuted_hidden.

    """
    n_token, n_hidden = hidden_states.size()
    topk = topk_ids.size(1)
    assert (n_hidden * hidden_states.element_size()) % 16 == 0, (
        "permue kernel need hidden dim align to 16B"
    )
    permuted_row_size = n_token * topk
    if n_local_expert == -1:
        n_local_expert = n_expert
    if permuted_hidden_states is None:
        permuted_hidden_states = torch.empty(
            (permuted_row_size, n_hidden),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    assert permuted_hidden_states.size() == (permuted_row_size, n_hidden), (
        f"Expected permuted hidden states to be {(permuted_row_size, n_hidden)}"
        f" but got {permuted_hidden_states.size()}"
    )

    if scratch is None:
        token_expert_indices = torch.arange(
            0, n_token * topk, dtype=torch.int32, device=hidden_states.device
        ).reshape((n_token, topk))

        expert_first_token_offset = torch.empty(
            n_local_expert + 1, dtype=torch.int64, device=hidden_states.device
        )
        permuted_idx = torch.full(
            (permuted_row_size,),
            n_token * topk,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        inv_permuted_idx = torch.empty(
            (n_token, topk), dtype=torch.int32, device=hidden_states.device
        )
        topk_ids_int32 = topk_ids.to(torch.int32)
        torch.ops._moe_C.moe_permute(
            hidden_states,
            topk_ids_int32,
            token_expert_indices,
            expert_map,
            n_expert,
            n_local_expert,
            topk,
            permuted_hidden_states,
            expert_first_token_offset,
            inv_permuted_idx,
            permuted_idx,
        )
    else:
        assert hidden_states.device == scratch.device
        assert topk_ids.device == scratch.device
        assert topk_ids.size(0) == n_token
        assert n_expert == scratch.num_experts
        assert n_local_expert == scratch.num_local_experts
        scratch.reserve(permuted_row_size)
        token_expert_indices = scratch.token_expert_indices[:permuted_row_size].view(
            n_token, topk
        )
        expert_first_token_offset = scratch.expert_first_token_offset
        permuted_idx = scratch.permuted_idx[:permuted_row_size]
        permuted_idx.fill_(permuted_row_size)
        inv_permuted_idx = scratch.inv_permuted_idx[:permuted_row_size].view(
            n_token, topk
        )
        permuted_experts_id = scratch.permuted_experts_id[:permuted_row_size].view(
            n_token, topk
        )
        sorted_row_idx = scratch.sorted_row_idx[:permuted_row_size].view(n_token, topk)
        topk_ids_for_sort = scratch.topk_ids_for_sort[:permuted_row_size].view(
            n_token, topk
        )
        topk_ids_int32 = scratch.prepare_topk_ids(topk_ids)
        torch.ops._moe_C.moe_permute_with_scratch(
            hidden_states,
            topk_ids_int32,
            token_expert_indices,
            expert_map,
            n_expert,
            n_local_expert,
            topk,
            permuted_hidden_states,
            expert_first_token_offset,
            inv_permuted_idx,
            permuted_idx,
            scratch.sort_workspace,
            permuted_experts_id,
            sorted_row_idx,
            topk_ids_for_sort,
        )

    if a1q_scale is not None and a1q_scale.dim() > 1:
        a1q_scale = a1q_scale[permuted_idx.clamp(max=n_token * topk - 1) // topk]
    return (
        permuted_hidden_states,
        a1q_scale,
        expert_first_token_offset,
        inv_permuted_idx.flatten(),
        permuted_idx,
    )


def moe_unpermute(
    out: torch.Tensor,
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None = None,
) -> None:
    """This function expands and permutes activation to gathering uncontinuous
      tokens for each expert.

    Args:
        out (torch.Tensor): output tensor
        permuted_hidden_states (torch.Tensor): permuted activation.
        topk_weights (torch.Tensor): topk expert route weight for each token.
        inv_permuted_idx (torch.Tensor): row idx map for moe_unpermute.
        expert_first_token_offset (Optional[torch.Tensor]): offset of the first
            token of each expert for grouped gemm.

    Returns:
    - hidden_states (torch.Tensor): The reduced and unpermuted activation
      tensor.

    """
    topk = topk_weights.size(1)
    n_hidden = permuted_hidden_states.size(-1)
    assert (n_hidden * permuted_hidden_states.element_size()) % 16 == 0, (
        "unpermue kernel need hidden dim align to 16B"
    )

    torch.ops._moe_C.moe_unpermute(
        permuted_hidden_states,
        topk_weights,
        inv_permuted_idx,
        expert_first_token_offset,
        topk,
        out,
    )


def moe_permute_unpermute_supported():
    return torch.ops._moe_C.moe_permute_unpermute_supported()
