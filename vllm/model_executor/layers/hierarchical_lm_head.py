# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent centroid routing with compiled and MoE scoring backends."""

from pathlib import Path
from typing import Literal

import torch
from torch import nn


def _sorted_expert_assignment(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group routed row IDs by cluster and pad each group to a GEMM tile.

    All buffer sizes depend on shapes, not tensor values, so sorting can be
    captured in a CUDA graph. Unlike the CUDA align op, this supports C4096.
    """
    flat = topk_ids.reshape(-1)
    num_routes = flat.numel()
    experts, row_ids = flat.sort(stable=True)
    boundaries = torch.searchsorted(
        experts,
        torch.arange(num_experts + 1, device=flat.device, dtype=flat.dtype),
    )
    counts = boundaries[1:] - boundaries[:-1]
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    padded_ends = padded_counts.cumsum(0)
    padded_starts = padded_ends - padded_counts
    positions = (
        padded_starts[experts.long()]
        + torch.arange(num_routes, device=flat.device)
        - boundaries[experts.long()]
    )
    max_rows = min(num_routes * block_size, num_routes + num_experts * (block_size - 1))
    max_rows = ((max_rows + block_size - 1) // block_size) * block_size
    sorted_ids = torch.full(
        (max_rows,), num_routes, dtype=torch.int32, device=flat.device
    )
    sorted_ids.scatter_(0, positions, row_ids.to(torch.int32))
    block_offsets = (
        torch.arange(max_rows // block_size, device=flat.device) * block_size
    )
    expert_ids = torch.searchsorted(padded_ends, block_offsets, right=True).to(
        torch.int32
    )
    return sorted_ids, expert_ids, padded_ends[-1:].to(torch.int32)


def _grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
    routing: Literal["auto", "unsorted", "sorted"] = "unsorted",
) -> torch.Tensor:
    """Single grouped GEMM: (T, K) x (E, N, K)[topk_ids] -> (T, top_k, N).

    Borrows the first half of `fused_experts`: no activation, no second
    GEMM, no routing-weight reduction. The kernel writes rows at the
    original flat (token, k) offset, so no unpermute is needed.

    The only place in this file that touches a private fused_moe API.
    """
    import triton.language as tl

    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
        dispatch_fused_moe_kernel,
    )

    compute_type = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[x.dtype]

    # Use the tile configuration from the previous BF16 hierarchical backend.
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    if routing == "sorted":
        sorted_ids, expert_ids, num_tokens_post_padded = _sorted_expert_assignment(
            topk_ids, config["BLOCK_SIZE_M"], num_experts
        )
    elif routing == "unsorted":
        # One GEMM row tile per token/cluster pair.
        sorted_ids = None
        expert_ids = topk_ids.view(-1)
        num_tokens_post_padded = torch.full(
            (1,),
            topk_ids.numel() * config["BLOCK_SIZE_M"],
            dtype=torch.int32,
            device=topk_ids.device,
        )
    else:
        sorted_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
            topk_ids,
            config,
            x.shape[0],
            top_k,
            num_experts,
            None,
        )
    out = torch.empty(
        (x.shape[0], top_k, weight.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    dispatch_fused_moe_kernel(
        x,
        weight,
        out,
        None,  # A_scale
        None,  # B_scale
        None,  # B_zp
        None,  # topk_weights (mul_routed_weight=False)
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # mul_routed_weight
        top_k,
        config,
        compute_type,
        False,  # use_fp8_w8a8
        False,  # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        False,  # per_channel_quant
    )
    return out


@torch.compile(dynamic=True, fullgraph=True)
def selected_logits(
    lm_head_weights: torch.Tensor, sel: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    """Keep the gather inside compilation so Inductor can fuse the reduction."""
    # Only token count varies within a model; keep kernel tiling model-specific.
    torch._dynamo.mark_static(lm_head_weights)
    torch._dynamo.mark_static(sel, 1)
    torch._dynamo.mark_static(hidden, 1)
    return (lm_head_weights[sel] * hidden[:, None, :]).sum(-1)


class HierarchicalLMHead(nn.Module):
    """Route to equal-size vocabulary clusters and score selected BF16/FP weights.

    The vocabulary weight remains BF16/FP, shaped `[vocab_size, hidden_size]`.
    Checkpoints retain `centroids.weight` and `token_ordering` as state names.
    """

    token_ordering: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
        backend: Literal["compiled", "moe"] = "compiled",
        moe_routing: Literal["auto", "unsorted", "sorted"] = "unsorted",
    ) -> None:
        super().__init__()
        if num_centroids <= 0 or vocab_size <= 0 or vocab_size % num_centroids:
            raise ValueError(
                "vocab_size must be positive and divisible by num_centroids"
            )
        if not 1 <= centroid_intermediate_top_k <= num_centroids:
            raise ValueError("top-k must be between 1 and num_centroids")
        if backend not in ("compiled", "moe"):
            raise ValueError("backend must be 'compiled' or 'moe'")
        if moe_routing not in ("auto", "unsorted", "sorted"):
            raise ValueError("moe_routing must be 'auto', 'unsorted' or 'sorted'")
        if backend != "moe" and moe_routing == "sorted":
            raise ValueError("Sorted MoE routing requires backend='moe'")
        self.backend = backend
        self.moe_routing = moe_routing
        self.register_buffer("centroid_weight", None, persistent=False)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = centroid_intermediate_top_k * self.vocab_size_per_centroid
        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        self.register_buffer("token_ordering", torch.arange(vocab_size))

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare_weights(self, lm_head_weight: torch.Tensor) -> None:
        """Prepare MoE weights after loading/reloading, before CUDA graph capture.

        Compiled scoring reads the original vocabulary weight directly;
        MoE scoring keeps a cluster-major copy. Reloads refill same-layout
        caches in place.
        """
        if self.backend != "moe":
            return
        if (
            lm_head_weight.ndim != 2
            or lm_head_weight.shape[0] < self.vocab_size
            or lm_head_weight.shape[1] != self.hidden_size
            or lm_head_weight.dtype
            not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            )
        ):
            raise ValueError(
                "MoE preparation requires unsharded FP16/BF16/FP32 weights"
            )
        weight = (
            lm_head_weight[self.token_ordering]
            .reshape(self.num_centroids, self.vocab_size_per_centroid, self.hidden_size)
            .contiguous()
        )
        if self.centroid_weight is None:
            self.centroid_weight = weight
        else:
            if (
                self.centroid_weight.shape != weight.shape
                or self.centroid_weight.dtype != weight.dtype
                or self.centroid_weight.device != weight.device
            ):
                raise ValueError(
                    "MoE weight shape, dtype and device must stay fixed on reload"
                )
            self.centroid_weight.copy_(weight)

    def _select_and_score(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            lm_head_weight.ndim != 2
            or lm_head_weight.shape[0] < self.vocab_size
            or lm_head_weight.shape[1] != self.hidden_size
            or not lm_head_weight.is_floating_point()
            or lm_head_weight.dtype != hidden_states.dtype
        ):
            raise ValueError(
                "Expected unsharded, unpacked head weights matching hidden dtype"
            )
        top_k_indices = (
            self.centroids(hidden_states)
            .topk(self.centroid_intermediate_top_k, dim=-1)
            .indices
        )
        clusters = self.token_ordering.view(
            self.num_centroids, self.vocab_size_per_centroid
        )
        selected = clusters[top_k_indices].reshape(
            hidden_states.shape[0], self.num_selected
        )
        if self.backend == "moe":
            if not hidden_states.is_cuda:
                raise ValueError("The MoE head backend requires a CUDA/ROCm device")
            if self.centroid_weight is None:
                raise RuntimeError("Call prepare_weights before using the MoE head")
            logits = _grouped_gemm(
                hidden_states,
                self.centroid_weight,
                top_k_indices.to(torch.int32),
                self.centroid_intermediate_top_k,
                self.num_centroids,
                routing=self.moe_routing,
            ).reshape(hidden_states.shape[0], self.num_selected)
        else:
            logits = selected_logits(lm_head_weight, selected, hidden_states)
        return logits, selected

    def forward(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        """Return full-vocabulary logits with unselected tokens masked."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        output = hidden_states.new_full(
            (hidden_states.shape[0], self.vocab_size),
            torch.finfo(hidden_states.dtype).min,
        )
        return output.scatter_(-1, indices, logits)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        """Return vocabulary argmax IDs without allocating full logits."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        return indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)

    def load_sidecar(self, path: str) -> None:
        """Load exported or training router weights and a balanced token mapping."""
        checkpoint = Path(path).expanduser()
        if checkpoint.is_dir():
            candidates = [
                checkpoint / name
                for name in ("model.pt", "pytorch_model.bin", "model.safetensors")
            ]
            checkpoint = next((p for p in candidates if p.is_file()), candidates[0])
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(checkpoint), device="cpu")
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for wrapper in ("state_dict", "model"):
            if isinstance(state.get(wrapper), dict):
                state = state[wrapper]
        state = {k.removeprefix("module."): v for k, v in state.items()}

        def tensor(*names: str) -> torch.Tensor | None:
            return next(
                (state[k] for k in names if isinstance(state.get(k), torch.Tensor)),
                None,
            )

        centroids = tensor(
            "masked_embedding.centroids.weight",
            "centroids.weight",
            "cluster_signature",
            "model.cluster_signature",
        )
        ordering = tensor(
            "masked_embedding.token_ordering",
            "token_ordering",
            "masked_embedding.cluster_token_table",
            "cluster_token_table",
            "model.cluster_token_table",
        )
        if ordering is None:
            assignment = tensor(
                "token_to_cluster",
                "token_to_centroid",
                "masked_embedding.token_to_cluster",
                "model.token_to_cluster",
            )
            if assignment is None:
                raise ValueError("Sidecar has no token ordering or cluster assignment")
            assignment = assignment.long().flatten()
            if (
                assignment.numel() != self.vocab_size
                or (assignment < 0).any()
                or (assignment >= self.num_centroids).any()
                or not torch.all(
                    torch.bincount(assignment, minlength=self.num_centroids)
                    == self.vocab_size_per_centroid
                )
            ):
                raise ValueError(
                    "Sidecar must define balanced clusters for the full vocabulary"
                )
            ordering = assignment.argsort(stable=True)
        ordering = ordering.long().flatten()
        expected = torch.arange(self.vocab_size, device=ordering.device)
        if not torch.equal(ordering.sort().values, expected):
            raise ValueError("Sidecar token ordering must be a vocabulary permutation")
        if centroids is None or centroids.shape != self.centroids.weight.shape:
            raise ValueError(
                "Sidecar centroid weights have missing or incompatible dimensions"
            )
        with torch.no_grad():
            self.centroids.weight.copy_(centroids)
            self.token_ordering.copy_(ordering)
