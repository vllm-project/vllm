# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MegaMoE expert layer for DeepSeek V4.

The AMD counterpart to ``nvidia/model.py``'s ``DeepseekV4MegaMoEExperts``. The
kernel is aiter's ``MegaMoEV2`` (FlyDSL), which fuses EP dispatch, both expert
GEMMs, and EP combine into a single launch chain. Unlike the CUDA path there is
no separate input-staging kernel: ``MegaMoEV2.forward`` quantizes the bf16
hidden states to fp8 with per-1x32 e8m0 scales itself.

Weight layout matches the existing ROCm MXFP4 fused-MoE path -- the packed fp4
weights and their e8m0 scales are run through aiter's ``shuffle_weight_a16w4``
and ``shuffle_scale_a16w4``, exactly as ``fused_moe/oracle/mxfp4.py`` does for
the CK kernels, so no new checkpoint handling is required.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.amd.mega_moe_runtime import (
    MegaMoERuntime,
    MegaMoEShape,
    resolve_max_tok_per_rank,
)

logger = init_logger(__name__)


def shuffle_mega_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert checkpoint-layout fp4 expert weights into aiter's kernel layout.

    Split out of :meth:`DeepseekV4MegaMoEExperts.finalize_weights` so it can be
    tested against the reference preparation in aiter's own
    ``test_mega_moe_v2.py`` without standing up a distributed environment.

    Inputs are in checkpoint layout -- ``w13`` is ``(E, 2*inter, hidden//2)``
    packed two fp4 per byte with w1 occupying rows ``[0, inter)`` and w3 rows
    ``[inter, 2*inter)``; scales are per-1x32 e8m0 bytes. ``shuffle_scale``
    asserts a 2-D input and treats the leading axis as ``E * N``, so the scales
    are flattened before the call.
    """
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    experts = w13.shape[0]

    # The shuffles are layout transforms driven by shape; the fp4 pair packing
    # only has to survive as opaque bytes.
    w13_packed = w13.view(torch.float4_e2m1fn_x2)
    w2_packed = w2.view(torch.float4_e2m1fn_x2)

    return (
        shuffle_weight_a16w4(w13_packed, 16, True).contiguous(),
        shuffle_scale_a16w4(
            w13_scale.reshape(-1, w13_scale.shape[-1]), experts, True
        ).contiguous(),
        shuffle_weight_a16w4(w2_packed, 16, False).contiguous(),
        shuffle_scale_a16w4(
            w2_scale.reshape(-1, w2_scale.shape[-1]), experts, False
        ).contiguous(),
    )


def make_deepseek_v4_mega_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    """Checkpoint -> parameter mapping for the mega-MoE expert layout.

    ``fused_moe_make_expert_params_mapping`` targets FusedMoE's parameter
    names, which the mega path does not have. This is the AMD copy of
    ``nvidia/model.py``'s ``make_deepseek_v4_expert_params_mapping``: w1/w3 land
    in the fused ``w13_`` parameters, w2 in ``w2_``, and the per-expert loader
    on :class:`DeepseekV4MegaMoEExperts` does the sharding.
    """
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in (("w1", "w1"), ("w2", "w2"), ("w3", "w3"))
    ]


def finalize_mega_moe_layers(layers) -> None:
    """Run mega weight finalization over a layer container.

    Tolerates the three shapes the DeepSeek V4 code uses -- a list of decoder
    layers, a ``ModuleDict`` of MTP layers, and MTP layers that wrap the real
    decoder layer in ``.mtp_block`` -- and is a no-op when the mega path is off.
    """
    if hasattr(layers, "values"):
        layers = layers.values()
    for layer in layers:
        layer = getattr(layer, "mtp_block", layer)
        ffn = getattr(layer, "ffn", None)
        finalize = getattr(ffn, "finalize_mega_moe_weights", None)
        if finalize is not None:
            finalize()


class DeepseekV4MegaMoEExperts(nn.Module):
    """Routed experts backed by aiter's fused MegaMoEV2 kernel."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit

        ep_group = get_ep_group()
        self.ep_size = ep_group.world_size
        self.ep_rank = ep_group.rank_in_group

        # MegaMoEV2 rejects any forward pass carrying more tokens than
        # max_tok_per_rank, so this must round the scheduler budget *up* to the
        # next power of two. It is also what sizes the symmetric buffers.
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_tok_per_rank = resolve_max_tok_per_rank(self.max_num_tokens)

        weight_attrs = {"weight_loader": self.weight_loader}

        # Loader-side layout, matching the checkpoint: fp4 packed two-per-byte
        # with per-1x32 e8m0 scales. finalize_weights() replaces these with the
        # shuffled layout the kernel wants.
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        self._shuffled: tuple[torch.Tensor, ...] | None = None

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # --- weight loading ---------------------------------------------------

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if not (self.experts_start_idx <= expert_id < self.experts_end_idx):
            return False if return_success else None
        local_expert_id = expert_id - self.experts_start_idx

        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            if "w13_" not in weight_name:
                return False if return_success else None
            shard_offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, shard_offset, self.intermediate_size)
        elif shard_id == "w2":
            if "w2_" not in weight_name:
                return False if return_success else None
        else:
            raise ValueError(f"Unsupported expert shard id: {shard_id}")

        if expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                f"vs checkpoint {tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight.view(expert_data.dtype))

        return True if return_success else None

    def finalize_weights(self) -> None:
        """Shuffle the loaded weights into aiter's a16w4 kernel layout.

        Idempotent: safe to call again after dummy-weight loading, mirroring
        the CUDA path.
        """
        if self._shuffled is not None:
            return

        (
            w13_shuffled,
            w13_scale_shuffled,
            w2_shuffled,
            w2_scale_shuffled,
        ) = shuffle_mega_moe_weights(
            self.w13_weight.data,
            self.w13_weight_scale.data,
            self.w2_weight.data,
            self.w2_weight_scale.data,
        )

        self._shuffled = (
            w13_shuffled,
            w13_scale_shuffled,
            w2_shuffled,
            w2_scale_shuffled,
        )

        # Build the shared instance here rather than lazily in forward().
        # MegaMoEV2.__init__ allocates the symmetric buffers and ends in
        # shmem_barrier_all(), so it is collective and host-blocking: every EP
        # rank must reach it. Weight finalization runs on all ranks at model
        # load, whereas a forward pass may legitimately skip this layer on a
        # rank that has no tokens under DP attention -- which would deadlock.
        self._runtime().build(
            w1=w13_shuffled,
            w1_scale=w13_scale_shuffled,
            w2=w2_shuffled,
            w2_scale=w2_scale_shuffled,
        )

        # Drop the loader-side parameters; the shuffle helpers return fresh
        # tensors, so the original storage is no longer referenced.
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        torch.cuda.empty_cache()

    # --- runtime ----------------------------------------------------------

    def _runtime(self) -> MegaMoERuntime:
        shape = MegaMoEShape(
            world_size=self.ep_size,
            model_dim=self.hidden_size,
            inter_dim=self.intermediate_size,
            experts=self.num_experts,
            topk=self.top_k,
            max_tok_per_rank=self.max_tok_per_rank,
            swiglu_limit=self.swiglu_limit,
        )
        return MegaMoERuntime.get(shape, self.ep_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_tok_per_rank:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {num_tokens} tokens, but the "
                f"symmetric buffers were sized for {self.max_tok_per_rank}."
            )

        # Covers the dummy-weight-loading path, as on CUDA.
        self.finalize_weights()
        assert self._shuffled is not None
        w13, w13_scale, w2, w2_scale = self._shuffled

        runtime = self._runtime()

        # The kernel's contracts: contiguous bf16 activations, float32 routing
        # weights, int32 expert ids.
        x = (
            hidden_states
            if hidden_states.is_contiguous()
            else hidden_states.contiguous()
        )
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        wts = topk_weights.to(torch.float32).contiguous()
        ids = topk_ids.to(torch.int32).contiguous()

        with runtime.bind_weights(
            self, w1=w13, w1_scale=w13_scale, w2=w2, w2_scale=w2_scale
        ) as moe:
            out = moe.forward(x, wts, ids)

        # The kernel returns a view into the shared symmetric combine buffer,
        # which the next layer's call will overwrite. Copy out before returning.
        return out.clone()


DeepseekV4MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
