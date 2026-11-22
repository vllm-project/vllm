# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route `fused_ep8`: DeepSeek-V4.1-Flash's routed experts under EP8 as one
persistent amd_kernels launch per rank (MXFP8 quant, dispatch, GEMM1 with
clamped SwiGLU, GEMM2, combine), behind `DeepseekV4AiterMegaMoEExperts`.

The kernel reads the tensors MegaMoE already holds (no copy), so MegaMoE stays
the fallback of every call. Dispatch and combine are collective: the branch is
decided by `akr_config.dp_rows`, the DP-group maximum row count of the step, so
all EP ranks take the same one; above `MAX_ROWS` rows per rank MegaMoE runs.
Building the expert-parallel context and its warm-up are collectives too; both
run at load on every rank in the same order, before graph capture.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.amd import akr_config, akr_hooks, akr_kernels

logger = init_logger(__name__)

# rows per rank the kernel takes (the DP-group maximum decides)
MAX_ROWS = 1024


def unsupported_reason(experts) -> str | None:
    """Why a MegaMoE layer cannot run the fused EP kernel (None: it can). Only
    config and shapes, so every rank reaches the same answer."""
    if experts.fuse_shared_expert:
        return "the shared expert is fused into the routed experts"
    if experts.swiglu_limit not in (0.0, 10.0):
        return f"SwiGLU limit {experts.swiglu_limit}"
    return akr_kernels.fused_ep_unsupported_reason(experts, MAX_ROWS)


class FusedEp8:
    """`DeepseekV4AiterMegaMoEExperts.akr` of one layer."""

    def __init__(self, ep, weights) -> None:
        self.ep = ep
        self.weights = weights

    def serves(self, m: int) -> bool:
        return m <= MAX_ROWS and akr_config.serves("fused_ep8", akr_config.dp_rows(m))

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        x_quant: QuantizedActivation | None,
    ) -> torch.Tensor | None:
        m = hidden_states.shape[0]
        if not self.serves(m):
            return None
        return akr_kernels.fused_ep_moe(
            self.ep,
            self.weights,
            hidden_states,
            x_quant,
            topk_weights,
            topk_ids,
            rows=akr_config.dp_rows(m),
        )


# expert-parallel contexts of this process, by layer geometry
_contexts: dict[tuple[int, ...], Any] = {}


def post_load(model) -> None:
    """Attach the kernel to every eligible layer and build the expert-parallel
    context (collectives: same order on every rank, before graph capture)."""
    from vllm.models.deepseek_v4.amd.mega_moe import DeepseekV4AiterMegaMoEExperts

    layers = []
    for experts in model.modules():
        if not isinstance(experts, DeepseekV4AiterMegaMoEExperts):
            continue
        why = unsupported_reason(experts) if experts._runtimes else "not finalized"
        if why is None:
            layers.append(experts)
        else:
            logger.info_once(
                "amd_kernels fused_ep8 off for %s: %s", experts.prefix, why
            )
    if not layers:
        if not akr_hooks.is_draft(model):
            akr_config.fail("fused_ep8", "no eligible MegaMoE layer")
        return
    first = layers[0]
    # one context per process serves every layer: with a placement plan the
    # slots per rank must agree
    mismatched = [e for e in layers if e.num_local_experts != first.num_local_experts]
    for experts in mismatched:
        logger.warning(
            "amd_kernels fused_ep8 off for %s: %d experts per rank, not %d",
            experts.prefix,
            experts.num_local_experts,
            first.num_local_experts,
        )
    layers = [e for e in layers if e.num_local_experts == first.num_local_experts]
    key = (
        first.hidden_size,
        first.intermediate_size,
        first.top_k,
        first.num_local_experts,
    )
    if key not in _contexts:
        _contexts[key] = akr_kernels.fused_ep_context(first)
    ep = _contexts[key]
    for experts in layers:
        experts.akr = FusedEp8(ep, akr_kernels.fused_ep_weights(experts))
    ep.warm_up(first.akr.weights)
    logger.info("amd_kernels fused_ep8 on %d layers", len(layers))
