# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When a DeepSeek-V4.1 decoder layer gets the gfx942 large-step MoE block
(`largestep.moe.LargeStepHook`): the layer has a mono decode layer (so
`--monokernel` is on and the gfx942 weight copies `mono942` exist), the GPU
is gfx942, TP is 4 or 8, and the layer is a target-model layer (384 experts,
top 6). The block then takes the decode steps the mono decode layer is too
small for (`largestep.moe.MIN_ROWS` to `MAX_ROWS` rows)."""

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

logger = init_logger(__name__)


def build(layer):
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() != 3:
        return None
    tp = get_tensor_model_parallel_world_size()
    ffn = getattr(layer, "ffn", None)
    if tp not in (4, 8) or ffn is None or ffn.shared_experts is None:
        return None
    from vllm.models.deepseek_v41.amd.largestep import moe as lsmoe

    if getattr(ffn, "n_routed_experts", 0) != lsmoe.EXPERTS:
        return None
    experts = ffn.experts.routed_experts
    if getattr(experts, "mono942", None) is None:
        return None
    if getattr(ffn.shared_experts.gate_up_proj, "mono942", None) is None:
        return None
    hook = lsmoe.LargeStepHook(ffn, tp, layer.attn_norm.weight.device)
    logger.info_once(
        "DSv4.1 gfx942 large-step MoE on (decode steps of %d to %d rows)",
        lsmoe.MIN_ROWS,
        lsmoe.MAX_ROWS,
    )
    return hook
