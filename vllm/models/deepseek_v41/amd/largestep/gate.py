# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When a DeepSeek-V4.1 decoder layer gets the gfx942 large-step MoE block
(`largestep.moe.LargeStepHook`): the layer has a mono decode layer (so
`--monokernel` is on and the gfx942 weight copies `mono942` exist), the GPU
is gfx942 with 304 CUs (MI300X and MI325X, the parts it is measured on: the
stage kernels launch one CTA a CU), the model runs in bf16, TP is 4 or 8, and
VLLM_BATCH_INVARIANT is off, and
the layer has the shapes the kernels are built for (`largestep.shapes`: 384
experts, top 6, hidden 5120). The block then
takes the decode steps the mono decode layer is too small for
(`largestep.moe.MIN_ROWS` to `MAX_ROWS` rows)."""

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

logger = init_logger(__name__)


def build(layer):
    import torch

    from vllm.platforms import current_platform
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() != 3:
        return None
    import vllm.envs as envs

    # The block quantizes activations to MXFP8 for steps of 43 to 256 rows
    # only, so a request's numerics would depend on the batch it lands in.
    if envs.VLLM_BATCH_INVARIANT:
        return None
    from vllm.models.deepseek_v41.amd.largestep.moe_ug import BLOCKS

    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus != BLOCKS:
        logger.info_once(
            "DSv4.1 gfx942 large-step MoE off: %d CUs, built for %d", cus, BLOCKS
        )
        return None
    # The quantization kernels read the activations as bf16 bits.
    if layer.attn_norm.weight.dtype != torch.bfloat16:
        return None
    tp = get_tensor_model_parallel_world_size()
    ffn = getattr(layer, "ffn", None)
    if tp not in (4, 8) or ffn is None or ffn.shared_experts is None:
        return None
    from vllm.models.deepseek_v41.amd.largestep import moe as lsmoe
    from vllm.models.deepseek_v41.amd.largestep import shapes

    # The kernels are built for these shapes only. Another model with 384
    # experts but another top-k or hidden width must not take the path.
    if (
        getattr(ffn, "n_routed_experts", 0) != shapes.EXPERTS
        or getattr(ffn, "n_activated_experts", 0) != shapes.TOPK
        or getattr(ffn, "hidden_size", 0) != shapes.HIDDEN
    ):
        return None
    experts = getattr(ffn.experts, "routed_experts", None)
    if getattr(experts, "mono942", None) is None:
        return None
    hook = lsmoe.LargeStepHook(layer, ffn, tp, layer.attn_norm.weight.device)
    logger.info_once(
        "DSv4.1 gfx942 large-step MoE on (decode steps of %d to %d rows)",
        lsmoe.MIN_ROWS,
        lsmoe.MAX_ROWS,
    )
    return hook
