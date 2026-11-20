"""Shared CPU cases for the four RoutedExperts.load_weights architectures.

Each variant worktree copies this file next to its own `make_experts(...)` and
runs the same assertions.  Shapes are tiny-random/gpt-oss-mxfp4: E=32, H=32, INTER=64.

STAND-IN LOADER (what this CPU harness does NOT exercise): the real
`RoutedExperts.weight_loader` is replaced by `standin_loader`, which mimics only
(a) the existing gpt_oss_mxfp4 branch (whole tensor copied into
param.data[:, :d1, :d2]) and (b) per-expert slot writes.  TP/EP slicing, Marlin
padding/repacking, quant-method process_weights_after_loading and kernels are
NOT run.
"""

import torch

E, H, INTER = 32, 32, 64
g = torch.Generator().manual_seed(0)


def rnd(*shape, dtype=torch.float32):
    if dtype == torch.uint8:
        return torch.randint(0, 255, shape, generator=g, dtype=torch.uint8)
    return torch.randn(*shape, generator=g, dtype=dtype)


# --- checkpoint tensors ------------------------------------------------------
NORMAL_FUSED = {"gate_up_proj": rnd(E, 2 * INTER, H), "down_proj": rnd(E, H, INTER)}
NORMAL_PER_EXPERT = {
    "3.gate_proj.weight": rnd(INTER, H),
    "3.up_proj.weight": rnd(INTER, H),
    "3.down_proj.weight": rnd(H, INTER),
}
GPT_OSS = {
    "gate_up_proj_blocks": rnd(E, 2 * INTER, H // 32, 16, dtype=torch.uint8),
    "gate_up_proj_scales": rnd(E, 2 * INTER, H // 32, dtype=torch.uint8),
    "gate_up_proj_bias": rnd(E, 2 * INTER),
    "down_proj_blocks": rnd(E, H, INTER // 32, 16, dtype=torch.uint8),
    "down_proj_scales": rnd(E, H, INTER // 32, dtype=torch.uint8),
    "down_proj_bias": rnd(E, H),
}
# parameter shapes the layer owns (unpadded; Marlin padding is NOT modelled)
GPT_OSS_PARAM_SHAPES = {
    "w13_weight": (E, 2 * INTER, H // 2),
    "w13_weight_scale": (E, 2 * INTER, H // 32),
    "w13_bias": (E, 2 * INTER),
    "w2_weight": (E, H, INTER // 2),
    "w2_weight_scale": (E, H, INTER // 32),
    "w2_bias": (E, H),
}
# expected param contents after loading GPT_OSS: whole tensors, blocks flattened
GPT_OSS_EXPECTED = {
    "w13_weight": GPT_OSS["gate_up_proj_blocks"].flatten(2),
    "w13_weight_scale": GPT_OSS["gate_up_proj_scales"],
    "w13_bias": GPT_OSS["gate_up_proj_bias"],
    "w2_weight": GPT_OSS["down_proj_blocks"].flatten(2),
    "w2_weight_scale": GPT_OSS["down_proj_scales"],
    "w2_bias": GPT_OSS["down_proj_bias"],
}
NORMAL_PARAM_SHAPES = {"w13_weight": (E, 2 * INTER, H), "w2_weight": (E, H, INTER)}
# expected for NORMAL_FUSED: gate half then up half per expert (halved, NOT interleaved)
NORMAL_FUSED_EXPECTED = {
    "w13_weight": torch.cat(NORMAL_FUSED["gate_up_proj"].chunk(2, dim=1), dim=1),
    "w2_weight": NORMAL_FUSED["down_proj"],
}


def standin_loader(
    param,
    loaded_weight,
    weight_name=None,
    shard_id=None,
    expert_id=None,
    return_success=False,
    **_,
):
    """Mimic RoutedExperts.weight_loader for the cases under test."""
    if expert_id is None:  # whole stacked tensor (gpt_oss_mxfp4 branch)
        idx = tuple(slice(0, s) for s in loaded_weight.shape)
        param.data[idx].copy_(loaded_weight)
    else:  # one expert slot; shard w1/w3 -> halves of dim 0, w2 -> whole
        slot = param.data[expert_id]
        if shard_id in ("w1", "w3"):
            half = slot.shape[0] // 2
            slot = slot[:half] if shard_id == "w1" else slot[half:]
        slot.copy_(loaded_weight)
    return True if return_success else None


def make_params(experts, shapes, dtypes=None):
    for name, shape in shapes.items():
        dt = (
            torch.uint8
            if (name.split("_", 1)[1] == "weight" and shapes is GPT_OSS_PARAM_SHAPES)
            or name.endswith("weight_scale")
            and shapes is GPT_OSS_PARAM_SHAPES
            else torch.float32
        )
        p = torch.nn.Parameter(torch.zeros(shape, dtype=dt), requires_grad=False)
        p.weight_loader = standin_loader
        setattr(experts, name, p)
