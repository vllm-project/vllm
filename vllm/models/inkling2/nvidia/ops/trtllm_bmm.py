# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRT-LLM-gen batched expert GEMMs (FlashInfer cubins) for the Inkling2 MoE.

The GEMMs read expert-sorted activation rows whose expert segments are padded
to the token tile, with block scales in the R8c4 (tile < 128) or R128c4 layout
(`block_scales`), and row-shuffled expert weights. Outputs are bf16; NVFP4
outputs are scaled by an fp32 factor per row.
"""

import functools
import math
import os
from dataclasses import dataclass

import regex as re
import torch

from . import use_sm100_kernels

_SOURCE = os.path.join(os.path.dirname(__file__), "trtllm_bmm.cu")

# The GEMMs' TMA accesses can reach the first 128 KiB of an operand however
# few of its rows are used (Blackwell's 128 KiB OOB address generation), so,
# as in FlashInfer's MoE launchers (`maybeGetMinTokenCount`), every operand
# and output is backed by at least that much memory.
MIN_OPERAND_BYTES = 128 * 1024


def operand(
    rows: int, *shape: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """`torch.empty((rows, *shape))`, allocated with enough rows to span
    `MIN_OPERAND_BYTES`."""
    row_bytes = math.prod(shape) * dtype.itemsize
    alloc_rows = max(rows, -(-MIN_OPERAND_BYTES // row_bytes))
    return torch.empty((alloc_rows, *shape), dtype=dtype, device=device)[:rows]


@functools.cache
def _module():
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    # Loads FlashInfer's MoE module (and its cubin loader), whose batched GEMM
    # runner this module calls into.
    get_trtllm_moe_sm100_module()
    moe = gen_trtllm_gen_fused_moe_sm100_module()
    flags = [f for f in moe.extra_cuda_cflags if f.startswith(("-DTLLM", "-DENABLE"))]
    spec = gen_jit_spec(
        "inkling2_trtllm_bmm",
        [_SOURCE],
        extra_cuda_cflags=flags
        + current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
        extra_ldflags=[str(moe.get_library_path())],
        extra_include_paths=moe.extra_include_dirs,
    )
    return spec.build_and_load(), moe.extra_include_dirs[0]


@functools.cache
def _config_names(include_dir) -> list[str]:
    # The kernel manifest FlashInfer's MoE module is built from.
    with open(include_dir / "flashinferMetaInfo.h") as f:
        return re.findall(r'^\{nullptr, \d+, \d+, "(\w+)"', f.read(), re.M)


@functools.cache
def configs(nvfp4: bool, tile_n: int) -> dict[str, tuple[int, int]]:
    """Kernel name -> (config index, B scale layout) of the (format, token
    tile) GEMMs."""
    module, include_dir = _module()
    names = _config_names(include_dir)
    flat = list(module.bmm_configs(nvfp4, tile_n))
    return {names[i]: (i, layout) for i, layout in zip(flat[::2], flat[1::2])}


# Token tile by batch size (tokens up to the first entry): tiles above the
# mean rows per expert pay off once the routing is skewed.
_TILES = ((8, 8), (128, 16), (384, 32), (768, 64), (4096, 128), (1 << 31, 256))
# At the largest tile (batches above the CUDA-graph-captured prefill sizes run
# eagerly at their exact token count) `max_ctas` is rounded up to a multiple
# of this. The routing kernel takes it as a constexpr, so this leaves it a
# handful of variants over the eager token range instead of one per 32 tokens,
# each compiled during the first request that reaches it. The extra tiles are
# launched idle (`tiles` bounds the GEMMs) and cost a few MiB of padded rows.
_LARGE_TILE_CTAS_GRANULE = 128
# Kernel of each (format, tile) for the gate/up and down projections.
_KERNELS = {
    (True, 8): ("t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd",) * 2,
    (True, 16): ("t128x16x512_s5_et128x16_m256x16x64_c2x1x1_rM_TN_transOut_schPd",) * 2,
    (True, 32): ("t128x32x512_s5_et128x32_m256x32x64_c2x1x1_rM_TN_transOut_schPd",) * 2,
    (True, 64): ("t128x64x512_s4_et128x64_m256x64x64_c2x1x1_rM_TN_transOut_schPd",) * 2,
    (True, 128): (
        "t128x128x256u2_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd",
        "t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd",
    ),
    (True, 256): ("t128x256x256_s5x5x5x5x1x1_et128x64_m256x256x64_c2x1x1",) * 2,
    (False, 8): ("t128x8x512_s3_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd",) * 2,
    (False, 16): (
        "t128x16x256u2_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_schPd",
        "t128x16x256_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_schPd",
    ),
    (False, 32): ("t128x32x256_s5_et128x32_m256x32x32_c2x1x1_rM_TN_transOut_schPd",)
    * 2,
    (False, 64): (
        "t128x64x256u2_s4_et128x64_m256x64x32_c2x1x1_rM_TN_transOut_schPd",
        "t128x64x256_s4_et128x64_m256x64x32_c2x1x1_rM_TN_transOut_schPd",
    ),
    (False, 128): (
        "t128x128x256u2_s4_et128x64_m256x128x32_c2x1x1_rM_TN_transOut_schPd",
        "t128x128x256_s4_et128x64_m256x128x32_c2x1x1_rM_TN_transOut_schPd",
    ),
    (False, 256): ("t128x256x128_s5x5x5x5x1x1_et128x64_m256x256x32_c2x1x1",) * 2,
}


@dataclass(frozen=True)
class ExpertPlan:
    """Token tile, launched tiles (an upper bound) and the kernel of each
    projection for a batch."""

    tile: int
    max_ctas: int
    kernels: tuple[int, int]

    @property
    def rows(self) -> int:
        return self.tile * self.max_ctas


@functools.cache
def expert_plan(
    tokens: int,
    topk: int,
    num_experts: int,
    nvfp4: bool,
    shapes: tuple[tuple[int, int], tuple[int, int]],
    tile: int | None = None,
) -> ExpertPlan:
    """Plan for `tokens` routed to `topk` of all experts, `num_experts`
    of them local, with (N, K) weight `shapes` of the two projections."""
    if tile is None:
        tile = next(t for max_tokens, t in _TILES if tokens <= max_tokens)
    # Tiles if the local rows fill as many experts as possible, then pile up.
    rows = tokens * topk
    max_ctas = min(num_experts, rows) + (rows - min(num_experts, rows)) // tile
    if tile >= _TILES[-1][1]:
        granule = _LARGE_TILE_CTAS_GRANULE
        max_ctas = -(-max_ctas // granule) * granule
    if not use_sm100_kernels():
        # No TRT-LLM-gen kernels (Hopper): only `route`'s layout.
        return ExpertPlan(tile, max_ctas, (0, 0))
    names = configs(nvfp4, tile)
    keys = _KERNELS[(nvfp4, tile)]
    kernels = []
    for projection, (n, k) in enumerate(shapes):
        # The projection's kernel, else the other's (an unrolled K loop, say,
        # needs K to be a multiple of its double K tile).
        for key in (keys[projection], keys[1 - projection]):
            ((index, layout),) = (v for name, v in names.items() if key in name)
            assert layout == (3 if tile >= 128 else 1)  # R128c4 / R8c4 activations
            size = _module()[0].bmm_workspace_size(
                nvfp4, tile, index, n, k, num_experts, max_ctas
            )
            if size == 0:
                kernels.append(index)
                break
        else:
            raise AssertionError(f"no Inkling2 expert GEMM for {(n, k)}, tile {tile}")
    return ExpertPlan(tile, max_ctas, (kernels[0], kernels[1]))


def load_kernels(
    weights: tuple[torch.Tensor, torch.Tensor],
    weight_scales: tuple[torch.Tensor, torch.Tensor],
    topk: int,
) -> None:
    """Loads the kernels of every batch size's plan by running them on no
    tiles: loading is not allowed while a CUDA graph is captured."""
    nvfp4 = weights[0].dtype == torch.uint8
    num_experts = weights[0].shape[0]
    pack = 2 if nvfp4 else 1
    shapes = tuple((w.shape[1], w.shape[2] * pack) for w in weights)
    device = weights[0].device
    ones = torch.ones(num_experts, dtype=torch.float32, device=device)
    for _, tile in _TILES:
        plan = expert_plan(1, topk, num_experts, nvfp4, shapes, tile)
        tiles = torch.zeros(2 * plan.max_ctas + 2, dtype=torch.int32, device=device)
        for projection, (weight, scale) in enumerate(zip(weights, weight_scales)):
            k = weight.shape[2]
            act = operand(plan.rows, k, dtype=weight.dtype, device=device)
            act_scale = operand(
                plan.rows, scale.shape[2], dtype=scale.dtype, device=device
            )
            row_scale = (
                operand(plan.rows, dtype=torch.float32, device=device)
                if nvfp4
                else None
            )
            bmm(act, act_scale, row_scale, weight, scale, ones, tiles, plan, projection)


def bmm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan: ExpertPlan,
    projection: int,
) -> torch.Tensor:
    """bf16 `[plan.rows, N]` expert GEMM of the tile-padded `act` rows.

    `tiles` holds the expert and row limit of each of the `plan.max_ctas`
    tiles, then the number of tiles and of padded rows. `ones` (one per
    expert) is the GEMM's output scale.
    """
    nvfp4 = weight.dtype == torch.uint8
    n = weight.shape[1]
    out = operand(plan.rows, n, dtype=torch.bfloat16, device=act.device)
    m = plan.max_ctas
    _module()[0].bmm_run(
        nvfp4,
        plan.tile,
        plan.kernels[projection],
        weight,
        weight_scale,
        act,
        act_scale,
        row_scale,
        ones,
        out,
        tiles[:m],
        tiles[m : 2 * m],
        tiles[2 * m : 2 * m + 1],
        tiles[2 * m + 1 :],
        m,
    )
    return out
