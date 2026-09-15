# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _engram_gather_quant_kernel(
    gathered,
    output,
    scales,
    num_tokens,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    PADDED_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    groups = tl.program_id(0).to(tl.int64) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    tokens = groups // PADDED_GROUPS
    cols = (groups % PADDED_GROUPS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    source = (cols // LOCAL_WIDTH * num_tokens + tokens[:, None]) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    values = tl.load(
        gathered + source,
        (tokens[:, None] < num_tokens) & (cols < WIDTH),
        other=0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), 1)
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    # Match FlashInfer UE8M0 rounding, including zero/subnormal scales.
    inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
    inv_scale = inv_bits.to(tl.float32, bitcast=True)
    tl.store(
        output + tokens[:, None] * WIDTH + cols,
        values * inv_scale[:, None],
        (tokens[:, None] < num_tokens) & (cols < WIDTH),
    )
    group_cols = groups % PADDED_GROUPS
    offsets = (
        tokens // 128 * (128 * PADDED_GROUPS)
        + group_cols // 4 * 512
        + tokens % 32 * 16
        + tokens % 128 // 32 * 4
        + group_cols % 4
    )
    tl.store(scales + offsets, sf, tokens < tl.cdiv(num_tokens, 128) * 128)


def quantize_gathered_engram_rows(
    gathered: torch.Tensor, num_tokens: int, width: int
) -> QuantizedActivation:
    """Quantize rank-major TP rows directly into token-major MXFP8 and F8_128x4 scales.

    Trailing padded heads are discarded without materializing a BF16 transpose.
    """
    assert gathered.ndim == 3 and gathered.is_contiguous()
    assert width > 0 and width % 32 == 0
    local_width = gathered.shape[1] * gathered.shape[2]
    assert local_width > 0
    assert num_tokens >= 0
    assert num_tokens == 0 or (
        gathered.shape[0] % num_tokens == 0
        and width <= gathered.shape[0] // num_tokens * local_width
    )
    shape = torch.Size((num_tokens, width))
    output = torch.empty(shape, dtype=torch.float8_e4m3fn, device=gathered.device)
    padded_tokens = triton.cdiv(num_tokens, 128) * 128
    padded_groups = triton.cdiv(width // 32, 4) * 4
    scales = torch.empty(
        padded_tokens * padded_groups, dtype=torch.uint8, device=gathered.device
    )
    if num_tokens:
        _engram_gather_quant_kernel[(triton.cdiv(padded_tokens * padded_groups, 32),)](
            gathered,
            output,
            scales,
            num_tokens,
            local_width,
            width,
            padded_groups,
            BLOCK_GROUPS=32,
            LAUNCH_PDL=current_platform.is_arch_support_pdl(),
        )
    return QuantizedActivation(output, scales, gathered.dtype, shape, kMxfp8Dynamic)
