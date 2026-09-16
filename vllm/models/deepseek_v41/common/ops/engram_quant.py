# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens", "token_start", "out_tokens"])
def _engram_gather_quant_kernel(
    gathered,
    output,
    scales,
    num_tokens,
    token_start,
    out_tokens,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    PADDED_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    groups = tl.program_id(0).to(tl.int64) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    tokens = groups // PADDED_GROUPS
    cols = (groups % PADDED_GROUPS)[:, None] * 32 + tl.arange(0, 32)[None, :]
    valid = (tokens[:, None] < out_tokens) & (cols < WIDTH)
    source_tokens = tokens + token_start
    ranks = cols // LOCAL_WIDTH
    source = (ranks * num_tokens + source_tokens[:, None]) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    values = tl.load(
        gathered + source,
        valid & (source_tokens[:, None] < num_tokens),
        other=0,
        eviction_policy="evict_first",
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
        output + tokens[:, None] * WIDTH + cols, values * inv_scale[:, None], valid
    )
    group_cols = groups % PADDED_GROUPS
    # F8_128x4: [token/128, group/4, token%32, token%128/32, group%4].
    offsets = (
        tokens // 128 * (128 * PADDED_GROUPS)
        + group_cols // 4 * 512
        + tokens % 32 * 16
        + tokens % 128 // 32 * 4
        + group_cols % 4
    )
    tl.store(scales + offsets, sf, tokens < tl.cdiv(out_tokens, 128) * 128)


def quantize_gathered_engram_rows(
    gathered: torch.Tensor,
    num_tokens: int,
    width: int,
    token_start: int = 0,
    out_tokens: int | None = None,
) -> QuantizedActivation:
    """Quantize rank-major TP rows directly into token-major MXFP8 and F8_128x4 scales.

    Trailing padded heads are discarded without materializing a BF16 transpose.
    ``token_start`` and ``out_tokens`` select one token window (the SP shard);
    rows past ``num_tokens`` quantize as zeros, like ``_engram_select_rows``.
    """
    if out_tokens is None:
        out_tokens = num_tokens
    assert gathered.ndim == 3 and gathered.is_contiguous()
    assert width > 0 and width % 32 == 0
    local_width = gathered.shape[1] * gathered.shape[2]
    assert local_width > 0
    assert num_tokens >= 0 and token_start >= 0 and out_tokens >= 0
    assert num_tokens == 0 or (
        gathered.shape[0] % num_tokens == 0
        and width <= gathered.shape[0] // num_tokens * local_width
    )
    shape = torch.Size((out_tokens, width))
    output = torch.empty(shape, dtype=torch.float8_e4m3fn, device=gathered.device)
    padded_tokens = triton.cdiv(out_tokens, 128) * 128
    padded_groups = triton.cdiv(width // 32, 4) * 4
    scales = torch.empty(
        padded_tokens * padded_groups, dtype=torch.uint8, device=gathered.device
    )
    if out_tokens:
        launch_pdl = current_platform.is_arch_support_pdl()
        _engram_gather_quant_kernel[(triton.cdiv(padded_tokens * padded_groups, 128),)](
            gathered,
            output,
            scales,
            num_tokens,
            token_start,
            out_tokens,
            local_width,
            width,
            padded_groups,
            BLOCK_GROUPS=128,
            LAUNCH_PDL=launch_pdl,
            num_warps=4,
            launch_pdl=launch_pdl,
        )
    return QuantizedActivation(output, scales, gathered.dtype, shape, kMxfp8Dynamic)


def can_quantize_engram_gather(
    embed_tokens: torch.nn.Module, wkv: torch.nn.Module
) -> bool:
    """Require a TP gather feeding an F8_128x4-swizzled MXFP8 projection.

    Call after ``process_weights_after_loading`` so ``wkv`` has its kernel.
    """
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )

    return (
        embed_tokens.tp_size > 1
        and embed_tokens.n_hash_cols * embed_tokens.dim % 32 == 0
        and get_input_quant_key(wkv) == kMxfp8Dynamic
        # QuantKey does not encode the F8_128x4 scale layout.
        and type(getattr(wkv.quant_method, "kernel", None))
        in (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel)
    )
