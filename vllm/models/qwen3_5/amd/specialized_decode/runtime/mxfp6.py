# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP6 E2M3 projections for the Qwen3.8 specialized decode.

With ``--additional-config '{"monokernel_projections": "mxfp6"}'`` and
``--monokernel``, ``prepare`` runs once after the weights are loaded, before
the kernels are compiled. It rounds the bf16 GDN in_proj_qkvz and out_proj
and the attention qkv_proj and o_proj to MXFP6 in place
(``round_projections``), so the prefill GEMMs, which keep multiplying the
bf16 weights, and the decode kernels use the same values. Then it packs
MXFP6 copies of them for the kernels (``fp6_pack``). The GDN block
multiplies in_proj_qkvz in MXFP6 (``q6``) and the layer block out_proj or
o_proj (``o6``), both with the bf16 rows (W6A16). The copies come on top of
the bf16 weights.

An E2M3 value has a sign, a 2-bit exponent with bias 1 and a 3-bit mantissa.
Its magnitudes are 0 to 1.875 in steps of 0.125, 2 to 3.75 in steps of 0.25
and 4 to 7.5 in steps of 0.5. A block of 32 values along the last dimension
shares one power-of-two scale, which an e8m0 code holds.

The block scale follows aiter's dynamic_mxfp4_quant, which makes vLLM's
online MXFP4 weights. The block maximum is first rounded to the element's
3 mantissa bits, and the scale is 2 to the power of that exponent minus 2.
So a block whose maximum would round up to the next power of two gets the
next scale, and no value in a block is left above 7.5 times its scale. Each
value then rounds to the nearest E2M3 value, with ties to the even one.
"""

import regex as re
import torch

from vllm.logger import init_logger
from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L

logger = init_logger(__name__)

BLOCK = 32
E2M3_MAX = 7.5

# The projections that the MXFP6 builds read: the GDN in_proj_qkvz and
# out_proj and the attention qkv_proj and o_proj of the decoder layers.
# in_proj_ba is 0.5 MB a GPU and stays bf16, as it does with the online
# MXFP4 targets. The router gate, the shared expert and lm_head are left
# alone.
PROJ_RE = re.compile(
    r"(^|\.)layers\.\d+\."
    r"(linear_attn\.(in_proj_qkvz|out_proj)|self_attn\.(qkv_proj|o_proj))$"
)


def round_e2m3(x: torch.Tensor) -> torch.Tensor:
    """Returns x with every block of 32 values along the last dimension
    rounded to MXFP6 E2M3, in x's dtype. Every E2M3 value times a power of two
    has at most 4 significant bits, so bf16, fp16, fp32 and FP8 e4m3 hold
    the result exactly as long as its exponent is in their range."""
    shape = x.shape
    if shape[-1] % BLOCK:
        raise ValueError(
            f"MXFP6 rounding needs a last dimension that is a multiple of "
            f"{BLOCK}, not {shape[-1]}"
        )
    v = x.float().reshape(-1, shape[-1] // BLOCK, BLOCK)
    amax = v.abs().amax(dim=-1, keepdim=True)
    # Adding 2**19 to the fp32 bits of the maximum rounds its mantissa at the
    # third bit, and a carry moves into the exponent. The mask then keeps only
    # the exponent field.
    bits = (amax.view(torch.int32) + 0x80000) & 0x7F800000
    exp = (bits >> 23) - 127 - 2
    # An all-zero block has an exponent field of 0. Its values round to zero
    # with any scale, so its scale only has to stay a normal fp32 number,
    # because a flushed scale of zero would turn 0 / 0 into NaN.
    scale = torch.exp2(exp.clamp(-126, 127).float())
    a = v.abs() / scale
    q = torch.where(
        a < 2.0,
        torch.round(a * 8.0) * 0.125,
        torch.where(a < 4.0, torch.round(a * 4.0) * 0.25, torch.round(a * 2.0) * 0.5),
    )
    q = q.clamp(max=E2M3_MAX)
    return (torch.sign(v) * q * scale).reshape(shape).to(x.dtype)


@torch.no_grad()
def round_projections(model: torch.nn.Module, rows: int = 1024) -> None:
    """Rounds the weights of the projections in PROJ_RE to MXFP6 in place, if
    their linear method is an unquantized one. A layer that an online
    method quantizes later is skipped, so the MXFP6 values are never
    quantized again. The rounding writes into the existing weight in slices
    of ``rows`` rows, which keeps the temporaries small and keeps a weight
    that is a view of a merged buffer a view of it."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    count: dict[str, int] = {}
    err2: dict[str, float] = {}
    ref2: dict[str, float] = {}
    for name, module in model.named_modules():
        if "mtp" in name or not PROJ_RE.search(name):
            continue
        method = type(getattr(module, "quant_method", None)).__name__
        weight = getattr(module, "weight", None)
        if "Unquantized" not in method or weight is None:
            continue
        if weight.dtype not in (torch.bfloat16, torch.float16):
            continue
        kind = name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1]
        w = weight.data
        for r0 in range(0, w.shape[0], rows):
            part = w[r0 : r0 + rows]
            rounded = round_e2m3(part)
            err2[kind] = err2.get(kind, 0.0) + float(
                (rounded.float() - part.float()).pow(2).sum()
            )
            ref2[kind] = ref2.get(kind, 0.0) + float(part.float().pow(2).sum())
            part.copy_(rounded)
        count[kind] = count.get(kind, 0) + 1
    if not count:
        raise RuntimeError(
            "projections mxfp6 found no bf16 projection to round. "
            "Serve the projections in bf16, for example with online MXFP4 "
            "targets that name only the shared expert."
        )
    if get_tensor_model_parallel_rank() == 0:
        for kind in sorted(count):
            logger.info(
                "Fake MXFP6 projection weights: %d %s layers, relative error %.4f",
                count[kind],
                kind,
                (err2[kind] / max(ref2[kind], 1e-30)) ** 0.5,
            )


def fp6_pack(w):
    """bf16 [N, K] -> (MXFP6 E2M3 weight, its e8m0 scales) as two flat uint8
    tensors in the layout that fp6_gemv_loads reads. The scale and the
    rounding are amd-quark's MXFP6 E2M3, as in mxfp6.py: a 32-block's scale
    is 2 to the power of (the exponent of its amax rounded to 3 mantissa
    bits) - 2, and the values round to nearest, ties to even, at most 7.5.

    The 16-row group rg and the 128-K step st form a tile of 1536 bytes at
    byte (rg (K / 128) + st) 1536. Lane l = 16 g + r of the tile holds row
    16 rg + r at K 128 st + 32 g + i, i = 0 .. 31, as 32 6-bit codes, code i
    at bits 6 i .. 6 i + 5 of 24 bytes. The first 16 bytes go to byte 16 l of
    the tile and the last 8 to byte 1024 + 8 l. The scale of the block is
    byte (st % 4) of scale word (rg (K / 512) + st / 4) 64 + l."""
    n, k = w.shape
    L.need(n % 16 == 0 and k % 512 == 0, "fp6_pack: N a multiple of 16 and K of 512")
    v = w.float().reshape(n, k // 32, 32)
    amax = v.abs().amax(dim=-1, keepdim=True)
    exp = (((amax.view(torch.int32) + 0x80000) & 0x7F800000) >> 23) - 129
    exp = exp.clamp(-126, 127)
    a = v.abs() / torch.exp2(exp.float())
    # The 32 E2M3 magnitudes in code order are 0 to 1.875 in steps of 0.125,
    # 2 to 3.75 in steps of 0.25 and 4 to 7.5 in steps of 0.5. torch.round
    # rounds ties to even, and a value that rounds up to the next range gets
    # that range's first code, which has the same value.
    idx = torch.where(
        a < 2.0,
        torch.round(a * 8.0),
        torch.where(
            a < 4.0,
            torch.round(a * 4.0) + 8.0,
            torch.round(a * 2.0).clamp(max=15.0) + 16.0,
        ),
    ).to(torch.int32)
    codes = (idx | ((v < 0).to(torch.int32) << 5)).reshape(n, k // 4, 4)
    # 4 codes are 24 bits, 3 bytes in little-endian order.
    u = (
        codes[..., 0]
        | (codes[..., 1] << 6)
        | (codes[..., 2] << 12)
        | (codes[..., 3] << 18)
    )
    b = torch.stack((u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF), dim=-1).to(
        torch.uint8
    )
    # [rg, r, st, g, 24 bytes] -> [rg, st, g, r, 24 bytes], so lane 16 g + r.
    b = b.reshape(n // 16, 16, k // 128, 4, 24).permute(0, 2, 3, 1, 4)
    tiles = torch.cat(
        (
            b[..., :16].reshape(n // 16, k // 128, 1024),
            b[..., 16:].reshape(n // 16, k // 128, 512),
        ),
        dim=-1,
    )
    # [rg, r, st / 4, st % 4, g] -> [rg, st / 4, g, r, st % 4].
    sc = (exp + 127).to(torch.uint8).reshape(n // 16, 16, k // 512, 4, 4)
    sc = sc.permute(0, 2, 4, 1, 3)
    return tiles.reshape(-1).contiguous(), sc.reshape(-1).contiguous()


def projections(layer) -> list:
    """The linears of a decoder layer that get an MXFP6 copy: out_proj or
    o_proj, and in a GDN layer in_proj_qkvz."""
    if layer.layer_type == "linear_attention":
        return [layer.linear_attn.out_proj, layer.linear_attn.in_proj_qkvz]
    return [layer.self_attn.o_proj]


def fp6_copy(linear) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The MXFP6 copy (weight bytes, scale bytes) that ``prepare`` packed for
    ``linear``, None if it has none."""
    return getattr(linear, "specialized_decode_fp6", None)


@torch.no_grad()
def prepare(model, rows: int = 1024) -> None:
    """Rounds the projections of ``model`` (a ``Qwen3_5Model``) to MXFP6 and
    packs their copies for the kernels, ``rows`` rows at a time, which keeps
    the temporaries small. A slice of whole 16-row groups is one contiguous
    byte range of both buffers, at 3/4 and 1/32 bytes a weight. The loaded
    model check (``guards.model_reasons``) has already refused a model whose
    projections are not bf16."""
    from vllm.distributed import get_tensor_model_parallel_rank

    round_projections(model)
    count, nbytes = 0, 0
    for layer in model.layers:
        for lin in projections(layer):
            w = lin.weight
            n, k = w.shape
            fp6 = (
                torch.empty(n * k * 3 // 4, dtype=torch.uint8, device=w.device),
                torch.empty(n * k // 32, dtype=torch.uint8, device=w.device),
            )
            for r0 in range(0, n, rows):
                part_w, part_s = fp6_pack(w.data[r0 : r0 + rows])
                at_w, at_s = r0 * k * 3 // 4, r0 * k // 32
                fp6[0][at_w : at_w + part_w.numel()].copy_(part_w)
                fp6[1][at_s : at_s + part_s.numel()].copy_(part_s)
            lin.specialized_decode_fp6 = fp6
            count += 1
            nbytes += fp6[0].numel() + fp6[1].numel()
    if get_tensor_model_parallel_rank() == 0:
        logger.info(
            "Qwen3.8 specialized decode: packed MXFP6 copies of %d projections, "
            "%.2f GB a GPU",
            count,
            nbytes / 1e9,
        )
