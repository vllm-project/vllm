# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP6 E2M3 rounding of the Qwen3.8 projections.

The rounding keeps every GEMM as it is and only changes values: the bf16
weights become bf16 copies of their MXFP6 values. So the prefill path,
which multiplies the bf16 weights, and the mono decode kernels, which read
the packed MXFP6 copies that mono_decode.pack_fp6 makes, use the same
weights.

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

VLLM_QWEN38_FAKE_MXFP6_PROJ=1 rounds the bf16 in and out projections once
after loading.
"""

import os

import regex as re
import torch

BLOCK = 32
E2M3_MAX = 7.5

ROUND_PROJ = os.environ.get("VLLM_QWEN38_FAKE_MXFP6_PROJ", "0") == "1"

# The projections that the FP6 mono kernels are meant to read: the GDN
# in_proj_qkvz and out_proj and the attention qkv_proj and o_proj of the
# main layers. in_proj_ba is 0.5 MB a GPU and stays bf16, as it does with
# the online MXFP4 targets. The MTP layer, the router gate, the shared
# expert and lm_head are left alone.
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
            "VLLM_QWEN38_FAKE_MXFP6_PROJ=1 found no bf16 projection to round. "
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
