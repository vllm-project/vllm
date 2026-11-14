# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MonoKernel's weights: views of one decoder layer's parameters as vLLM loaded
them (TP8 rank, after `process_weights_after_loading`). The kernel reads them in
place: no checkpoint read, no copy of the experts or of the BF16 attention / router
matrices.

Kernel contract (`flydsl.weights.LayerWeights.t`; 8 local heads, expert
intermediate 256 of 2048):

  g_in, g_post   input_layernorm / post_attention_layernorm .weight, BF16 [6144]
  g_q, g_kv      self_attn.q_a_layernorm / kv_a_layernorm .weight, BF16 [2048] / [512]
  w_qkv_a        self_attn.fused_qkv_a_proj.weight, BF16 [2624, 6144]
                 (q_a | kv_a | k_pe rows)
  w_q_b          self_attn.q_b_proj.weight, BF16 [2048, 2048]
                 (per head nope 192 | rope 64)
  w_o            self_attn.o_proj.weight, BF16 [6144, 2048]
  w_uk, w_uv     COPIES (3.5 MiB a layer) of self_attn.kv_b_proj.weight
                 [8 * 448, 512] per head [:192]^T -> [4096, 192] and [192:] ->
                 [2048, 512]: MLA's W_UK_T / W_UV are transposed (or FP8 under the
                 AITER FP8 BMM)
  w_r, bias      mlp.gate.weight BF16 [256, 6144]; e_score_correction_bias FP32
                 [256] (widened if vLLM keeps it BF16)
  w_ug, s_ug     mlp.experts.routed_experts .w13_weight [256, 512, 3072] /
                 .w13_weight_scale [256 * 512, 192]
  w_dn, s_dn     .w2_weight [256, 6144, 128] / .w2_weight_scale [256 * 6144, 8]:
                 the AITER_MXFP4_MXFP4 MoE backend's aiter shuffle_weight(16, 16)
                 values and e8m0_shuffle scales (gate rows | up rows)
  w_ug_sh ...    mlp.shared_experts .gate_up_proj / .down_proj (.weight,
                 .weight_scale) under AiterMxfp4LinearKernel: row-major values with
                 scales transposed to [K / 32, rows] (Triton FP4 GEMM, the default),
                 or the experts' shuffles (VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1);
                 `shared_native` says which

MXFP4 is e2m1 low-nibble-first with an E8M0 scale per 32 along K.

FP8 checkpoints (Quark `QuarkW8A8Fp8` per-channel weights, per-token dynamic
activations, e.g. amd/GLM-5.2-Quark-MXFP4-AttnFP8): each FP8 attention linear gives
`w_<name>` = the checkpoint's float8_e4m3fn values as [rows, K] row-major and
`s_<name>` = its F32 scale per output row, never requantized. vLLM's FP8 linear
kernel decides the stored layout (notes in `fp8_linear_rows`): a [K, N] / [N, K]
view is read as is, aiter's (16, 16) preshuffle is un-shuffled (a byte permutation,
copied). FP8 `kv_b_proj` gives w_uv / s_uv (rows of kv_b: a copy of those rows); its
per-row scale runs along w_uk's K, so w_uk is then MLA's BF16 `W_UK_T` (vLLM's own
dequantization, transposed into a copy). The kernel's FP8 matrices are the keys with an
`s_<name>` entry.

FP8 block checkpoints (Quark `QuarkW8A8Fp8PerBlock`, 128 x 128 weight blocks, per-128
dynamic activations, e.g. amd/GLM-5.3-Quark-MXFP4-AttnFP8): `w_<name>` the same
[rows, K] values (rows possibly padded apart by VLLM_ROCM_FP8_PADDING), `s_<name>`
the checkpoint's F32 block scales [ceil(rows / 128), K / 128] (vLLM widens E8M0
scales), a view (`fp8_linear_blocks`). w_uv's scales are regrouped per 64 value rows
(`FP8_BLOCK_ROWS`; a copy of 128 scales), w_uk is MLA's BF16 `W_UK_T` as above. All
FP8 linears of a layer, the indexer's `wq_b` included, use one scale format (the
kernel is built for one).

Refused (`WeightViewError`): a shared expert fused into the routed tensors, any MoE
backend other than AITER_MXFP4_MXFP4 (the expert layout is the backend's), attention
linears quantized any other way (non-Quark block FP8, per-tensor FP8, FNUZ, unknown
FP8 linear kernels), and every shape / dtype / contiguity mismatch: the kernel's
buffer loads are unbounded, so a wrong layout would read garbage rather than fault.
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.config import (
    FP8_ATTENTION,
    FP8_BLOCK,
    FP8_BLOCK_ROWS,
    GLM5_CONFIG,
    HIDDEN,
    KV_LORA,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    V_DIM,
    LayerConfig,
)
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.layout import (
    INDEX_DIM,
    INDEX_HEADS,
)

HEADS = GLM5_CONFIG.local_heads  # at TP8
SUPPORTED_MOE_BACKENDS = ("AITER_MXFP4_MXFP4",)
# kernel attention matrix -> the loaded vLLM linear it is read from
ATTENTION_MODULES = dict(
    zip(FP8_ATTENTION, ("fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj"))
)
FP8 = torch.float8_e4m3fn
# vLLM's FP8 ScaledMM kernels for per-channel weights (kernels/linear/scaled_mm) ->
# how process_weights_after_loading leaves the weight bytes
FP8_LINEAR_LAYOUTS = {
    # shuffle_weight([N, K], (16, 16)); HipbMM keeps a .t() view of it
    "AiterPreshuffledPerTokenFp8ScaledMMLinearKernel": "shuffled",
    "AiterHipbMMPerTokenFp8ScaledMMLinearKernel": "shuffled",
    # the checkpoint's [N, K] storage ([N, K] or a [K, N] view of it)
    "AiterPerTokenFp8ScaledMMLinearKernel": "rows",
    "RowWiseTorchFP8ScaledMMLinearKernel": "rows",
    "ChannelWiseTorchFP8ScaledMMLinearKernel": "rows",
}
# vLLM's FP8 block-scaled kernels on ROCm: process_fp8_weight_block_strategy keeps the
# checkpoint's [N, K] bytes; AITER's B-preshuffle (gfx1250 only) marks the layer
# `aiter_bpreshuffled` after a (16, 16) shuffle
FP8_BLOCK_LINEAR_KERNELS = (
    "AiterFp8BlockScaledMMKernel",
    "TritonFp8BlockScaledMMKernel",
    "BlockWiseTorchFP8ScaledMMLinearKernel",
)


class WeightViewError(RuntimeError):
    pass


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise WeightViewError(f"GLM-5.2 MonoKernel weights: {what}")


def _data(p) -> torch.Tensor:
    return p.data if isinstance(p, torch.nn.Parameter) else p


def _u8(p, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    d = _data(p)
    n = 1
    for x in shape:
        n *= x
    _need(
        d.element_size() == 1,
        f"{name} must hold 1-byte MXFP4 / E8M0 values, got {d.dtype}",
    )
    _need(d.is_contiguous(), f"{name} must be contiguous")
    _need(
        d.numel() == n and d.shape[-1] == shape[-1],
        f"{name} has shape {tuple(d.shape)}, expected {shape} bytes",
    )
    return d.view(torch.uint8).view(shape)


def _bf16(p, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    d = _data(p)
    _need(
        d.dtype is torch.bfloat16,
        f"{name} is {d.dtype}, expected bf16",
    )
    _need(
        tuple(d.shape) == shape, f"{name} has shape {tuple(d.shape)}, expected {shape}"
    )
    _need(d.is_contiguous(), f"{name} must be contiguous (strides {d.stride()})")
    _need(
        not getattr(p, "is_shuffled", False),
        f"{name} BF16 weight must not be preshuffled",
    )
    return d


def unshuffle_16x16(w: torch.Tensor) -> torch.Tensor:
    """Inverse of aiter `shuffle_weight(w, layout=(16, 16))` for 1-byte elements
    (`[N/16][K/32][2][16 rows][16 B]` -> row-major `[N, K]`): a byte permutation
    (copy)."""
    n, k = w.shape
    _need(n % 16 == 0 and k % 32 == 0, f"preshuffled FP8 shape {(n, k)}")
    return w.view(n // 16, k // 32, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(n, k)


def is_fp8(p) -> bool:
    return _data(p).dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def fp8_linear_rows(mod, name: str, shape: tuple[int, int]):
    """(values float8_e4m3fn [rows, K] row-major, scales F32 [rows]) of a loaded Quark
    per-channel FP8 linear: the checkpoint's bytes and scales, never requantized.

    QuarkW8A8Fp8.process_weights_after_loading stores `weight.t()` (a [K, N] view)
    and the scale as [N, 1]; then the FP8 linear kernel may repack: aiter's
    preshuffled / hipBLASLt kernels shuffle_weight the [N, K] bytes (16, 16), the
    per-token CK kernel transposes back to [N, K] (kernels/linear/scaled_mm/aiter.py).
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    sch = getattr(mod, "scheme", None)
    _need(
        type(sch).__name__ == "QuarkW8A8Fp8"
        and getattr(sch, "weight_quant_key", None) == kFp8StaticChannelSym
        and getattr(sch, "activation_quant_key", None) == kFp8DynamicTokenSym,
        f"{name}: FP8 weights need Quark per-channel FP8 with per-token dynamic "
        f"activations (PTPC), got {type(sch).__name__} "
        f"{getattr(sch, 'weight_quant_key', None)} / "
        f"{getattr(sch, 'activation_quant_key', None)}",
    )
    kern = type(getattr(sch, "fp8_linear", None)).__name__
    how = FP8_LINEAR_LAYOUTS.get(kern)
    _need(how is not None, f"{name}: unknown FP8 linear kernel {kern} (layout)")
    w, rows, k = _data(mod.weight), shape[0], shape[1]
    _need(w.dtype is FP8, f"{name} is {w.dtype}, expected float8_e4m3fn (OCP)")
    if tuple(w.shape) == (rows, k) and w.is_contiguous():
        m = w
    elif tuple(w.shape) == (k, rows) and w.t().is_contiguous():
        m = w.t()
    else:
        raise WeightViewError(
            f"GLM-5.2 MonoKernel weights: {name} FP8 weight {tuple(w.shape)} strides "
            f"{w.stride()}, expected {shape} (or its transposed view) dense"
        )
    if how == "shuffled":
        m = unshuffle_16x16(m)
    s = _data(mod.weight_scale)
    _need(
        s.dtype is torch.float32 and s.numel() == rows and s.is_contiguous(),
        f"{name} weight_scale {tuple(s.shape)} {s.dtype}: expected {rows} F32 "
        "per-row scales",
    )
    s = s.view(rows)
    _need(
        bool(torch.isfinite(s).all() and (s > 0).all()),
        f"{name} weight_scale: non-finite or non-positive",
    )
    return m, s


def is_fp8_block(mod) -> bool:
    """A Quark per-block FP8 linear (GLM-5.3's attention), read by fp8_linear_blocks."""
    return type(getattr(mod, "scheme", None)).__name__ == "QuarkW8A8Fp8PerBlock"


def fp8_linear_blocks(mod, name: str, shape: tuple[int, int]):
    """(values float8_e4m3fn [rows, K] row-major, scales F32 [ceil(rows / 128), K /
    128]) of a loaded Quark per-block FP8 linear: the checkpoint's bytes and block
    scales, never requantized (QuarkW8A8Fp8PerBlock keeps both as loaded; vLLM widens
    E8M0 scales to F32)."""
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        FP8BlockParams,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Static128BlockSym,
    )

    sch = getattr(mod, "scheme", None)
    _need(
        is_fp8_block(mod)
        and getattr(sch, "weight_quant_key", None) == kFp8Static128BlockSym,
        f"{name}: FP8 block weights need Quark 128 x 128 block FP8, got "
        f"{type(sch).__name__} {getattr(sch, 'weight_quant_key', None)}",
    )
    kern = type(getattr(sch, "fp8_linear", None)).__name__
    _need(
        kern in FP8_BLOCK_LINEAR_KERNELS,
        f"{name}: unknown FP8 block linear kernel {kern} (layout)",
    )
    w, (rows, k) = _data(mod.weight), shape
    _need(w.dtype is FP8, f"{name} is {w.dtype}, expected float8_e4m3fn (OCP)")
    # row-major; VLLM_ROCM_FP8_PADDING (default on) leaves rows of a multiple of
    # 512 bytes 256 bytes apart (the op's tile packing reads through the padding)
    _need(
        tuple(w.shape) == (rows, k) and w.stride(1) == 1 and w.stride(0) >= k,
        f"{name} FP8 weight {tuple(w.shape)} strides {w.stride()}, expected {shape} "
        "row-major",
    )
    if getattr(mod, "aiter_bpreshuffled", False):
        w = unshuffle_16x16(w)
    s = _data(FP8BlockParams.from_layer(mod).block_scale)
    blocks = (-(-rows // FP8_BLOCK), k // FP8_BLOCK)
    _need(
        s.dtype is torch.float32 and tuple(s.shape) == blocks and s.is_contiguous(),
        f"{name} block scale {tuple(s.shape)} {s.dtype}: expected F32 {blocks}",
    )
    _need(
        bool(torch.isfinite(s).all() and (s > 0).all()),
        f"{name} block scale: non-finite or non-positive",
    )
    return w, s


def uv_block_scales(s: torch.Tensor, heads: int = HEADS) -> torch.Tensor:
    """kv_b_proj's [heads * 448 / 128, 4] block scales as w_uv's [heads * 256 / 64, 4]:
    every 64 value rows lie in one of kv_b's 128-row blocks (the value rows of a head
    start at 448 h + 192, both multiples of 64)."""
    r = FP8_BLOCK_ROWS["uv"]
    _need(
        NOPE_DIM % r == 0 and (NOPE_DIM + V_DIM) % r == 0 and FP8_BLOCK % r == 0,
        "uv value rows not aligned to its scale blocks",
    )
    g = torch.arange(heads * V_DIM // r, device=s.device)
    kv_row = (
        (g // (V_DIM // r)) * (NOPE_DIM + V_DIM) + NOPE_DIM + (g % (V_DIM // r)) * r
    )
    return s[kv_row // FP8_BLOCK].contiguous()


def fp8_linear(mod, name: str, shape: tuple[int, int]):
    """(values, scales) of a loaded FP8 attention linear: per-row or block scales."""
    if is_fp8_block(mod):
        return fp8_linear_blocks(mod, name, shape)
    return fp8_linear_rows(mod, name, shape)


def routed_experts_module(mlp):
    """The expert parameters live on the MoERunner's RoutedExperts child."""
    ex = mlp.experts
    rex = getattr(ex, "routed_experts", None)
    return ex if rex is None else rex


def unsupported_moe(mlp) -> str | None:
    """Why the kernel cannot read this layer's MoE weights in place, or None."""
    if getattr(mlp, "shared_experts", None) is None:
        return "no separate shared-expert module (fused shared experts)"
    qm = getattr(routed_experts_module(mlp), "quant_method", None)
    be = getattr(getattr(qm, "mxfp4_backend", None), "value", None)
    if be not in SUPPORTED_MOE_BACKENDS:
        return (
            f"the {be} MoE backend (needs one of {SUPPORTED_MOE_BACKENDS}: the "
            "kernel reads its layout)"
        )
    return None


def weights_from_model(
    layer, cfg: LayerConfig = GLM5_CONFIG
) -> tuple[dict[str, torch.Tensor], dict]:
    """(t, LayerWeights kwargs) of one loaded DeepseekV32DecoderLayer on this rank:
    views of vLLM's tensors (only w_uk / w_uv are copied); never writes them. `cfg`:
    the TP shard's geometry (`glm5_tp_config`: 8 local heads and expert intermediate
    256 at TP8, 16 and 512 at TP4)."""
    heads, inter = cfg.local_heads, cfg.inter
    a, mlp = layer.self_attn, layer.mlp
    why = unsupported_moe(mlp)
    _need(why is None, f"unsupported {why}")
    ex, sh = routed_experts_module(mlp), mlp.shared_experts
    t: dict[str, torch.Tensor] = {}
    for key, mod, n in (
        ("g_in", layer.input_layernorm, HIDDEN),
        ("g_post", layer.post_attention_layernorm, HIDDEN),
        ("g_q", a.q_a_layernorm, Q_LORA),
        ("g_kv", a.kv_a_layernorm, KV_LORA),
    ):
        t[key] = _bf16(mod.weight, key, (n,))
    for key, rows_k in (
        ("qkv_a", (Q_LORA + KV_LORA + PE_DIM, HIDDEN)),
        ("q_b", (heads * (NOPE_DIM + PE_DIM), Q_LORA)),
        ("o", (HIDDEN, heads * V_DIM)),
    ):
        name = ATTENTION_MODULES[key]
        mod = getattr(a, name)
        if is_fp8(mod.weight):
            t[f"w_{key}"], t[f"s_{key}"] = fp8_linear(mod, name, rows_k)
        else:
            t[f"w_{key}"] = _bf16(mod.weight, name, rows_k)
    kv_b_shape = (heads * (NOPE_DIM + V_DIM), KV_LORA)
    if is_fp8(a.kv_b_proj.weight):
        q, s = fp8_linear(a.kv_b_proj, "kv_b_proj", kv_b_shape)
        t["w_uv"] = q.view(heads, -1, KV_LORA)[:, NOPE_DIM:].reshape(
            heads * V_DIM, KV_LORA
        )
        t["s_uv"] = (
            uv_block_scales(s, heads)
            if is_fp8_block(a.kv_b_proj)
            else s.view(heads, -1)[:, NOPE_DIM:].reshape(heads * V_DIM)
        )
        uk_t = getattr(a, "W_UK_T", None)
        _need(
            isinstance(uk_t, torch.Tensor)
            and _data(uk_t).dtype is torch.bfloat16
            and tuple(uk_t.shape) == (heads, NOPE_DIM, KV_LORA),
            "FP8 kv_b_proj needs MLA's BF16 W_UK_T [8, 192, 512] (absent or "
            "requantized under VLLM_ROCM_USE_AITER_FP8BMM / _FP4BMM: unset them)",
        )
        t["w_uk"] = (
            _data(uk_t).transpose(1, 2).reshape(heads * KV_LORA, NOPE_DIM).contiguous()
        )
    else:
        by_head = _bf16(a.kv_b_proj.weight, "kv_b_proj", kv_b_shape).view(
            heads, -1, KV_LORA
        )
        t["w_uk"] = (
            by_head[:, :NOPE_DIM]
            .transpose(1, 2)
            .reshape(heads * KV_LORA, NOPE_DIM)
            .contiguous()
        )
        t["w_uv"] = by_head[:, NOPE_DIM:].reshape(heads * V_DIM, KV_LORA).contiguous()
    gw = _data(mlp.gate.weight)
    _need(
        tuple(gw.shape) == (N_EXPERTS, HIDDEN) and gw.is_contiguous(),
        f"gate.weight {tuple(gw.shape)}",
    )
    t["w_r"] = gw if gw.dtype is torch.bfloat16 else gw.to(torch.bfloat16)
    bias = _data(mlp.gate.e_score_correction_bias)
    _need(
        tuple(bias.shape) == (N_EXPERTS,),
        f"e_score_correction_bias {tuple(bias.shape)}",
    )
    t["bias"] = (
        bias
        if bias.dtype is torch.float32 and bias.is_contiguous()
        else bias.float().contiguous()
    )
    for key, p, shape in (
        ("w_ug", ex.w13_weight, (N_EXPERTS, 2 * inter, HIDDEN // 2)),
        ("s_ug", ex.w13_weight_scale, (N_EXPERTS * 2 * inter, HIDDEN // 32)),
        ("w_dn", ex.w2_weight, (N_EXPERTS, HIDDEN, inter // 2)),
        ("s_dn", ex.w2_weight_scale, (N_EXPERTS * HIDDEN, inter // 32)),
    ):
        t[key] = _u8(p, f"experts.{key}", shape)
    gu_s, dn_s = _data(sh.gate_up_proj.weight_scale), _data(sh.down_proj.weight_scale)
    native = tuple(gu_s.shape) == (HIDDEN // 32, 2 * inter)
    _need(
        native or tuple(gu_s.shape) == (2 * inter, HIDDEN // 32),
        f"shared gate_up weight_scale {tuple(gu_s.shape)}: unknown layout",
    )
    t["w_ug_sh"] = _u8(
        sh.gate_up_proj.weight, "shared gate_up", (2 * inter, HIDDEN // 2)
    )
    t["w_dn_sh"] = _u8(sh.down_proj.weight, "shared down", (HIDDEN, inter // 2))
    s_shapes = (
        ((HIDDEN // 32, 2 * inter), (inter // 32, HIDDEN))
        if native
        else ((2 * inter, HIDDEN // 32), (HIDDEN, inter // 32))
    )
    t["s_ug_sh"] = _u8(gu_s, "shared gate_up scale", s_shapes[0])
    t["s_dn_sh"] = _u8(dn_s, "shared down scale", s_shapes[1])
    return t, dict(shared_native=native)


def index_weights_from_model(attn) -> dict[str, torch.Tensor] | None:
    """The fused indexer's projections, or None when the checkpoint's `wq_b` is not
    FP8 (the fused indexer then stays off): index Q = the FP8 `wq_b` [4096, 2048] +
    per-row or block scales (as fp8_linear), index K / head weights = row views of
    vLLM's BF16 `wk_weights_proj` [128 | 32, 6144] (vLLM dequantizes an FP8 `wk` itself
    at load, deepseek_v2._try_load_fp8_indexer_wk), k_norm F32 [128] weight / bias."""
    ind = attn.indexer
    if not is_fp8(ind.wq_b.weight):
        return None
    t: dict[str, torch.Tensor] = {}
    t["w_index_q"], t["s_index_q"] = fp8_linear(
        ind.wq_b, "indexer.wq_b", (INDEX_HEADS * INDEX_DIM, Q_LORA)
    )
    kw = _bf16(
        ind.wk_weights_proj.weight,
        "indexer.wk_weights_proj",
        (INDEX_DIM + INDEX_HEADS, HIDDEN),
    )
    t["w_index_k"], t["w_index_w"] = kw[:INDEX_DIM], kw[INDEX_DIM:]
    for key, p in (("g_index_k", ind.k_norm.weight), ("b_index_k", ind.k_norm.bias)):
        d = _data(p)
        _need(tuple(d.shape) == (INDEX_DIM,), f"indexer.k_norm {key} {tuple(d.shape)}")
        t[key] = d if d.dtype is torch.float32 and d.is_contiguous() else d.float()
    return t
