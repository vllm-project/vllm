# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One MoE decoder layer's kernel arguments, built from the loaded vLLM modules.

Attention / indexer / router weights are repacked into the kernels' tiled
formats (copies). The routed experts are read in place: vLLM's AITER a4w4 MXFP4
storage (aiter ``shuffle_weight`` values, ``e8m0_shuffle`` scales) is the
kernels' ATOM layout. The shared expert, a separate block-FP8 MLP in vLLM, is
re-quantized to MXFP4 the way ``VLLM_FP8_MOE_REQUANT`` quantizes the routed
experts and passed through its own pointers.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.online.fp8_requant import (
    dequant_block_fp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.consensus import (  # noqa: E501
    MonoUnsupported,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.weights.config import (  # noqa: E501
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.weights.layer import (  # noqa: E501
    LayerWeights,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.weights.packing import (  # noqa: E501
    pack_bf16,
    pack_fp8,
)
from vllm.models.mistral_large4.amd.specialized_decode.runtime.variants import (
    KernelVariant,
)

FP8 = torch.float8_e4m3fn
N_ROUTED = 128
INDEX_HEADS = 32
INDEX_DIM = 128
FP8_BLOCK = [128, 128]


class LayerKind:
    """Kernel variant of one MoE layer (``variants.load_variant`` + build flags)."""

    ROPE = "rope"  # RoPE, own indexer
    ROPE_SHARED = "rope_shared"  # RoPE, top-k of the preceding indexer layer
    NOPE = "nope"  # NoPE, own indexer
    NOPE_KVS = "nope_kvs"  # NoPE, own indexer, reads the source layer's latent cache


def layer_kind(spec) -> str:
    if spec.use_rope:
        if spec.shares_kv:
            raise MonoUnsupported(f"layer {spec.layer_idx}: RoPE layer sharing KV")
        if spec.has_indexer and spec.indexer_use_rope:
            return LayerKind.ROPE
        if not spec.has_indexer and spec.use_sparse:
            return LayerKind.ROPE_SHARED
    elif spec.has_indexer and not spec.indexer_use_rope:
        return LayerKind.NOPE_KVS if spec.shares_kv else LayerKind.NOPE
    raise MonoUnsupported(f"layer {spec.layer_idx}: unsupported layer type {spec}")


# kind -> (NoPE, KV-share, fused indexer)
KINDS = {
    LayerKind.ROPE: (False, False, True),
    LayerKind.ROPE_SHARED: (False, False, False),
    LayerKind.NOPE: (True, False, True),
    LayerKind.NOPE_KVS: (True, True, True),
}


def moe_kinds(layers) -> list[str]:
    """Kernel variant per MoE layer (1..); raises when the layer mix is unsupported."""
    if layers[0].self_attn.spec.is_moe or not all(
        lyr.self_attn.spec.is_moe for lyr in layers[1:]
    ):
        raise MonoUnsupported("layer mix: expected one dense layer, then MoE layers")
    kinds = [layer_kind(lyr.self_attn.spec) for lyr in layers[1:]]
    last_indexer = None
    for i, kind in enumerate(kinds):
        if kind == LayerKind.ROPE_SHARED and last_indexer != LayerKind.ROPE:
            # the top-k is read from the RoPE package's scratch
            raise MonoUnsupported(f"layer {i + 1}: shared top-k after {last_indexer}")
        if KINDS[kind][2]:
            last_indexer = kind
    return kinds


def check_moe_layer(layer, npes: int = 4) -> None:
    """The weight checks of ``build_moe_layer`` without building anything."""
    attn = layer.self_attn
    if attn.num_local_heads * npes != 64 or attn.v_head_dim != 128:
        raise MonoUnsupported("attention geometry")
    _fp8_param(attn.q_proj)
    _fp8_param(attn.o_proj)
    w = attn.kv_b_proj.weight
    if w.dtype != torch.bfloat16 or w.shape[-1] != 512:
        raise MonoUnsupported(f"kv_b_proj {w.dtype} {tuple(w.shape)}")
    if attn.ssmax is None:
        raise MonoUnsupported("attention without SSMax")
    ix = attn.indexer
    if ix is not None:
        if ix.ssmax is None or ix.n_head != INDEX_HEADS or ix.head_dim != INDEX_DIM:
            raise MonoUnsupported("indexer geometry")
        _fp8_param(ix.wq_b)
    if layer.mlp.shared_experts is None:
        raise MonoUnsupported("MoE layer without a shared expert")
    _fp8_param(layer.mlp.shared_experts.gate_up_proj)
    _fp8_param(layer.mlp.shared_experts.down_proj)
    _routed_experts(layer.mlp.experts)


def _q128(w: torch.Tensor, block_cols: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [r, c] -> (fp8, fp32 scale per 128 x block_cols block), amax / 448."""
    r, c = w.shape
    w = w.float()
    wp = F.pad(w, (0, (-c) % block_cols, 0, (-r) % 128))
    b = wp.view(wp.shape[0] // 128, 128, wp.shape[1] // block_cols, block_cols)
    s = b.abs().amax((1, 3)).clamp_min(1e-12) / 448.0
    q = (b / s[:, None, :, None]).clamp(-448, 448).view_as(wp)[:r, :c].to(FP8)
    return q.contiguous(), s.contiguous()


def _fp8_param(linear) -> tuple[torch.Tensor, torch.Tensor]:
    """A block-FP8 linear's [N, K] fp8 weight and fp32 [N/128, K/128] scales."""
    w = linear.weight
    s = getattr(linear, "weight_scale_inv", None)
    if s is None:
        s = getattr(linear, "weight_scale", None)
    if w.dtype != FP8 or s is None or s.dtype != torch.float32:
        raise MonoUnsupported(f"{type(linear).__name__}: expected block-FP8 weights")
    n, k = w.shape[-2:]
    if tuple(s.shape) != ((n + 127) // 128, (k + 127) // 128):
        # transposed storage (a kernel that keeps W^T) is not read here
        raise MonoUnsupported(f"block scales {tuple(s.shape)} for weight {(n, k)}")
    return w.contiguous(), s.contiguous()


def _shared_expert_mxfp4(mlp) -> dict[str, torch.Tensor]:
    """The shared expert re-quantized to MXFP4 in vLLM's a4w4 storage layout."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops

    def mx(linear):
        w, s = _fp8_param(linear)
        q, sc = mxfp4_quantize(dequant_block_fp8_to_bf16(w, s, FP8_BLOCK))
        return q.view(torch.uint8), sc.view(torch.uint8)

    ug, s_ug = mx(mlp.gate_up_proj)  # rows [gate; up], as w13
    dn, s_dn = mx(mlp.down_proj)
    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = rocm_aiter_ops.shuffle_weights(
        ug[None].contiguous().view(fp4), dn[None].contiguous().view(fp4)
    )
    out = {
        "w_ug_sh": w13.view(torch.uint8).contiguous(),
        "w_dn_sh": w2.view(torch.uint8).contiguous(),
        "s_ug_sh": e8m0_shuffle(s_ug.contiguous()).contiguous(),
        "s_dn_sh": e8m0_shuffle(s_dn.contiguous()).contiguous(),
    }
    return out


def _routed_experts(experts) -> dict[str, torch.Tensor]:
    """Views of vLLM's routed-expert storage (no copy)."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    experts = getattr(experts, "routed_experts", experts)  # shared-expert wrapper
    method = getattr(experts, "quant_method", None)
    method = getattr(method, "old_quant_method", method)  # modular-kernel wrapper
    backend = getattr(method, "mxfp4_backend", None)
    if backend != Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
        raise MonoUnsupported(
            f"routed experts on {backend} ({type(experts).__name__}/"
            f"{type(method).__name__}); mono reads AITER a4w4 storage "
            "(VLLM_FP8_MOE_REQUANT=mxfp4 VLLM_FP8_MOE_REQUANT_ACT=a4w4)"
        )
    out = {}
    for name, key in (
        ("w13_weight", "w_ug"),
        ("w13_weight_scale", "s_ug"),
        ("w2_weight", "w_dn"),
        ("w2_weight_scale", "s_dn"),
    ):
        t = getattr(experts, name).data
        if t.shape[0] != N_ROUTED or not t.is_contiguous():
            raise MonoUnsupported(f"{name} {tuple(t.shape)}")
        v = t.view(torch.uint8)
        if key.startswith("w"):
            v.is_shuffled = True  # aiter shuffle_weight layout (backend checked above)
        out[key] = v
    return out


@dataclass
class MoeLayer:
    """One MoE layer's kernel inputs (``args``: launch-order tensors by name)."""

    layer_id: int
    kind: str
    args: dict[str, torch.Tensor]
    params: torch.Tensor  # int64 pointer table (indexer weights, shared expert)
    attn: Any  # MistralMLAAttention (cache bindings are read per step)


def build_moe_layer(layer, variant: KernelVariant, rank: int, npes: int) -> MoeLayer:
    attn = layer.self_attn
    spec = attn.spec
    kind = layer_kind(spec)
    gc = variant.geometry_config
    C = gc.glm5_tp_config(npes)
    nh, nope, pe, kvl, vd = C.local_heads, C.nope_dim, C.pe_dim, C.kv_lora, C.v_dim
    hidden = C.hidden
    dev = layer.input_layernorm.weight.device
    if (
        attn.num_local_heads != nh
        or attn.qk_nope_head_dim != nope
        or attn.v_head_dim != vd
    ):
        raise MonoUnsupported(f"layer {spec.layer_idx}: attention geometry")

    t: dict[str, torch.Tensor] = {}
    t["w_q_b"], t["s_q_b"] = _fp8_param(attn.q_proj)
    if spec.shares_kv:
        # the kernel reads this launch's latent row from the source cache
        t["w_qkv_a"], t["s_qkv_a"] = _q128(torch.zeros(kvl + pe, hidden, device=dev))
        t["g_kv"] = torch.ones(kvl, dtype=torch.bfloat16, device=dev)
    else:
        wkva = attn.kv_a_proj_with_mqa.weight.float()
        if wkva.shape[0] == kvl:  # NoPE: zero rope-key rows -> cached rope key 0
            wkva = torch.cat([wkva, torch.zeros(pe, hidden, device=dev)])
        t["w_qkv_a"], t["s_qkv_a"] = _q128(wkva)
        t["g_kv"] = attn.kv_a_layernorm.weight.to(torch.bfloat16).contiguous()
    wkvb = attn.kv_b_proj.weight
    if wkvb.dtype != torch.bfloat16 or wkvb.shape != (nh * (nope + vd), kvl):
        raise MonoUnsupported(f"kv_b_proj {wkvb.dtype} {tuple(wkvb.shape)}")
    wkvb = wkvb.float().view(nh, nope + vd, kvl)
    w_uk = wkvb[:, :nope, :].transpose(1, 2).reshape(nh * kvl, nope)
    w_uv = wkvb[:, nope:, :].reshape(nh * vd, kvl)
    t["w_uk"], t["s_uk"] = _q128(w_uk, 64)  # the absorb GEMV reads 128 x 64 scales
    t["w_uv"], t["s_uv"] = _q128(w_uv)
    t["w_o"], t["s_o"] = _fp8_param(attn.o_proj)
    t["g_in"] = layer.input_layernorm.weight.to(torch.bfloat16).contiguous()
    t["g_post"] = layer.post_attention_layernorm.weight.to(torch.bfloat16).contiguous()
    t["w_r"] = layer.mlp.gate.weight.to(torch.bfloat16).contiguous()
    t["bias"] = torch.zeros(N_ROUTED, dtype=torch.float32, device=dev)

    # fp32 parameter table read through the g_q pointer: indexer q LayerNorm,
    # attention and indexer SSMax
    prm = torch.zeros(gc.ML4_P_WORDS, dtype=torch.float32, device=dev)
    if attn.ssmax is None:
        raise MonoUnsupported("attention without SSMax")
    prm[gc.ML4_P_SS_S : gc.ML4_P_SS_S + nh] = attn.ssmax.s.float()
    prm[gc.ML4_P_SS_B : gc.ML4_P_SS_B + nh] = attn.ssmax.b.float()
    ix = attn.indexer
    if ix is not None:
        if ix.ssmax is None or ix.n_head != INDEX_HEADS or ix.head_dim != INDEX_DIM:
            raise MonoUnsupported("indexer geometry")
        prm[gc.ML4_P_IQ_W : gc.ML4_P_IQ_W + hidden] = ix.q_input_norm.weight.float()
        prm[gc.ML4_P_IQ_B : gc.ML4_P_IQ_B + hidden] = ix.q_input_norm.bias.float()
        prm[gc.ML4_P_ISS_S : gc.ML4_P_ISS_S + INDEX_HEADS] = ix.ssmax.s.float()
        prm[gc.ML4_P_ISS_B : gc.ML4_P_ISS_B + INDEX_HEADS] = ix.ssmax.b.float()
        t["w_index_q"], t["s_index_q"] = _fp8_param(ix.wq_b)
        kw = ix.wk_weights_proj.weight
        t["w_index_k"], t["s_index_k"] = _q128(kw[:INDEX_DIM])
        t["w_index_w"] = kw[INDEX_DIM:].to(torch.bfloat16).contiguous()
        t["g_index_k"] = ix.k_norm.weight.float().contiguous()
        t["b_index_k"] = ix.k_norm.bias.float().contiguous()
    t["g_q"] = prm

    t.update(_routed_experts(layer.mlp.experts))
    shared = layer.mlp.shared_experts
    if shared is None:
        raise MonoUnsupported("MoE layer without a shared expert")
    t.update(_shared_expert_mxfp4(shared))

    W = LayerWeights(
        nh, t, C, rank, npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=N_ROUTED,
    )  # fmt: skip
    packed = variant.op.prepare_glm5_weights(W, "fp8_block128")
    if ix is not None:
        packed["w_index_k"] = pack_fp8(t["w_index_k"])
        packed["w_index_q"] = pack_fp8(t["w_index_q"])
        packed["w_index_w"] = pack_bf16(t["w_index_w"])
    args = dict(t, **packed)
    params = torch.tensor(
        variant.op.param_table(args, ix is not None, True),
        dtype=torch.int64,
        device=dev,
    )
    # drop the unpacked copies this build made (packed replacements exist)
    for name in ("w_qkv_a", "w_uk", "w_uv", "w_index_k", "w_index_w"):
        if name in packed and args[name] is not t.get(name):
            t.pop(name, None)
    return MoeLayer(spec.layer_idx, kind, args, params, attn)
