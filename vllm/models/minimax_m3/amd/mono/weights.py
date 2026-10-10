# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-copy views of one sparse MoE layer, checked against what the kernels read.

The mono kernels read the tensors the original path already loaded, quantized and
shuffled, and they read them through raw pointers: a differently-laid-out weight
would not fault, it would produce plausible garbage. So every assumption is
asserted here once and a model that does not match is refused
(``MonoUnsupported``) rather than misread.

Two tensors are not shared with the original path. The router gate is kept as a
private bf16 copy because vLLM's gate is fp32 and K4's GEMV is bf16, and the QKV
weight is shuffled here when the fp8 kernel that loaded it left it unshuffled.
"""

from dataclasses import dataclass

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.minimax_m3.amd.mono.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
    IndexHeads,
    MonoUnsupported,
)

FP8_DTYPE = torch.float8_e4m3fn

# fp8 linear kernels by what they leave in the weight parameter. The first group
# shuffles on load, so the GEMV can read the parameter itself; the second keeps
# the plain (N, K) layout its GEMM wants, so mono shuffles a private copy.
_PRESHUFFLING_FP8_KERNELS = (
    "AiterPreshuffledPerTokenFp8ScaledMMLinearKernel",
    "AiterHipbMMPerTokenFp8ScaledMMLinearKernel",
)
_PLAIN_FP8_KERNELS = (
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "AiterPerTokenFp8ScaledMMLinearKernel",
)


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _attr(obj, name: str, what: str):
    """``obj.name``, or a refusal naming the type that lacks it.

    Every module mono reads is one vLLM refactor away from moving an attribute, and
    a refusal that names the type is worth far more during bring-up than an
    ``AttributeError`` from three frames down.
    """
    got = getattr(obj, name, None)
    _need(got is not None, f"{what}: {type(obj).__name__} has no {name}")
    return got


def _scaled_mm_kernel(linear):
    """The kernel object a quantized linear will run its GEMM through.

    Where it hangs depends on how the layer was quantized: a checkpoint scheme
    keeps it under the scheme, while online quantization is itself the quant
    method and holds it directly.
    """
    qm = getattr(linear, "quant_method", None)
    scheme = getattr(linear, "scheme", None) or getattr(qm, "scheme", None)
    for owner in (scheme, qm):
        if owner is None:
            continue
        for name in ("fp8_linear", "kernel"):
            kernel = getattr(owner, name, None)
            if kernel is not None:
                return kernel
    return None


def _ptpc_fp8(linear, rows: int, cols: int, name: str):
    """Weight + per-output-channel scale of a ptpc-FP8 linear (per-token
    activation quant, per-channel weight scale), in the layout the GEMVs read.

    Returns the weight already shuffled: the original when the kernel that loaded
    it shuffles in place, else a private copy.
    """
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    _need(w is not None and s is not None, f"{name}: not a quantized linear")
    _need(w.dtype == FP8_DTYPE, f"{name}: weight dtype {w.dtype}")
    # The kernels index the weight by (out, in); which of the two the loader left
    # in the leading dim is the kernel's business, so only the count is checked.
    _need(
        w.numel() == rows * cols,
        f"{name}: weight numel {w.numel()} != {rows * cols}",
    )
    _need(
        s.dtype == torch.float32 and s.numel() == rows,
        f"{name}: scale {s.dtype} numel {s.numel()} (want fp32, {rows})",
    )
    _need(not getattr(linear, "is_output_padded", False), f"{name}: output padded")

    kname = type(_scaled_mm_kernel(linear)).__name__
    _need(
        kname in _PRESHUFFLING_FP8_KERNELS or kname in _PLAIN_FP8_KERNELS,
        f"{name}: unrecognized fp8 kernel {kname} (quant_method "
        f"{type(getattr(linear, 'quant_method', None)).__name__}, scheme "
        f"{type(getattr(linear, 'scheme', None)).__name__})",
    )
    if kname in _PLAIN_FP8_KERNELS:
        shuffled = rocm_aiter_ops.shuffle_weight(
            w.data.t().contiguous(), layout=(16, 16)
        )
        return shuffled, s.data.view(rows)
    return w.data, s.data.view(rows)


def _bf16_vec(t: torch.Tensor, n: int, name: str) -> torch.Tensor:
    _need(
        t.dtype == torch.bfloat16 and t.numel() == n and t.is_contiguous(),
        f"{name}: {t.dtype} {tuple(t.shape)}",
    )
    return t.data


def _cos_sin(attn, layer_id: int) -> torch.Tensor:
    """The [max_pos, rotary_dim] cos|sin table the rotary embedding built."""
    t = _attr(attn.rotary_emb, "cos_sin_cache", f"layer {layer_id} rotary_emb")
    _need(
        t.dtype == torch.bfloat16 and t.dim() == 2 and t.shape[1] == ROTARY_DIM,
        f"layer {layer_id}: cos/sin cache {t.dtype} {tuple(t.shape)}",
    )
    return t.data


def sparse_selection(indexer, layer_id: int) -> tuple[int, int, int]:
    """The indexer's (top-k, init, local) block counts.

    The MSA indexer is a thin surface over an impl and re-exports only what its
    caller needs, so the selection parameters live one level down.
    """
    owner = getattr(indexer, "impl", indexer)
    got = tuple(
        getattr(owner, name, None)
        for name in ("topk_blocks", "init_blocks", "local_blocks")
    )
    _need(
        None not in got,
        f"layer {layer_id}: {type(owner).__name__} exposes no sparse selection "
        f"(topk/init/local blocks {got})",
    )
    return got


@dataclass(frozen=True)
class SparseMoeLayer:
    """Everything K1 / K4 read for one sparse-attention MoE layer."""

    layer_id: int
    attn: object  # MiniMaxM3SparseAttention: layer name, KV cache views, scales
    indexer: object  # MiniMaxM3MSAIndexer: the index cache and its slot mapping
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor

    @staticmethod
    def from_layer(layer, layer_id: int, heads: IndexHeads) -> "SparseMoeLayer":
        """``heads``: the index q heads the deployment's fused projection holds."""
        attn = _attr(layer, "self_attn", f"layer {layer_id}")
        indexer = getattr(attn, "indexer", None)
        _need(
            getattr(layer, "is_moe_layer", False) and indexer is not None,
            f"layer {layer_id}: not a sparse-attention MoE layer",
        )
        _need(
            getattr(attn, "use_aiter_sparse_pa", False),
            f"layer {layer_id}: not on the AITER sparse PA path",
        )
        _need(
            (
                attn.num_heads,
                attn.num_kv_heads,
                indexer.num_index_heads,
                attn.head_dim,
            )
            == (LOCAL_Q_HEADS, 1, heads.count, HEAD_DIM),
            f"layer {layer_id}: heads {attn.num_heads}/{attn.num_kv_heads}/"
            f"{indexer.num_index_heads}/{attn.head_dim}",
        )
        _need(
            attn.rotary_emb.rotary_dim == ROTARY_DIM,
            f"layer {layer_id}: rotary_dim {attn.rotary_emb.rotary_dim}",
        )
        selection = sparse_selection(indexer, layer_id)
        _need(
            selection == (TOPK_BLOCKS, 0, 1),
            f"layer {layer_id}: sparse selection {selection}, "
            f"kernels are built for {(TOPK_BLOCKS, 0, 1)}",
        )
        index_cache = _attr(
            indexer.index_cache, "kv_cache", f"layer {layer_id} index_cache"
        )
        _need(
            index_cache.dtype in (FP8_DTYPE, torch.uint8),
            f"layer {layer_id}: index cache {index_cache.dtype}",
        )

        w_qkv, s_qkv = _ptpc_fp8(
            attn.qkv_proj, heads.rows, HIDDEN, f"layer {layer_id} qkv_proj"
        )
        w_o, s_o = _ptpc_fp8(attn.o_proj, HIDDEN, O_K, f"layer {layer_id} o_proj")

        moe = _attr(layer, "block_sparse_moe", f"layer {layer_id}")
        # ``moe.experts`` orchestrates the step (routing, dispatch, the shared
        # expert); the expert weights and top-k belong to the RoutedExperts it
        # holds. Fall back to the runner itself so a future flattening still works.
        runner = _attr(moe, "experts", f"layer {layer_id} moe")
        experts = getattr(runner, "routed_experts", runner)
        # K4 reads every routed expert on this rank, each INTER columns wide, which
        # is how tensor parallelism shards them. Expert parallelism gives a rank
        # whole experts instead, so the weights are a different shape entirely --
        # refused by name here rather than as a byte count below.
        moe_config = _attr(experts, "moe_config", f"layer {layer_id} experts")
        _need(
            not moe_config.use_ep,
            f"layer {layer_id}: expert parallelism (ep_size {moe_config.ep_size})",
        )
        _need(
            getattr(moe, "is_fused_shared_expert_enabled", False)
            and getattr(moe, "use_aiter_moe_fse", False),
            f"layer {layer_id}: the shared expert is not fused into the routed set",
        )
        top_k = _attr(experts, "top_k", f"layer {layer_id} experts")
        _need(top_k == TOP_K, f"layer {layer_id}: top_k {top_k}, built for {TOP_K}")

        E = N_ROUTED + 1  # the routed experts plus the fused shared one
        w13 = _attr(experts, "w13_weight", f"layer {layer_id} experts")
        w2 = _attr(experts, "w2_weight", f"layer {layer_id} experts")
        _need(
            getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
            f"layer {layer_id}: expert weights are not shuffled",
        )
        _need(
            w13.numel() * w13.element_size() == E * 2 * INTER * HIDDEN // 2
            and w2.numel() * w2.element_size() == E * HIDDEN * INTER // 2,
            f"layer {layer_id}: expert weight bytes "
            f"{w13.numel() * w13.element_size()}/{w2.numel() * w2.element_size()}",
        )
        s13 = _attr(experts, "w13_weight_scale", f"layer {layer_id} experts")
        s2 = _attr(experts, "w2_weight_scale", f"layer {layer_id} experts")
        _need(
            s13.numel() == E * 2 * INTER * HIDDEN // 32
            and s2.numel() == E * HIDDEN * INTER // 32,
            f"layer {layer_id}: expert scale counts {s13.numel()}/{s2.numel()}",
        )

        gate = _attr(moe.gate, "weight", f"layer {layer_id} gate")
        _need(
            tuple(gate.shape) == (N_ROUTED, HIDDEN),
            f"layer {layer_id}: gate {tuple(gate.shape)}",
        )
        bias = getattr(moe, "e_score_correction_bias", None)
        _need(
            bias is not None
            and bias.dtype == torch.float32
            and bias.numel() == N_ROUTED,
            f"layer {layer_id}: routing bias correction",
        )
        # K4's router GEMV is bf16 while vLLM keeps the gate in fp32; a private
        # copy rather than a cast per step.
        gate_bf16 = gate.data.to(torch.bfloat16).contiguous()

        return SparseMoeLayer(
            layer_id=layer_id,
            attn=attn,
            indexer=indexer,
            g_in=_bf16_vec(layer.input_layernorm.weight, HIDDEN, "input_layernorm"),
            w_qkv=w_qkv,
            s_qkv=s_qkv,
            g_q=_bf16_vec(attn.q_norm.weight, HEAD_DIM, "q_norm"),
            g_k=_bf16_vec(attn.k_norm.weight, HEAD_DIM, "k_norm"),
            g_iq=_bf16_vec(attn.index_q_norm.weight, HEAD_DIM, "index_q_norm"),
            g_ik=_bf16_vec(attn.index_k_norm.weight, HEAD_DIM, "index_k_norm"),
            cos_sin=_cos_sin(attn, layer_id),
            w_o=w_o,
            s_o=s_o,
            g_post=_bf16_vec(
                layer.post_attention_layernorm.weight,
                HIDDEN,
                "post_attention_layernorm",
            ),
            gate=gate_bf16,
            bias=bias.data,
            w13=w13.data,
            s13=s13.data,
            w2=w2.data,
            s2=s2.data,
        )
