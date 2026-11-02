# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-copy views (and two converted copies) of one vLLM ``MiniMaxM3DecoderLayer``
for the mono kernels.

The kernels read tensors through raw pointers in fixed layouts: a differently
laid-out weight would not fault, it would produce plausible garbage. So every
layout assumption is asserted once and a model that does not match is refused
(``MonoUnsupported``). The experts already have the kernels' layout (MXFP4, aiter
``shuffle_weight(16, 16)`` + ``e8m0_shuffle`` scales, fused shared expert in slot
128). The attention linears are MXFP8 in vLLM and the router gate is fp32, so
those are converted once when the runner is built: MXFP8 -> bf16 -> per-channel
fp8, preshuffled (16, 16); gate -> bf16.

Without shared-expert fusion the routed table holds the 128 routed experts and
the shared expert stays the separate MXFP8 MLP; its weights are copied once into
the f8f6f4 MFMA's fp8 A-operand order (``mxfp8_a_tiles``, the same bytes) with
``e8m0_shuffle``-ordered scales, so the kernels run it in MXFP8.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from aiter import dtypes
from aiter.ops.quant import pertoken_quant
from aiter.ops.shuffle import shuffle_weight

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.models.minimax_m3.amd.mono.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ONE_INDEX_HEAD,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
    TP,
    IndexHeads,
    MonoUnsupported,
)
from vllm.models.minimax_m3.amd.mono.kernels.post_attn import SH_ARGS


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _bf16_vec(t: torch.Tensor, n: int, name: str) -> torch.Tensor:
    _need(
        t.dtype == torch.bfloat16 and t.numel() == n and t.is_contiguous(),
        f"{name}: {t.dtype} {tuple(t.shape)}",
    )
    return t.data


@dataclass(frozen=True)
class SparseMoeLayer:
    """Everything the layer kernel reads for one sparse-attention MoE layer."""

    layer_id: int
    attn: object  # MiniMaxM3SparseAttention: layer name, KV cache views, scales
    index_cache: torch.Tensor  # [blocks, block, head_dim] fp8, shared block ids
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
    # the separate MXFP8 shared expert (``mxfp8_a_tiles``) and the int64 table of
    # their addresses the kernels take (``SH_ARGS`` order); None when it is fused
    sh_w13: torch.Tensor | None = None
    sh_s13: torch.Tensor | None = None
    sh_w2: torch.Tensor | None = None
    sh_s2: torch.Tensor | None = None
    sh_args: torch.Tensor | None = None

    @property
    def shared_fp8(self) -> bool:
        return self.sh_args is not None

    def shared_bytes(self) -> int:
        """Bytes of the kernels' copy of the separate shared expert."""
        if not self.shared_fp8:
            return 0
        nbytes = 0
        for t in (self.sh_w13, self.sh_s13, self.sh_w2, self.sh_s2):
            assert t is not None
            nbytes += t.numel()
        return nbytes


def _ptpc_fp8_from_mxfp8(
    linear, rows: int, cols: int, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 linear (fp8 [rows, cols], e8m0 [rows, cols/32]) -> per-channel fp8
    preshuffled (16, 16) + fp32 scale [rows], as the kernels' PTPC GEMV reads."""
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    _need(
        w is not None and w.dtype == dtypes.fp8 and tuple(w.shape) == (rows, cols),
        f"{name}: weight {None if w is None else (w.dtype, tuple(w.shape))}",
    )
    _need(
        s is not None and tuple(s.shape) == (rows, cols // 32),
        f"{name}: scale {None if s is None else tuple(s.shape)}",
    )
    assert w is not None and s is not None
    bf = dequant_mxfp8_to_bf16(w.data, s.data.view(torch.uint8))
    q, scale = pertoken_quant(bf, quant_dtype=dtypes.fp8)
    return (
        shuffle_weight(q, layout=(16, 16)).contiguous(),
        scale.view(rows).to(torch.float32).contiguous(),
    )


def mxfp8_a_tiles(
    linear, rows: int, cols: int, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 linear (fp8 [rows, cols], e8m0 [rows, cols/32]) -> its bytes in the
    f8f6f4 MFMA's fp8 A order and its scales in ``e8m0_shuffle`` order.

    A 16-row x 128-k tile is 2 KiB; lane g * 16 + r of a tile holds row r's k
    16 g .. + 16 then 64 + 16 g .. + 16 (the B operand's order), 32 bytes."""
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    _need(
        w is not None and w.dtype == dtypes.fp8 and tuple(w.shape) == (rows, cols),
        f"{name}: weight {None if w is None else (w.dtype, tuple(w.shape))}",
    )
    _need(
        s is not None and tuple(s.shape) == (rows, cols // 32),
        f"{name}: scale {None if s is None else tuple(s.shape)}",
    )
    assert w is not None and s is not None
    # the row-major layout the native MXFP8 linear kernel keeps; a kernel that
    # preshuffles or pads would make the tiles silently wrong
    _need(
        not getattr(w, "is_shuffled", False) and not getattr(s, "is_shuffled", False),
        f"{name}: weight preshuffled",
    )
    _need(rows % 32 == 0 and cols % 256 == 0, f"{name}: shape {rows} x {cols}")
    tiles = (
        w.data.contiguous()
        .view(torch.uint8)
        .view(rows // 16, 16, cols // 128, 2, 4, 16)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(-1)
    )
    scales = (
        s.data.contiguous()
        .view(torch.uint8)
        .view(rows // 32, 2, 16, cols // 256, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(-1)
    )
    return tiles, scales


def _ptpc_fp8_from_bf16(
    bf: torch.Tensor, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [rows, cols] -> per-channel fp8 preshuffled (16, 16) + fp32 scale [rows]."""
    q, scale = pertoken_quant(bf, quant_dtype=dtypes.fp8)
    return (
        shuffle_weight(q, layout=(16, 16)).contiguous(),
        scale.view(rows).to(torch.float32).contiguous(),
    )


def _build_cp_qkv(
    linear, rank: int, tp_group, layer_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The wider fused projection of indexer context parallelism: every rank's
    index q head, all-gathered once at build time (``IndexHeads(TP, rank)``).

    One rank's rows (2560): q(2048) | k(128) | v(128) | index_q(128) | index_k(128)
    CP rows (2944):         q(2048) | k(128) | v(128) | iq_rank0..3 | index_k(128)

    The gather's list order is the TP group's rank order, which ``IndexHeads.own``
    indexes; the rank's own slot is checked against its local rows."""
    one_rows = (LOCAL_Q_HEADS + 3 + 1) * HEAD_DIM  # 2560
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    _need(
        w is not None
        and w.dtype == dtypes.fp8
        and tuple(w.shape) == (one_rows, HIDDEN),
        f"layer {layer_id} qkv_proj CP: weight "
        f"{None if w is None else (w.dtype, tuple(w.shape))}",
    )
    _need(
        s is not None and tuple(s.shape) == (one_rows, HIDDEN // 32),
        f"layer {layer_id} qkv_proj CP: scale {None if s is None else tuple(s.shape)}",
    )
    assert w is not None and s is not None
    _need(
        dist.get_rank(tp_group) == rank and dist.get_world_size(tp_group) == TP,
        f"layer {layer_id} qkv_proj CP: TP group rank {dist.get_rank(tp_group)}"
        f" of {dist.get_world_size(tp_group)}, not {rank} of {TP}",
    )
    bf = dequant_mxfp8_to_bf16(w.data, s.data.view(torch.uint8))

    iq_start = (LOCAL_Q_HEADS + 2) * HEAD_DIM  # 2304
    iq_end = iq_start + HEAD_DIM  # 2432
    my_iq = bf[iq_start:iq_end].contiguous()

    all_iq = [torch.empty_like(my_iq) for _ in range(TP)]
    dist.all_gather(all_iq, my_iq, group=tp_group)
    _need(
        torch.equal(all_iq[rank], my_iq),
        f"layer {layer_id} qkv_proj CP: gathered index q heads not in rank order",
    )

    cp_bf = torch.cat([bf[:iq_start], *all_iq, bf[iq_end:]], dim=0)
    cp_rows = (LOCAL_Q_HEADS + 3 + TP) * HEAD_DIM  # 2944
    assert cp_bf.shape[0] == cp_rows, f"CP QKV rows {cp_bf.shape[0]} != {cp_rows}"
    return _ptpc_fp8_from_bf16(cp_bf, cp_rows)


def _cos_sin(attn, layer_id: int) -> torch.Tensor:
    t = attn.rotary_emb.cos_sin_cache
    _need(
        t.dim() == 2 and t.shape[1] == ROTARY_DIM,
        f"layer {layer_id}: cos/sin cache {tuple(t.shape)}",
    )
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.contiguous()


def build_sparse_layer(
    layer,
    layer_id: int,
    heads: IndexHeads = ONE_INDEX_HEAD,
    rank: int = 0,
    tp_group=None,
) -> SparseMoeLayer:
    attn = layer.self_attn
    _need(
        hasattr(attn, "indexer") and layer.is_moe_layer,
        f"layer {layer_id}: not sparse MoE",
    )
    # the model has one index head per rank; under indexer context parallelism
    # the fused projection is built wider (every rank's), see _build_cp_qkv
    _need(
        (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim)
        == (LOCAL_Q_HEADS, 1, 1, HEAD_DIM),
        f"layer {layer_id}: heads "
        f"{attn.num_heads}/{attn.num_kv_heads}/{attn.num_idx_heads}",
    )
    _need(attn.rotary_emb.rotary_dim == ROTARY_DIM, f"layer {layer_id}: rotary_dim")
    sel = attn.indexer.impl  # the selection config lives on the indexer impl
    _need(
        (sel.topk_blocks, sel.init_blocks, sel.local_blocks, attn.skip_index_topk)
        == (TOPK_BLOCKS, 0, 1, False),
        f"layer {layer_id}: sparse selection config",
    )
    _need(
        attn.use_aiter_sparse_pa and str(attn.kv_cache_dtype).startswith("fp8"),
        f"layer {layer_id}: needs the AITER sparse PA path with an fp8 KV cache",
    )
    index_cache = attn.indexer.index_cache.kv_cache
    _need(
        index_cache.numel() > 0
        and index_cache.dtype == dtypes.fp8
        and index_cache.dim() == 3
        and index_cache.shape[2] == HEAD_DIM
        and index_cache.is_contiguous(),
        f"layer {layer_id}: index cache {index_cache.dtype} {tuple(index_cache.shape)}",
    )

    if heads.count > 1:
        _need(tp_group is not None, f"layer {layer_id}: CP requires tp_group")
        w_qkv, s_qkv = _build_cp_qkv(attn.qkv_proj, rank, tp_group, layer_id)
    else:
        w_qkv, s_qkv = _ptpc_fp8_from_mxfp8(
            attn.qkv_proj, heads.rows, HIDDEN, f"layer {layer_id} qkv_proj"
        )

    w_o, s_o = _ptpc_fp8_from_mxfp8(
        attn.o_proj, HIDDEN, O_K, f"layer {layer_id} o_proj"
    )

    moe = layer.block_sparse_moe
    # FusedMoEFactory returns a MoERunner wrapping the RoutedExperts module
    experts = getattr(moe.experts, "routed_experts", moe.experts)
    fused = moe.is_fused_shared_expert_enabled
    E = N_ROUTED + 1 if fused else N_ROUTED
    w13, w2 = experts.w13_weight, experts.w2_weight
    _need(
        w13.shape[0] == E,
        f"layer {layer_id}: {'fused' if fused else 'separate'} shared expert "
        f"(experts {w13.shape[0]})",
    )
    shared = {}
    if not fused:
        mlp = moe.shared_experts
        _need(mlp is not None, f"layer {layer_id}: no shared expert")
        shared["sh_w13"], shared["sh_s13"] = mxfp8_a_tiles(
            mlp.gate_up_proj, 2 * INTER, HIDDEN, f"layer {layer_id} shared gate_up"
        )
        shared["sh_w2"], shared["sh_s2"] = mxfp8_a_tiles(
            mlp.down_proj, HIDDEN, INTER, f"layer {layer_id} shared down"
        )
        shared["sh_args"] = torch.tensor(
            [shared[a].data_ptr() for a in SH_ARGS],
            dtype=torch.int64,
            device=shared["sh_w13"].device,
        )
    _need(experts.top_k == TOP_K, f"layer {layer_id}: top_k {experts.top_k}")
    _need(
        getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
        f"layer {layer_id}: experts not in aiter shuffled layout",
    )
    _need(
        w13.numel() * w13.element_size() == E * 2 * INTER * HIDDEN // 2
        and w2.numel() * w2.element_size() == E * HIDDEN * INTER // 2,
        f"layer {layer_id}: expert weight bytes",
    )
    s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
    _need(
        s13.numel() == E * 2 * INTER * HIDDEN // 32
        and s2.numel() == E * HIDDEN * INTER // 32,
        f"layer {layer_id}: expert scale bytes",
    )
    gate = moe.gate.weight.data
    _need(tuple(gate.shape) == (N_ROUTED, HIDDEN), f"layer {layer_id}: gate")
    bias = moe.e_score_correction_bias
    _need(
        bias is not None and bias.dtype == torch.float32 and bias.numel() == N_ROUTED,
        f"layer {layer_id}: bias",
    )
    return SparseMoeLayer(
        layer_id=layer_id,
        attn=attn,
        index_cache=index_cache,
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
            layer.post_attention_layernorm.weight, HIDDEN, "post_attention_layernorm"
        ),
        gate=gate.to(torch.bfloat16).contiguous(),
        bias=bias.data,
        w13=w13.data,
        s13=s13.data,
        w2=w2.data,
        s2=s2.data,
        **shared,
    )
