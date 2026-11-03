# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launches of one decoder layer: vLLM's modules and tensors in, the
kernels' arguments out (``Launcher``), and ``qsa_core``, the QSA attention
up to o_proj's input (the wide indexer launch: vLLM's fused prepare, the
scores and the top-k selection; then the sparse attention).

``Launcher`` holds the per-rank state every launch reads: the scratch every
width shares (launches of a step run in order), the block-input buffer, the
two barrier slots the launches of a step alternate (``wide.common.grid``), the
fold counters, the step epoch (bumped once a step; a CUDA graph captures the
bump, and the TP exchange tags carry it), the health words and the peer
mailboxes of the TP exchange."""

from typing import cast

import torch

from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.utils import warmup_rocm_skinny_gemm_workspaces
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.guards import (
    experts_rip,
    gdn_states,
    mxfp4_shuffled,
    router_gate,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide import attention, indexer
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layer as K
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.wide.abi import (
    PEER_HEAD,
    health_words,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    FOLD_BASE,
    SLOT_WORDS,
    sync_words,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import qsa as qsa_
from vllm.models.qwen4_exp.common.qsa import Qwen4ExpQSAMetadata
from vllm.models.qwen4_exp.common.qsa_indexer import QSAIndexerMetadata
from vllm.platforms import current_platform

# Profiling: set True in the worker before the model is built (for example on
# import of a ``--worker-extension-cls`` module) to build every launch with
# ``LayerBuild.trace``: CTA 0 stamps the clock at each of its phase ends into
# ``Launcher.trace`` ([layers, layer.TRACE_WORDS] int64, s_memrealtime ticks at
# 100 MHz; the phase names are ``layer.phase_names(key)``). Off: no stamps are
# compiled.
TRACE = False


def allreduce_codec() -> str:
    """The TP exchange's line codec: QuickReduce's INT4 codec when the
    deployment asks QuickReduce for INT4 (or INT3) all-reduces
    (``VLLM_ROCM_QUICK_REDUCE_QUANTIZATION``), else exact bf16 lines."""
    level = envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION
    return "int4" if level in ("INT4", "INT3") else "bf16"


def _hc_args(hc, prefix: str) -> dict:
    return {
        f"{prefix}_norm": hc.hc_norm.weight,
        f"{prefix}_di": hc.input_mix_weight_down_block_inject.weight,
        f"{prefix}_up": hc.input_mix_weight_up.weight,
    }


def _unused() -> dict:
    """Every ABI argument as 0, the base of a layer's argument set."""
    return dict.fromkeys(K.ABI.names, 0)


class Launcher:
    """Per-rank launch state and the launches of every layer of a step."""

    def __init__(self, model, widths, tp: int, rank: int, group, dev):
        from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import PeerBuffer

        self.model, self.widths, self.tp, self.rank = model, widths, tp, rank
        # zeroed: the MoE's split-unit marks are compared against step tags
        self.scratch = torch.zeros(
            K.scratch_bytes(max(widths)), dtype=torch.uint8, device=dev
        )
        self.x = torch.empty(max(widths), L.HIDDEN, dtype=torch.bfloat16, device=dev)
        # qsa_front's projections of x: qkv_proj's and index_qk_proj's outputs
        self.qkv = torch.empty(max(widths), qsa_.QKV, dtype=torch.bfloat16, device=dev)
        self.ixqk = torch.empty(max(widths), qsa_.IX, dtype=torch.bfloat16, device=dev)
        # vLLM's skinny GEMMs create their workspace pool on first use, which
        # must not be inside a graph capture; the decode steps no longer run
        # one eagerly before the stock graphs are captured
        warmup_rocm_skinny_gemm_workspaces(dev)
        self.sync = torch.zeros(sync_words(), dtype=torch.int32, device=dev)
        self.health = torch.zeros(health_words(L.BLOCKS), dtype=torch.int32, device=dev)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=dev)
        # collective: every TP rank builds its executor at the same point
        self.peers = PeerBuffer(
            PEER_HEAD + exchange.mailbox_bytes(tp), group, rank, tp, dev
        )
        self.peers.bytes.zero_()
        self.peer_table = torch.zeros(4, dtype=torch.int64, device=dev)
        self.peer_table[:tp] = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.flags = self.peers.bytes[: 4 * tp].view(torch.int32)
        base = self.sync.data_ptr()
        self.slots = (base, base + 4 * SLOT_WORDS)
        self.folds = base + 4 * FOLD_BASE
        self.rip = experts_rip(next(ly.mlp for ly in model.layers))
        self.traced = TRACE
        self.allreduce = allreduce_codec()
        self.trace = torch.zeros(
            len(model.layers) * K.TRACE_WORDS, dtype=torch.int64, device=dev
        )
        self.args = [self._static_args(ly) for ly in model.layers]
        self.launches = sum(
            2 if ly.layer_type != "linear_attention" else 1 for ly in model.layers
        )
        if self.launches % 2:
            raise ValueError(
                f"{self.launches} launches a step: the barrier slots need an even count"
            )

    # the arguments every step of a layer shares (weights; caches at bind)
    def _static_args(self, layer) -> dict:
        a = _unused()
        mlp = layer.mlp
        ex = mlp.experts.routed_experts
        a |= _hc_args(layer.attn_hyper_connection, "a")
        a |= _hc_args(layer.mlp_hyper_connection, "m")
        a |= {
            "scratch": self.scratch,
            "w_gate": router_gate(mlp),
            "w13": ex.w13_weight,
            "w13s": ex.w13_weight_scale,
            "w2": ex.w2_weight,
            "w2s": ex.w2_weight_scale,
            "peers": self.peer_table,
            "rank": self.rank,
            "epoch": self.epoch,
            "layer": layer.layer_idx,
            "folds": self.folds,
            "health": self.health,
            "alarm": self.alarm,
            "trace": self.trace if self.traced else 0,
        }
        if layer.layer_type == "linear_attention":
            g = layer.linear_attn
            a |= {
                "w_qkvz": g.in_proj_qkvz.weight,
                "w_qkvz_s": g.in_proj_qkvz.weight_scale,
                "w_ba": g.in_proj_ba.weight,
                "w_conv": g.conv1d.weight,
                "a_log": g.A_log,
                "dt_bias": g.dt_bias,
                "gdn_norm": g.norm.weight,
                "w_out": g.out_proj.weight,
                "w_out_s": g.out_proj.weight_scale,
            }
        else:
            attn = layer.self_attn
            a |= {
                "w_out": attn.o_proj.weight,
                "w_qkv": attn.qkv_proj.weight,
                "w_ix": attn.indexer.index_qk_proj.weight,
                "qkv_out": self.qkv,
                "ix_out": self.ixqk,
            }
        return a

    def bind_caches(self) -> None:
        """The GDN states vLLM just bound, into every GDN layer's arguments."""
        for layer, a in zip(self.model.layers, self.args):
            if layer.layer_type == "linear_attention":
                conv_state, rstate = gdn_states(layer.linear_attn)
                a |= {
                    "conv_state": conv_state,
                    "conv_stride": conv_state.stride(0),
                    "ssm": rstate,
                    "ssm_stride": rstate.stride(0),
                }

    def keys(self, layer, rows: int, stats: bool) -> list:
        """The builds a layer's launches use at ``rows`` (``stats``: its input
        state came from vLLM)."""
        common = dict(
            tokens=rows,
            tp=self.tp,
            rip=self.rip,
            hc_eps=layer.attn_hyper_connection.config.rms_norm_eps,
            trace=self.traced,
            allreduce=self.allreduce,
        )
        if layer.layer_type == "linear_attention":
            g = layer.linear_attn
            return [
                K.LayerBuild(
                    kind="gdn",
                    stats=stats,
                    qkvz_shuffled=bool(mxfp4_shuffled(g.in_proj_qkvz)),
                    out_shuffled=bool(mxfp4_shuffled(g.out_proj)),
                    gdn_eps=g.norm.eps,
                    **common,
                )
            ]
        return [
            K.LayerBuild(kind="qsa_front", stats=stats, **common),
            K.LayerBuild(kind="qsa_back", **common),
        ]

    def launch(self, key, i: int, seq: int, dynamic: dict) -> int:
        """Launch ``key`` for layer ``i`` as the step's launch ``seq`` (which
        picks its barrier slot); returns the next ``seq``."""
        a = self.args[i] | dynamic
        a["sync"] = self.slots[seq % 2]
        a["sync_next"] = self.slots[(seq + 1) % 2]
        K.run(key, a)
        return seq + 1


def _wide_qsa(attn) -> bool:
    """The layer's indexer scores and sparse attention run the wide kernels:
    bf16 K/V (one kv head a rank) and bf16 compressed keys in vLLM's paged
    layouts, the fused prepare, no top-k reuse (MTP)."""
    ix = attn.indexer
    return (
        attn.use_fused_qsa_prepare
        and attn.impl.octave_fmt is None
        and not ix.skip_topk
        and attn.kv_cache.dtype == torch.bfloat16
        and attn.num_kv_heads == 1
        and attn.head_dim == attention.HEAD_DIM
        and attn.num_heads <= attention.COLS
        and ix.indexer_dtype == torch.bfloat16
        and ix.compressed_key_cache.kv_cache.dtype == torch.bfloat16
        and ix.index_head_dim == indexer.DIM
    )


_SELECT: dict = {}


def _select_buffers(rows: int, columns: int, blocks: int, device):
    """The scores, visible counts and top-k blocks of a width's rows (one
    set: the layers run in order)."""
    k = (rows, columns, blocks, str(device))
    if k not in _SELECT:
        _SELECT[k] = (
            torch.empty(rows, columns, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
            torch.empty(rows, blocks, dtype=torch.int32, device=device),
        )
    return _SELECT[k]


def _prep(attn, rows, qkv, ixqk, query, gate, positions, main, raw, comp):
    """The score launch's fused prepare (``indexer.Prep``) when the layer's
    rope, caches and metadata are the ones it is written for, else None
    (vLLM's prepare runs)."""
    ix = attn.indexer
    rope = ix.rotary_emb
    section = getattr(rope, "mrope_section", None)
    ring = ix.raw_key_cache.kv_cache
    if ring.dim() == 4 and ring.shape[2] == 1:
        ring = ring.flatten(2)  # [rings, ring rows, 1 kv head, row]
    tail = ix.raw_key_cache.cache_rope_positions
    ok = (
        ix.use_fused_pre_indexer
        and ix.compress_ratio == 4
        and ix.index_n_heads == 4
        and attn.num_heads == indexer.MAIN_HEADS
        and positions.dtype == torch.int64
        and rope.cos_sin_cache.dtype == torch.bfloat16
        and rope.cos_sin_cache.shape[1] == 64
        and (not section or bool(getattr(rope, "mrope_interleaved", False)))
        and ring.dtype == torch.bfloat16
        and ring.dim() == 3
        and ring.shape[2] == (indexer.DIM + 12 if tail else indexer.DIM)
        and (not tail or ix.raw_key_cache.rope_position_offset == indexer.DIM)
        and all(
            t.dtype == torch.int64
            for t in (main.slot_mapping, raw.slot_mapping, comp.slot_mapping)
        )
        and raw.block_table.dtype == torch.int32
    )
    if not ok:
        return None
    return indexer.Prep(
        ixqk[:rows],
        qkv[:rows],
        query[:rows],
        gate[:rows],
        positions,
        rope.cos_sin_cache,
        attn.q_norm.weight,
        attn.k_norm.weight,
        ix.q_layernorm.weight,
        ix.k_layernorm.weight,
        attn.q_norm.variance_epsilon,
        ix.q_layernorm.variance_epsilon,
        (int(section[1]), int(section[2])) if section else None,
        attn.kv_cache,
        main.slot_mapping,
        ring,
        raw.slot_mapping,
        raw.block_table,
        comp.slot_mapping,
    )


def qsa_core(attn, x, qkv, ixqk, positions) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen4ExpQSAAttention.forward`` up to the output gate (the qsa_back
    launch applies it to o_proj's input) -> (attention output [S, heads HD],
    gate [S, heads HD]). ``qkv`` and ``ixqk``: ``attn.qkv_proj(x)`` and
    ``attn.indexer.index_qk_proj(x)`` (qsa_front's projections). The wide
    indexer launch runs vLLM's fused prepare (the queries, the gate, the K/V
    and indexer caches; ``indexer.Prep``), scores the compressed blocks and
    selects them (the decode top-k's set; a row within the budget takes all
    of its blocks unscored), and ``wide.attention`` attends their tokens in
    place (other cache formats run vLLM's whole QSA; other ropes or metadata
    vLLM's prepare)."""
    from vllm.models.qwen4_exp.amd import qsa

    n = x.shape[0]
    name = qsa._encode_layer_name(attn.layer_name)
    out = qkv.new_empty(n, attn.num_heads, attn.head_dim)
    gate = torch.empty_like(out)
    if _wide_qsa(attn):
        ix = attn.indexer
        md = get_forward_context().attn_metadata
        main = cast(Qwen4ExpQSAMetadata, md[attn.layer_name])
        raw = cast(QSAIndexerMetadata, md[ix.raw_key_cache.prefix])
        comp = cast(QSAIndexerMetadata, md[ix.compressed_key_cache.prefix])
        rows = raw.num_actual_tokens
        query = torch.empty_like(out)
        prep = _prep(attn, rows, qkv, ixqk, query, gate, positions, main, raw, comp)
        q_ix = None
        if prep is None:
            q_raw, raw_keys = ixqk[:rows].split(
                (ix.index_n_heads * ix.index_head_dim, ix.index_head_dim), dim=-1
            )
            q_ix, (query, _) = ix._fused_prepare(
                q_raw,
                raw_keys,
                positions[..., :rows],
                raw,
                comp,
                attn,
                qkv,
                main.slot_mapping,
                gate,
            )
        keys = ix.compressed_key_cache.kv_cache
        top = ix.token_topk // ix.compress_ratio
        logits, visible, blocks = _select_buffers(
            n, comp.block_table.shape[1] * keys.shape[1], top, x.device
        )
        logits, visible, blocks = logits[:rows], visible[:rows], blocks[:rows]
        token_to_req = comp.token_to_req[:rows]
        indexer.score(
            q_ix,
            keys,
            comp.block_table,
            token_to_req,
            comp.logical_positions[:rows],
            comp.seq_lens,
            ix.compress_ratio,
            logits,
            visible,
            selected=blocks,
            prep=prep,
        )
        attention.sparse_attention(
            query[:rows],
            attn.kv_cache,
            blocks,
            main.block_table,
            token_to_req,
            comp.seq_lens,
            ix.compress_ratio,
            out[:rows],
            dense=True,
        )
        if rows < n:
            # padded rows: zero output, as vLLM's attention leaves them
            out[rows:].zero_()
            gate[rows:].zero_()
    elif current_platform.opaque_attention_op():
        torch.ops.vllm.qwen4_exp_qsa_prepare_with_output(
            x, positions, qkv, out, gate, name
        )
    else:
        qsa.qwen4_exp_qsa_prepare_with_output(x, positions, qkv, out, gate, name)
    return out.view(n, -1), gate.view(n, -1)
