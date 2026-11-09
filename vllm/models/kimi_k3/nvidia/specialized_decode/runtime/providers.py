# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer attention providers (input projection, KDA/MLA core, o-projection) and
the helpers `SpecializedDecode.build_providers` uses."""

from typing import Any

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    trtllm_gen_query_heads,
)

from ..cutedsl.dsa.attn_ds import TOPK
from ..cutedsl.dsa.index import DIM, HEADS
from ..cutedsl.dsa.prepared import QUERY_HEADS, PreparedDsa, dsa_on_flashinfer
from ..cutedsl.gemm.decode import DecodeProjection
from ..cutedsl.kda.prepared import PreparedKda
from ..cutedsl.mla.balanced import BalancedMla
from ..cutedsl.mla.dcp import DcpBox, resolve_partials
from ..cutedsl.mla.prepared import PreparedMla
from ..cutedsl.oproj.prepared import PreparedOproj

# Decode batches (tokens) served by the prepared providers: every batch up to 8, the
# padded CUDA-graph batches and each further captured size; others run stock.
BATCHES = (*range(1, 9), 16, 32, 64)
PREPARED_MAX = BATCHES[-1]
# Rows the attention cores (MLA, DSA, their DCP mailboxes and scratch) can prepare:
# 64 requests x 5 DSpark verify tokens. The step stays at PREPARED_MAX rows until the
# KDA, MoE, o-projection and AttnRes providers serve wider batches.
ATTENTION_MAX = 320
# Rows up to which the specialized attention layer beats the stock one at every measured
# context (8K, 32K, 128K, AgentX; GB300, DSpark q5, fp8 KV, input projection through
# o_proj, us per layer), per (layer kind, DCP):
#   K3 MLA   TP8   R12 (60 rows) -5..-110; R16 (80) +12 at 8K, +7 at 32K
#            DCP8  R12 -18..-123; R16 +4 at 8K
#   K3.1 DSA TP8   with the per-row-count input projection (``DecodeProjection``:
#                  cuBLAS past 64 rows) every row count to the cores' maximum (stock /
#                  spec us; 8K, 32K, 128K, AgentX): R16 (80) 136.6 / 107.5, 194.8 /
#                  128.6, 262.6 / 194.2, 231.3 / 157.3; R32 (160) 176.7 / 160.0,
#                  290.4 / 200.9, 452.9 / 338.2, 370.0 / 248.7; R48 (240) 194.1 /
#                  187.3, -, 555.6 / 443.7, 488.8 / 344.9; R64 (320) 233.0 / 238.2
#                  (8K: +2%), 365.9 / 314.6, 658.1 / 539.5, 564.8 / 402.1 (1-GPU TP8
#                  rank, fp8 KV, L2 flushed). The #108 split-K projection (64-column
#                  tiles at every width) lost from R32 at 8K and capped TP8 at R16.
#            DCP8  R64 (320, the cores' maximum) still -60..-71
# The cap sits at the crossover (TP8 R64: the 8K loss is 2%, the AgentX gain 29%),
# never past what the cores prepare (ATTENTION_MAX).
SPECIALIZED_ROWS = {
    ("mla", False): 64,
    ("mla", True): 64,
    ("dsa", False): ATTENTION_MAX,
    ("dsa", True): ATTENTION_MAX,
}


# Wide steps (more rows than the whole specialized step serves, up to ATTENTION_MAX):
# the stock layer loop (AttnRes, o_proj + all-reduce, MoE / dense MLP stay stock: their
# specialized providers stop at PREPARED_MAX), with each attention layer's input
# projection and core specialized up to its kind's crossover (rows; 0: stock), per
# (layer kind, DCP). GB300, DSpark q5, fp8 KV (per layer, and per K3.1 step with
# dummy weights, ms per step):
#   K3.1 DSA  as SPECIALIZED_ROWS (projection + core + stock o_proj: TP8 and DCP8
#             to R64, DCP8 R64 -59..-70 us per layer); step DCP8 24K R16 / R32 / R48
#             31.04 -> 29.72 / 40.74 -> 39.35 / 50.17 -> 47.67, R64 57.64 -> 55.79;
#             TP8 24K R16 30.33 -> 29.55, 4K R16 / R32 29.71 -> 29.85 / 39.09 -> 38.71
#   K3 MLA    the cores lose from R16 at short contexts (SPECIALIZED_ROWS)
#   KDA       per verify kind (``recoverssm``: the deployment's KDA layers keep
#             RecoverSSM records). RecoverSSM verify: KdaSplit, then KdaWide from
#             384 (request, head) items (kda.prepared.WIDE_ITEMS), beats the stock
#             RecoverSSM verify at every batch (TP8 12 heads, FP32 state and conv,
#             q5, us per layer stock / spec: R16 39.1 / 16.1, R64 80.1 / 47.9,
#             R128 139.3 / 81.8; its commit 45.3 -> 35.4 at R64).
#             Plain verify (q state slots): KdaSplit past 16 requests is no clear
#             win over the stock KDA (step with it vs without, TP8 4K R16 -0.95 /
#             R32 +1.84, DCP8 24K R16 +0.08 / R32 +0.57 / R48 +1.77) -> stock.
KDA_WIDE_ROWS = {True: ATTENTION_MAX, False: 0}
WIDE_ROWS = {
    ("dsa", False): SPECIALIZED_ROWS["dsa", False],
    ("dsa", True): SPECIALIZED_ROWS["dsa", True],
    ("mla", False): 0,
    ("mla", True): 0,
}


def step_cap(dsa: bool, dcp: bool) -> int:
    """The most rows a specialized decode step takes: the crossover of every
    attention layer kind of the model (K3: MLA; K3.1: MLA and DSA), within what the
    step's providers serve (PREPARED_MAX)."""
    kinds = ("mla", "dsa") if dsa else ("mla",)
    return min(PREPARED_MAX, *(SPECIALIZED_ROWS[kind, dcp] for kind in kinds))


def wide_caps(dcp: bool, recoverssm: bool = False) -> dict[str, int]:
    """Per attention layer kind, the most rows of a wide step its specialized input
    projection and core take (0: the stock attention module), at most ATTENTION_MAX.
    `recoverssm`: the KDA layers verify with RecoverSSM records."""
    caps = {kind: WIDE_ROWS[kind, dcp] for kind in ("mla", "dsa")}
    caps["kda"] = KDA_WIDE_ROWS[recoverssm]
    assert all(0 <= rows <= ATTENTION_MAX for rows in caps.values()), caps
    return caps


def projection_policies(tp, rows, batches):
    """Split-K (M tile, N tile, split, stages) per execution rows' next tested bin
    (wider batches: 64-row N tiles)."""
    policies = {}
    for batch in batches:
        n = next((b for b in (8, 16, 32) if batch <= b), 64)
        if rows == 6288 and (tp == 8 or n > 8):
            tactic = (64, 64, 1, 6) if n == 64 else (128, n, 2, 5 if n == 32 else 6)
        else:  # TP16 KDA B64: a sixth stage, -0.7 us per layer in the KDA chain
            stages = {8: 8, 16: 10, 32: 8, 64: 6 if rows == 3216 else 5}[n]
            tactic = (64, n, 4 if n == 8 else 2, stages)
        policies[batch] = tactic
    return policies


class SkinnyProjection:
    """B1/B2 skinny GEMM; static K loads two weight tiles before the PDL wait (the
    projections now run `DecodeProjection`; kept for the layer benchmarks)."""

    def __init__(self, weight, fallback):
        self.weight, self.fallback, k = weight, fallback, weight.shape[1]
        tactics = {1: (128, 4, 1), 2: (64, 4, 2)}
        self.configs = {m: SkinnyGemmConfig(m, *t, 8, k) for m, t in tactics.items()}
        for m, cfg in self.configs.items():  # compile before capture
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)

    def run(self, x, out):
        cfg = self.configs.get(x.shape[0])
        if cfg is None:
            return self.fallback.run(x, out)
        return shape_dynamic_skinny_gemm(x, self.weight, cfg, None)


def scratch(runtime, name, shape, dtype=torch.bfloat16):
    if name not in runtime.buffers:
        runtime.buffers[name] = torch.empty(shape, device=runtime.device, dtype=dtype)
    return runtime.buffers[name]


def pack_dsa_projection(attention):
    """One [q_a | kv_a | gate | wk | wv] weight for a K3.1 DSA layer; the stock
    projections become views of it."""
    packed = getattr(attention, "packed_projection", None)
    if packed is not None:
        return packed
    linears = (
        attention.fused_qkv_a_proj,
        attention.g_proj,
        attention.dsa_indexer.wkv_index,
    )
    packed = torch.cat([linear.weight.data for linear in linears])
    row = 0
    for linear in linears:
        rows = linear.weight.shape[0]
        linear.weight.data = packed[row : row + rows]
        row += rows
    attention.packed_projection = packed
    return packed


def attention_kind(attention):
    if hasattr(attention, "in_proj_qkvgfab"):
        return "kda"
    return "dsa" if hasattr(attention, "dsa_indexer") else "mla"


class Attention:
    """Input projection -> KDA/MLA/DSA core -> o-projection with its fused AttnRes."""

    def __init__(self, decoder, runtime, prefetch, mailbox, small_batch, twoshot=None):
        attention, tp = decoder.self_attn, runtime.tp
        self.kind = kind = attention_kind(attention)
        if kind == "dsa":
            weight = pack_dsa_projection(attention)
        else:
            name = "in_proj_qkvgfab" if kind == "kda" else "fused_qkv_a_g_proj"
            weight = getattr(attention, name).weight
        pitch = weight.shape[0]
        # The rows of wide steps this layer's projection and core take (verify rows).
        self.wide = tuple(runtime.wide_batches(kind))
        batches = (*runtime.batches, *self.wide)
        policies = projection_policies(tp, pitch, batches)
        self.projection = DecodeProjection(weight, batches, policies)
        self.core: Any  # PreparedKda, PreparedMla or PreparedDsa
        if kind == "dsa":
            pitches = {b: pitch for b in batches}
            rows = dsa_scratch(runtime, max(batches))
            policy = dict(queries=runtime.queries)
            policy.update(kernels=runtime.fi_kernels or None)
            policy.update(prims_ts_rows=runtime.prims_ts_rows or None)
            if runtime.dcp > 1:
                heads = attention.num_local_heads
                policy.update(
                    dcp=dcp_box(
                        runtime,
                        heads,
                        dsa=True,
                        flashinfer=dsa_on_flashinfer(attention),
                    ),
                    parity=runtime.dsa_layers % 2,
                )
            runtime.dsa_layers += 1
            self.core = PreparedDsa(attention, pitches, rows, **policy)
        elif kind == "kda":
            mode = lambda b: (1 if b == 1 else 2 if b <= 8 else 0) if tp == 8 else 0
            policies = {b: (pitch, mode(b)) for b in runtime.batches}
            q = runtime.q  # verify steps: B requests x q tokens (wide steps: verify)
            verify = [b // q for b in batches if q > 1 and b % q == 0]
            policies.update({(b, q): (pitch, int(tp == 8)) for b in verify})
            self.core = PreparedKda(attention, policies, is_conv_state_dim_first())
        else:
            heads = attention.num_local_heads
            rows = (max(batches), heads, 576)
            query = scratch(runtime, "mla.query", rows, torch.uint8)
            policy = dict(max_context=runtime.max_context)
            policy.update(queries=runtime.queries)
            if runtime.dcp > 1:
                policy.update(
                    dcp=dcp_box(runtime, heads), parity=runtime.mla_layers % 2
                )
            # The step's first MLA layer plans the balanced core for every MLA layer.
            balanced = mla_balanced(runtime, heads)
            policy.update(balanced=balanced, plans=runtime.mla_layers == 0)
            runtime.mla_layers += 1
            pitches = {b: pitch for b in batches}
            self.core = PreparedMla(attention, pitches, query, **policy)
        parity = decoder.layer_idx % 2
        front: tuple | None = None
        if decoder.is_moe_layer:
            front = small_batch.front_views[parity], small_batch.hint_views[parity]
            front += (small_batch.batches,)
        self.output = PreparedOproj(
            decoder, mailbox, runtime.batches, kind != "kda", front, twoshot
        )
        cap = max(batches)
        projected = scratch(runtime, kind + ".projected", (cap, pitch))
        # Shared by every layer kind: the widest kind's rows.
        rows = runtime.attention_rows
        local = scratch(runtime, "attention.local", (rows, 12288 // tp))
        self.prefetch, self.layer = prefetch, decoder
        self.storage = {b: (projected[:b], local[:b]) for b in range(1, cap + 1)}

    def prepare_layout(self, caches):
        if self.kind == "kda":
            self.core.prepare_caches(caches)
        else:
            self.core.prepare_cache(caches[0])

    def run_dsa(self, projected, local, caches, metadata, slot_mapping, q):
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

        attention = self.layer.self_attn
        slots = slot_mapping[attention.mla_attn.mla_attn.layer_name]
        # The packed indexer splits into kernel pages; its metadata builder
        # derives the write slots from their block table.
        indexer = get_forward_context().attn_metadata[
            attention.dsa_indexer.k_cache.prefix
        ]
        assert isinstance(indexer, DeepseekV32IndexerMetadata)
        index_slots = indexer.slot_mapping
        self.core.run(
            projected,
            local,
            cache=caches[0],
            slots=slots,
            index_slots=index_slots,
            metadata=metadata,
            q=q,
        )
        self.prefetch.fork("attention_moe", self.layer.layer_idx, projected.shape[0])

    def run_core(self, projected, local, caches, metadata, slot_mapping, q):
        """The KDA / MLA / DSA core of the projected rows into `local`."""
        b = projected.shape[0]
        if self.kind == "dsa":
            self.run_dsa(projected, local, caches, metadata, slot_mapping, q)
        elif self.kind == "kda":
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
            if q > 1:  # uniform q-token verify rows
                slots = metadata.spec_state_indices_tensor
                accepted = metadata.num_accepted_tokens
                self.core.run(
                    projected,
                    local,
                    caches=caches,
                    slots=slots,
                    accepted=accepted,
                    q=q,
                    deferred=getattr(metadata, "recoverssm_deferred", None),
                )
            else:
                slots = metadata.non_spec_state_indices_tensor
                self.core.run(projected, local, caches=caches, slots=slots)
        else:
            decode = metadata.decode
            tables, lengths = decode.block_table, decode.seq_lens
            # Request metadata shorter than b: the core runs the prepared batch b
            # with the missing rows as empty requests.
            rows = min(b, q * min(lengths.numel(), tables.shape[0]))
            if rows < b:
                local[rows:].zero_()
            slots = slot_mapping[self.layer.self_attn.layer_name]
            mla = dict(cache=caches[0], block_table=tables, lengths=lengths)
            # DCP: lengths are the local shard's, totals the full sequences'.
            mla.update(totals=decode.dcp_tot_seq_lens)
            self.core.run(projected, local, slots=slots, q=q, **mla)
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)

    def run(self, x, out, prefix, blocks, caches, metadata, slot_mapping, q=1):
        b = x.shape[0]
        projected, local = self.storage[b]
        projected = self.projection.run(x, projected)
        self.run_core(projected, local, caches, metadata, slot_mapping, q)
        self.output.run(local, prefix, blocks, out)
        self.prefetch.fork("oproj_down", self.layer.layer_idx, b)
        return out

    def run_wide(self, x, caches, metadata, slot_mapping, q):
        """A wide step's attention (`x`: the normalized rows of a batch in
        `self.wide`): the specialized input projection and core, then the stock
        o_proj with its TP all-reduce, as the stock module ends. Returns its output."""
        projected, local = self.storage[x.shape[0]]
        projected = self.projection.run(x, projected)
        self.run_core(projected, local, caches, metadata, slot_mapping, q)
        attention = self.layer.self_attn
        fused = getattr(attention, "gemm_rs_ar", None)
        if fused is not None and fused.should_run(local):
            return fused.apply(local, attention.o_proj)
        return attention.o_proj(local)[0]


def dsa_scratch(runtime, cap):
    """The DSA layers' shared rows (`cap`: one per prepared batch row): the BF16
    query (heads past the local ones stay zero), the normalized q_c, the indexer
    projection, FP8 query and head weights, and FlashInfer's flat top-k rows,
    their counts, latent (up to 128 heads) and LSE."""
    if "dsa.query" not in runtime.buffers:
        shape = (cap, QUERY_HEADS, 576)
        runtime.buffers["dsa.query"] = torch.zeros(
            shape, device=runtime.device, dtype=torch.bfloat16
        )
    return (
        runtime.buffers["dsa.query"],
        scratch(runtime, "dsa.qc", (cap, 1536)),
        scratch(runtime, "dsa.iq", (cap, HEADS * DIM)),
        scratch(runtime, "dsa.q_fp8", (cap, HEADS, DIM), torch.uint8),
        scratch(runtime, "dsa.weights", (cap, HEADS)),
        (
            scratch(runtime, "dsa.fi_slots", (cap, TOPK), torch.int32),
            scratch(runtime, "dsa.fi_valid", (cap,), torch.int32),
            scratch(runtime, "dsa.fi_latent", (cap * 128 * 512,)),
            scratch(runtime, "dsa.fi_lse", (cap * 128,), torch.float32),
        ),
    )


def mla_balanced(runtime, heads):
    """The model's balanced MLA cores (one plan per step, shared by the MLA
    layers)."""
    if "mla.balanced" not in runtime.buffers:
        dcp = runtime.dcp > 1
        decode_heads = heads * (runtime.dcp if dcp else 1)
        runtime.buffers["mla.balanced"] = BalancedMla(decode_heads, runtime.device, dcp)
    return runtime.buffers["mla.balanced"]


def dcp_box(runtime, heads, dsa=False, flashinfer=False, partials="default"):
    """The model's DCP mailboxes (one set for every MLA layer, one for every DSA
    layer: a q mailbox padded to the sparse decode's heads, BF16 for FlashMLA
    sparse's 64 or 128, FP8 for FlashInfer's trtllm-gen / prims_ts query heads),
    one slot per prepared batch row (at least 64). The partial rows DcpValue
    exchanges: `partials` ("default": the layer kind's, see
    mla.dcp.PARTIALS_DEFAULT)."""
    name = "dsa_dcp_box" if dsa else "dcp_box"
    if getattr(runtime, name) is None:
        from vllm.distributed import get_dcp_group

        group = get_dcp_group()
        assert group.world_size == runtime.dcp == runtime.tp
        from vllm.config import get_current_vllm_config

        args = group.device_group, runtime.device, group.world_size, group.rank_in_group
        parallel = get_current_vllm_config().parallel_config
        interleave = parallel.cp_kv_cache_interleave_size
        q_heads = None
        if dsa and flashinfer:
            q_heads = trtllm_gen_query_heads(group.world_size * heads)
        elif dsa:
            q_heads = 64 if group.world_size * heads <= 64 else 128
        # The wide rows of this box's own layer kind (the MLA box past 64 rows only
        # if WIDE_ROWS ever widens MLA).
        kind = "dsa" if dsa else "mla"
        wide = runtime.wide_batches(kind)
        cap = max(PREPARED_MAX, *runtime.batches, *wide)
        if cap > ATTENTION_MAX:
            raise ValueError(f"DCP mailboxes of {cap} rows: at most {ATTENTION_MAX}")
        box = DcpBox(
            *args,
            heads,
            cap=cap,
            interleave=interleave,
            q_heads=q_heads,
            q_bf16=dsa and not flashinfer,
            partials=resolve_partials(partials, kind),
        )
        setattr(runtime, name, box)
    return getattr(runtime, name)


def pack_front(moe, rank, tp):
    """[this rank's router rows (padded to 16); this rank's latent down shard]."""
    router, down = moe.gate.weight, moe.routed_expert_down_proj.weight
    rows, shard = 896 // tp, 3584 // tp
    router = router[rank * rows : (rank + 1) * rows]
    down = down[rank * shard : (rank + 1) * shard]
    return torch.cat((router, router.new_zeros((-rows % 16, 7168)), down)).detach()
