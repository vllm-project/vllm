# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode at DEP (data-parallel attention at TP1 + expert-parallel
MoE; e.g. DEP16 / DEP32): every rank runs all 96 heads of its own requests, so the
attention needs no collective. The step is the stock layer loop (AttnRes, the EP
MoE with its dispatch / combine, KDA) with every K3.1 DSA layer's attention
specialized end to end for TP1:

  input projection   split-K (or cuBLAS) over the packed [q_a | kv_a | gate | wk |
                     wv] weight (14560 x 7168 BF16, 208 MB per layer: the whole
                     weight on one GPU), per batch (DEP_PROJECTION)
  core               PreparedDsa at 96 heads: the split query (mla.query_wide), the
                     fused FP8 indexer, FlashInfer on the query head-padded to 128
                     (the stock decode's kernel: trtllm-gen on fp8, prims_ts on
                     nvfp4_ds_mla), MlaValue
  o_proj             split-K (or cuBLAS) over the 7168 x 12288 weight (no
                     all-reduce at TP1), per batch (DEP_OPROJ)

Every KDA layer runs its input projection (split-K or cuBLAS, the 49376 x 7168
packed weight), the prepared KDA core at 96 heads (KdaSplit / KdaWide per step
kind and batch: conv, F_B gate, recurrence, gated norm; the RecoverSSM verify and
records where the deployment uses them) and o_proj (as the DSA layers'). On a
MegaMoE deployment every MoE layer runs the EpMoeLayer schedule (moe.ep.layer:
split-K [shared gate_up | latent down] front, router and shared expert on a side
stream beside the front and MegaMoE) up to its FRONT_MAX tokens.

Precision is the stock layer's at every stage (FP32 accumulation, BF16 outputs;
the split-K order differs). K3 MLA layers run stock (per-kind policy:
`DEP_KINDS`)."""

from typing import Any

import torch

from ..cutedsl.gemm.projection import PreparedProjection
from .providers import attention_kind, dsa_scratch, pack_dsa_projection

# Rows (tokens per DP rank: requests x (1 + draft tokens)) the DEP attention takes;
# every batch past it runs the stock layer (the dsa_scratch, prims_ts plans and the
# fused indexer's buffers are sized by it).
DEP_MAX_ROWS = 320
# The DEP attention's rows per KV cache dtype: up to its crossover against the stock
# layer (GB300, DSpark q5, per DSA layer incl. projection and o_proj, L2 flushed),
# both with the core head-padded to 128 (fi_heads) on the stock decode's kernel:
#   fp8 (trtllm-gen)  32K R1 / R4 / R16 / R32 / R64 1.41 / 1.36 / 1.26 / 1.20 /
#     1.10x; AgentX 1.39 / 1.34 / 1.39 / 1.38 / 1.36x
#   nvfp4_ds_mla (prims_ts)  32K R1 / R4 / R16 / R64 1.40 / 1.35 / 1.28 / 1.09x;
#     AgentX 1.37 / 1.34 / 1.36 / 1.33x
DEP_ROWS = dict.fromkeys(("fp8", "fp8_e4m3", "nvfp4_ds_mla"), DEP_MAX_ROWS)
# The attention layer kinds specialized at TP1. KDA at 96 heads (RecoverSSM
# verify + commit, FP32 state and conv, q5, us per layer stock / spec):
# R1 44.2 / 27.7, R4 82.6 / 53.5, R16 220 / 143, R64 792 / 481.
DEP_KINDS = ("dsa", "kda")


def _bins(table, batch):
    """The tactic of `batch` rows from a {max rows: tactic} table."""
    return table[next((n for n in sorted(table) if batch <= n), max(table))]


# Split-K tactics (M tile, N tile, split K, stages) per batch bin of the TP1 input
# projection (N 14560, K 7168: 208 MB) and o_proj (N 7168, K 12288: 176 MB); None:
# cuBLAS (torch.mm, the stock linear's GEMM). GB300, L2 flushed (write), us
# (SOL 25.5 / 21.5 at 8.18 TB/s):
#   projection   5 rows 36.3 (cuBLAS 43.4), 20: 36.9 (45.6), 80: 39.6 (45.5),
#                160: 43.8 (47.3), 320: 79.4 (two 160-row tiles stream the weight
#                twice; cuBLAS 57.0)
#   o_proj       5 rows 31.1 (stock 37.5), 20: 34.2 (39.7), 80: 39.4 (42.7),
#                160: 51.4 (43.6), 320: 57.6 (50.8)
DEP_PROJECTION = {
    8: (128, 8, 1, 6),
    16: (128, 16, 1, 6),
    32: (128, 32, 1, 5),
    64: (128, 64, 1, 4),
    128: (128, 128, 1, 3),
    160: (128, 160, 1, 3),
    DEP_MAX_ROWS: None,
}
DEP_OPROJ = {
    8: (64, 8, 1, 8),
    16: (128, 16, 1, 6),
    32: (128, 32, 1, 5),
    80: (128, 64, 1, 4),
    DEP_MAX_ROWS: None,
}


def dep_policies(table, batches):
    """{batch: tactic} of the split-K batches of `table` (cuBLAS ones left out)."""
    tactics = {b: _bins(table, b) for b in batches}
    return {b: t for b, t in tactics.items() if t is not None}


class DepProjection:
    """A TP1 linear: split-K at the batches its table gives a tactic, else cuBLAS
    (`torch.mm` into the output: the stock linear's FP32-accumulated BF16 GEMM)."""

    def __init__(self, weight, table, batches):
        self.weight = weight
        policies = dep_policies(table, batches)
        self.split = (
            PreparedProjection(weight, policies).prepare() if policies else None
        )
        self.batches = set(policies)

    def run(self, x, out):
        if x.shape[0] in self.batches:
            assert self.split is not None
            return self.split.run(x, out)
        return torch.mm(x, self.weight.t(), out=out)


class DepAttention:
    """One K3.1 DSA layer's attention at TP1: the input projection, the prepared
    core and o_proj for `runtime.wide` rows (the DEP step rows)."""

    def __init__(self, decoder, runtime):
        attention = decoder.self_attn
        assert attention_kind(attention) == "dsa", "DEP specializes the DSA layers"
        self.layer, self.attention = decoder, attention
        weight = pack_dsa_projection(attention)
        batches = runtime.wide
        self.wide = batches
        self.projection = DepProjection(weight, DEP_PROJECTION, batches)
        o_weight = attention.o_proj.weight
        self.output = DepProjection(o_weight, DEP_OPROJ, batches)
        from ..cutedsl.dsa.prepared import PreparedDsa

        cap = max(batches)
        scratch = dsa_scratch(runtime, cap)
        policy = dict(queries=runtime.queries)
        policy.update(kernels=runtime.fi_kernels or None)
        policy.update(prims_ts_rows=runtime.prims_ts_rows or None)
        pitches = {b: weight.shape[0] for b in batches}
        self.core = PreparedDsa(attention, pitches, scratch, **policy)
        device = runtime.device
        empty = lambda *shape: torch.empty(  # noqa: E731
            shape, dtype=torch.bfloat16, device=device
        )
        if "dep.projected" not in runtime.buffers:
            runtime.buffers["dep.projected"] = empty(cap, weight.shape[0])
            runtime.buffers["dep.local"] = empty(cap, attention.num_local_heads * 128)
        self.projected = runtime.buffers["dep.projected"]
        self.local = runtime.buffers["dep.local"]

    def prepare_layout(self, caches):
        self.core.prepare_cache(caches[0])

    def run_wide(self, x, caches, metadata, slot_mapping, q):
        """The attention output [B, 7168] of the normalized rows `x`."""
        from vllm.forward_context import get_forward_context

        b = x.shape[0]
        projected = self.projection.run(x, self.projected[:b])
        local = self.local[:b]
        attention = self.attention
        slots = slot_mapping[attention.mla_attn.mla_attn.layer_name]
        index: Any = get_forward_context().attn_metadata[
            attention.dsa_indexer.k_cache.prefix
        ]
        self.core.run(
            projected,
            local,
            cache=caches[0],
            slots=slots,
            index_slots=index.slot_mapping,
            metadata=metadata,
            q=q,
        )
        return self.output.run(local, x.new_empty((b, x.shape[1])))


class DepKda:
    """One KDA layer's attention at TP1: the input projection, the prepared KDA
    core (96 heads) and o_proj for `runtime.wide` rows: q = 1 decode steps and
    uniform verify steps of `runtime.q` tokens per request."""

    def __init__(self, decoder, runtime):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        from ..cutedsl.kda.prepared import PreparedKda

        attention = decoder.self_attn
        assert attention_kind(attention) == "kda"
        self.layer, self.attention = decoder, attention
        weight = attention.in_proj_qkvgfab.weight
        batches = runtime.wide
        self.wide = batches
        self.projection = DepProjection(weight, DEP_PROJECTION, batches)
        self.output = DepProjection(attention.o_proj.weight, DEP_OPROJ, batches)
        pitch, q = weight.shape[0], runtime.q
        # Gate mode 0 (the F_B GEMM inside the core) at TP1; verify keys are
        # (requests, q), decode keys the batch.
        policies = {b: (pitch, 0) for b in batches}
        if q > 1:
            policies.update({(b // q, q): (pitch, 0) for b in batches if b % q == 0})
        self.core = PreparedKda(attention, policies, is_conv_state_dim_first())
        device, cap = runtime.device, max(batches)
        if "dep.kda.projected" not in runtime.buffers:
            runtime.buffers["dep.kda.projected"] = torch.empty(
                cap, pitch, dtype=torch.bfloat16, device=device
            )
            runtime.buffers["dep.kda.local"] = torch.empty(
                cap, self.core.dim, dtype=torch.bfloat16, device=device
            )
        self.projected = runtime.buffers["dep.kda.projected"]
        self.local = runtime.buffers["dep.kda.local"]

    def prepare_layout(self, caches):
        self.core.prepare_caches(caches)

    def run_wide(self, x, caches, metadata, slot_mapping, q):
        """The attention output [B, 7168] of the normalized rows `x`."""
        b = x.shape[0]
        projected = self.projection.run(x, self.projected[:b])
        local = self.local[:b]
        if q > 1:  # uniform q-token verify rows
            self.core.run(
                projected,
                local,
                caches=caches,
                slots=metadata.spec_state_indices_tensor,
                accepted=metadata.num_accepted_tokens,
                q=q,
                deferred=getattr(metadata, "recoverssm_deferred", None),
            )
        else:
            slots = metadata.non_spec_state_indices_tensor
            self.core.run(projected, local, caches=caches, slots=slots)
        return self.output.run(local, x.new_empty((b, x.shape[1])))


class DepMoe:
    """One MegaMoE layer of a DEP step: the EpMoeLayer schedule over the stock
    module's weights (its batches up to FRONT_MAX; wider: the stock module)."""

    def __init__(self, module, batches, side_stream):
        from ..cutedsl.moe.ep.layer import EpMoeLayer, pack_front

        norm = module.routed_expert_norm
        weights = dict(
            gate=module.gate.weight,
            bias=module.gate.e_score_correction_bias,
            front=pack_front(module),
            shared_down=module.shared_experts.down_proj.weight,
            norm=norm.weight,
            up=module.routed_expert_up_proj.weight,
            eps=norm.variance_epsilon,
            renormalize=module.moe_renormalize,
            scale=module.routed_scaling_factor,
        )
        experts = module.experts

        def routed(latent, topk_weights, topk_ids):
            return experts(latent, topk_weights, topk_ids)

        act = module.shared_experts.act_fn
        self.layer = EpMoeLayer(weights, act, routed, batches, side_stream=side_stream)

    def serves(self, batch):
        return batch in self.layer.batches

    def run(self, x):
        return self.layer(x)


def dep_moe_supported(module) -> bool:
    """Whether a DEP MoE layer runs DepMoe: a MegaMoE latent MoE with a shared
    expert and a latent RMSNorm, routed by one-group sigmoid top-16 (EpMoeLayer.topk;
    else stock)."""
    return (
        getattr(module, "use_mega_moe", False)
        and getattr(module.experts, "top_k", None) == 16
        and module.shared_experts is not None
        and module.routed_expert_norm is not None
        and module.use_grouped_topk
        and module.num_expert_group == 1
        and module.topk_group == 1
        and module.moe_router_activation_func == "sigmoid"
    )
