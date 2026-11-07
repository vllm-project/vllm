# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode of the hybrid decomposition: attention (KDA, DSA) and
its BF16 linears TP-sharded over a TP group of G GPUs (with decode context
parallel G over the same ranks), and the routed experts expert-parallel over all
EP = G x DP GPUs (`-tp G -dp N --enable-expert-parallel` [`-dcp G`] with
MegaMoE; e.g. DCP8 x EP32 = -tp 8 -dcp 8 -dp 4).

Stock serves this deployment with sequence parallelism: the residual stream is
sharded over the TP ranks, every attention all-gathers its rows and reduce-scatters
the o-projection, and the MoE runs on the rank's 1/G row shard with the shared
expert replicated.

The hybrid step keeps every row on every TP rank instead (the TP8 decode's
layout) for attention, and runs the MLPs exactly as stock does:

  AttnRes            stock, on the group's rows (as the TP8 wide step)
  attention          the TP-G specialized input projection and core (KDA, DSA; the
                     wide-step providers of `providers.Attention`), the stock
                     o_proj partial, one all-reduce
  MoE the stock SP MoE module on this rank's row shard (stock `sp_shard`, zero padding
                     marked as padding for MegaMoE), all-gathered back to the group rows
  dense MLP          the stock (replicated) dense MLP on all rows

Precision is the stock layer's at every stage (BF16 GEMMs with FP32 accumulation,
BF16 all-reduces); only the attention's summation order differs.
"""

from typing import Any

from vllm.distributed import tensor_model_parallel_all_reduce as all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.common.ops.sequence_parallel import sp_padding_mask, sp_shard

from ..cutedsl.dsa.prepared import PreparedDsa
from ..cutedsl.gemm.decode import DecodeProjection
from ..cutedsl.kda.prepared import PreparedKda
from .providers import (
    Attention,
    attention_kind,
    dcp_box,
    dsa_scratch,
    pack_dsa_projection,
    projection_policies,
    scratch,
)

# The attention kinds the hybrid step specializes (K3 MLA layers run stock).
HYBRID_KINDS = ("dsa", "kda")


class NoPrefetch:
    """The hybrid step prefetches nothing (`Attention.run_core` forks)."""

    def fork(self, *args):
        pass

    def join(self):
        pass


class HybridAttention(Attention):
    """One KDA / DSA layer's attention in a hybrid step: the TP-G specialized
    input projection and core of `Attention` (its wide-step path), then the
    stock o_proj partial and one TP all-reduce."""

    output: DecodeProjection  # type: ignore[assignment]

    def __init__(self, decoder, runtime):  # noqa: D107 (no super().__init__)
        attention, tp = decoder.self_attn, runtime.tp
        self.kind = kind = attention_kind(attention)
        assert kind in HYBRID_KINDS, kind
        if kind == "dsa":
            weight = pack_dsa_projection(attention)
        else:
            weight = attention.in_proj_qkvgfab.weight
        pitch = weight.shape[0]
        self.wide = batches = tuple(runtime.wide_batches(kind))
        policies = projection_policies(tp, pitch, batches)
        self.projection = DecodeProjection(weight, batches, policies)
        self.core: Any
        if kind == "dsa":
            pitches = {b: pitch for b in batches}
            rows = dsa_scratch(runtime, max(batches))
            policy: dict[str, Any] = dict(queries=runtime.queries)
            policy.update(kernels=runtime.fi_kernels or None)
            policy.update(prims_ts_rows=runtime.prims_ts_rows or None)
            if runtime.dcp > 1:
                from ..cutedsl.dsa.prepared import dsa_on_flashinfer

                heads = attention.num_local_heads
                box = dcp_box(
                    runtime, heads, dsa=True, flashinfer=dsa_on_flashinfer(attention)
                )
                policy.update(dcp=box, parity=runtime.dsa_layers % 2)
            runtime.dsa_layers += 1
            self.core = PreparedDsa(attention, pitches, rows, **policy)
        else:
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            q = runtime.q  # hybrid steps verify (B requests x q tokens)
            verify = {(b // q, q): (pitch, int(tp == 8)) for b in batches}
            self.core = PreparedKda(attention, verify, is_conv_state_dim_first())
        cap = max(batches)
        projected = scratch(runtime, kind + ".projected", (cap, pitch))
        local = scratch(
            runtime, "attention.local", (runtime.attention_rows, 12288 // tp)
        )
        self.prefetch, self.layer = NoPrefetch(), decoder
        self.storage = {b: (projected[:b], local[:b]) for b in batches}
        # o_proj (the SP deployment's: no reduction): warp split-K up to 16 rows,
        # cuBLAS past it (the stock GEMM).
        self.output = DecodeProjection(attention.o_proj.weight, batches, {})

    def run_wide(self, x, caches, metadata, slot_mapping, q):
        """The all-reduced attention output of the normalized group rows `x`."""
        projected, local = self.storage[x.shape[0]]
        projected = self.projection.run(x, projected)
        self.run_core(projected, local, caches, metadata, slot_mapping, q)
        out = x.new_empty((x.shape[0], self.output.weight.shape[0]))
        return all_reduce(self.output.run(local, out))


def run_dense(mlp, x):
    """The dense MLP on all rows: the stock replicated weights give every rank the
    whole output (all-reduced only if stock loaded them TP-sharded)."""
    gate_up, _ = mlp.gate_up_proj(x)
    out, _ = mlp.down_proj(mlp.act_fn(gate_up))
    return all_reduce(out) if mlp.shard_sequence_parallel else out


def run_stock_moe(module, x):
    """A MoE layer the hybrid step does not specialize: the stock SP sequence on
    this rank's shard, all-gathered back to the group rows."""
    from vllm.models.common.ops.sequence_parallel import sp_all_gather

    return sp_all_gather(module(sp_shard(x)))[: x.shape[0]]


class HybridStep:
    """Builds the hybrid providers on a `SpecializedDecode` and runs its steps."""

    def __init__(self, runtime, rounded):
        from .providers import ATTENTION_MAX, wide_caps

        self.runtime = runtime
        runtime.batches = ()
        q = runtime.q
        runtime.aux_layers = tuple(runtime.backbone.aux_hidden_state_layers)
        decoders = runtime.layers
        recoverssm = any(
            getattr(d.self_attn, "recoverssm_records", None) is not None
            for d in decoders
        )
        caps = wide_caps(runtime.dcp > 1, recoverssm)
        # Every verify step whose MoE shard fits MegaMoE's buffer is a hybrid step
        # (its rows are the KV capacity's point: past the cores' ATTENTION_MAX rows
        # the attention layers run the stock module too; the MLPs are always stock).
        tokens = min(
            (
                d.mlp.experts.max_num_tokens
                for d in decoders
                if d.is_moe_layer and getattr(d.mlp, "use_mega_moe", False)
            ),
            default=0,
        )
        runtime.wide = tuple(
            sorted(
                m
                for m in rounded
                if q > 1 and m % q == 0 and -(-m // runtime.tp) <= tokens
            )
        )
        runtime.wide_cap = {k: caps.get(k, 0) * (k in HYBRID_KINDS) for k in caps}
        runtime.attention_rows = max(
            (m for m in runtime.wide if m <= ATTENTION_MAX), default=0
        )
        runtime.attention, runtime.feed_forward, runtime.wide_moe = {}, {}, {}
        if not runtime.wide:
            return
        for layer in decoders:
            kind = attention_kind(layer.self_attn)
            if (
                kind in HYBRID_KINDS
                and runtime.attention_rows
                and runtime.wide_batches(kind)
            ):
                runtime.attention[layer.layer_idx] = HybridAttention(layer, runtime)

    def run(self, step, input_ids, inputs_embeds):
        """A hybrid step: the stock decoder loop over the group's rows with the
        specialized attention and the stock MLPs (module docstring)."""
        from ...ops import attn_res

        runtime = self.runtime
        backbone = runtime.backbone
        x = inputs_embeds
        if x is None:
            x = backbone.embed_input_ids(input_ids)
        b, q, slots, positions = x.shape[0], step.q, step.slot_mapping, step.positions
        # MegaMoE reads the shard's padding rows from the forward context (stock SP;
        # without a runner mask, the shard's SP padding rows alone are padding).
        context = get_forward_context() if is_forward_context_available() else None
        padding = getattr(context, "is_padding", None)
        if context is not None:
            context.is_padding = sp_padding_mask(padding, x)
        aux = [x] if backbone.start_layer in runtime.aux_layers else []
        residual = x.new_empty(b, backbone.num_attn_res_blocks, x.shape[1])
        prefix_sum, hidden = x, None
        try:
            for layer, md in zip(runtime.layers, step.metadata):
                index = layer.layer_idx
                hidden, prefix_sum, residual = layer._pre_attn_norm(
                    hidden, residual, prefix_sum
                )
                attention = runtime.attention.get(index)
                if attention is not None and b in attention.wide:
                    caches = _caches(layer)
                    hidden = attention.run_wide(hidden, caches, md, slots, q)
                else:
                    hidden = all_reduce(layer._run_self_attn(positions, hidden))
                hidden, prefix_sum, residual = layer._post_attn_norm(
                    hidden, residual, prefix_sum
                )
                if layer.is_moe_layer:
                    hidden = run_stock_moe(layer.mlp, hidden)
                else:
                    hidden = run_dense(layer.mlp, hidden)
                if index + 1 in runtime.aux_layers:
                    aux.append(
                        backbone._capture_aux_hidden_stream(
                            index, prefix_sum, hidden, residual
                        )
                    )
        finally:
            if context is not None:
                context.is_padding = padding
        final = backbone.output_attn_res_norm
        hidden = attn_res(
            prefix_sum,
            hidden,
            residual,
            final.weight,
            backbone.output_attn_res_proj.weight.squeeze(0),
            None,
            num_blocks=backbone.num_attn_res_blocks,
            block_write_idx=-1,
            eps=final.variance_epsilon,
            output_norm_eps=0.0,
        )
        return runtime.normalize((hidden, aux) if aux else hidden)


def _caches(layer):
    from .decode import caches_of

    return caches_of(layer.self_attn)
