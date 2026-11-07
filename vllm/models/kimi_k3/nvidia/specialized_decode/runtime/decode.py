# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor over the stock Kimi K3 backbone: one routing
decision per step, then every layer's prepared providers in one loop."""

from functools import partial
from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.distributed import tensor_model_parallel_all_reduce as all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from ...ops import attn_res
from ..cutedsl.attn_res.prepared import PreparedAttnRes
from ..cutedsl.common.embedding import PreparedEmbedding
from ..cutedsl.common.twoshot import TwoShot
from ..cutedsl.kda.prepared import PreparedKda
from ..cutedsl.moe.dense import PreparedDenseMLP
from ..cutedsl.moe.experts_runtime import PersistentExperts
from ..cutedsl.moe.front import SharedFront
from ..cutedsl.moe.prepared import MoEScratch, PreparedMoE
from ..cutedsl.moe.small_batch import chain as small_batch_chain
from ..cutedsl.moe.tail_runtime import MoeTailBox, PreparedTail
from ..cutedsl.moe.wide import WideMoE, WideScratch, wide_batches
from ..cutedsl.oproj.prepared import OprojMailbox
from ..recoverssm.state import install_layers as install_recoverssm_layers
from ..recoverssm.state import install_model_state as install_recoverssm_model_state
from .prefetch import PrefetchController
from .providers import (
    ATTENTION_MAX,
    BATCHES,
    PREPARED_MAX,
    WIDE_ROWS,
    Attention,
    attention_kind,
    pack_dsa_projection,
    pack_front,
    step_cap,
    wide_caps,
)

# The attention layer kinds of the wide-step caps.
WIDE = sorted({*(kind for kind, _ in WIDE_ROWS), "kda"})


class Step(NamedTuple):
    q: int  # tokens per request
    metadata: list  # per layer
    slot_mapping: Any
    # A wide step: more rows than the specialized step serves (``run_wide``).
    wide: bool = False
    positions: Any = None


def attention_layer(module):
    """The registered attention layer of a decoder's attention: a K3.1 DSA layer's
    inner sparse MLA layer, else the module itself."""
    return module.mla_attn.mla_attn if hasattr(module, "dsa_indexer") else module


def layer_name(module):
    module = attention_layer(module)
    return getattr(module, "prefix", getattr(module, "layer_name", None))


def decode_metadata(attention, context, q, b):
    """One layer's metadata if this step is a pure decode (q = 1) or a uniform
    verify step of b tokens, else None."""
    md = context.attn_metadata
    md = md.get(layer_name(attention)) if isinstance(md, dict) else None
    if md is None:
        return None
    if q > 1:
        return md if is_uniform_verify(md, q, b) else None
    if md.num_prefills or getattr(md, "num_spec_decodes", 0):
        return None
    return md


def is_uniform_verify(md, q, b):
    """Whether every request of the b // q graph rows verifies q tokens (FULL graphs
    pad whole requests: KDA stages NULL state rows, MLA zero-length decodes)."""
    if md.num_prefills:
        return False
    if not hasattr(md, "num_spec_decodes"):
        return md.num_decode_tokens == md.num_decodes * q == b
    rows, tokens = md.spec_state_indices_tensor, md.num_spec_decode_tokens
    full = rows is not None and rows.shape[0] * q == b
    return full and not md.num_decodes and tokens == md.num_spec_decodes * q


def caches_of(attention):
    cache = getattr(attention_layer(attention), "kv_cache", ())
    return [cache] if isinstance(cache, torch.Tensor) else list(cache)


class SpecializedDecode:
    """Built once the weights are loaded; owns every prepared provider, the
    mailboxes and the decode buffers. Reads the backbone, never modifies it (but
    for the deferred RecoverSSM's per-instance KDA verify, `recoverssm.state`)."""

    attention: dict[int, Any]
    feed_forward: dict[int, Any]
    wide_moe: dict[int, Any]

    def __init__(self, backbone, config):
        # The deferred RecoverSSM's KDA layers, before any provider reads them.
        install_recoverssm_layers(
            backbone.layers[backbone.start_layer : backbone.end_layer]
        )
        self.backbone, self.buffers = backbone, {}
        self.kv_prepared = False  # prepare_kv for the bound KV cache
        self.cache_config = get_current_vllm_config().cache_config
        q = config.q
        # Verify steps decode q tokens per request at M = requests * q tokens; FULL
        # verify graphs round each capture size up to whole requests.
        rounded = {round_up(s, q) for s in config.capture_sizes}
        verify = {m for m in rounded if m <= PREPARED_MAX}
        # Steps up to the model's specialized-vs-stock crossover (wider: stock).
        cap = step_cap(config.dsa, config.dcp > 1)
        cap = min(cap, PREPARED_MAX)
        batches = {b for b in (*BATCHES, *verify) if b <= cap}
        self.batches, self.q = tuple(sorted(batches)), q
        # Wide verify steps (cap < rows <= ATTENTION_MAX): the stock layer loop with
        # each attention layer kind's projection and core specialized up to its own
        # crossover (providers.WIDE_ROWS); kinds and rows past it run stock.
        decoders = backbone.layers[backbone.start_layer : backbone.end_layer]
        recoverssm = any(
            getattr(d.self_attn, "recoverssm_records", None) is not None
            for d in decoders
        )
        self.wide_cap = wide_caps(config.dcp > 1, recoverssm)
        kinds = ("kda", "mla", "dsa") if config.dsa else ("kda", "mla")
        widest = max(self.wide_cap[kind] for kind in kinds)
        wide = {m for m in rounded if q > 1 and cap < m <= min(widest, ATTENTION_MAX)}
        self.wide = tuple(sorted(wide))
        self.attention_rows = max((*self.batches, *self.wide))
        self.queries = (1, q) if q > 1 else (1,)
        # The longest sequence the deployment serves bounds the MLA split count.
        self.max_context, self.fi_kernels = config.max_context, config.fi_kernels
        self.prims_ts_rows = config.prims_ts_rows
        # Decode context parallel over the TP ranks: MLA KV is sequence-sharded.
        self.dcp, self.dcp_box, self.mla_layers = config.dcp, None, 0
        self.dsa_dcp_box, self.dsa_layers = None, 0
        group = get_tp_group()
        self.tp, self.rank = group.world_size, group.rank_in_group
        self.group = group.device_group
        self.device = torch.device(current_platform.current_device())

        layers = backbone.layers[backbone.start_layer : backbone.end_layer]
        self.layers = layers = tuple(layers)
        # DEP (TP1 attention per DP rank, EP MoE): the stock layer loop with the
        # TP1-specialized attention layers (runtime.dep), every step a wide one.
        self.dep = self.tp == 1
        self.hybrid = None
        if self.dep:
            self.build_dep(rounded)
            return
        # Hybrid (TP-G attention groups + EP MoE over every rank): runtime.hybrid.
        if get_current_vllm_config().parallel_config.enable_expert_parallel:
            from .hybrid import HybridStep

            self.hybrid = HybridStep(self, rounded)
            return
        self.build_providers()
        self.embedding = PreparedEmbedding(backbone.embed_tokens, self.tp, self.rank)
        self.before = {
            d.layer_idx: PreparedAttnRes.for_target(d, has_delta=d is not layers[0])
            for d in layers
        }
        # The final AttnRes also applies the backbone's final norm.
        self.final = PreparedAttnRes.for_target(backbone, final=True)
        # The next (or final) AttnRes consumes each MoE output from its MoE tail output
        # row.
        self.fused_before = {}
        followers = [self.before[f.layer_idx] for f in layers[1:]] + [self.final]
        for layer, consumer in zip(layers, followers):
            moe = self.feed_forward[layer.layer_idx]
            if isinstance(moe, PreparedMoE):
                consumer.bind_ring(moe.tail.tail_box.out.buffer)
                self.fused_before[layer.layer_idx] = consumer
        # The AttnRes entering each aux layer (or the final one) also stores the
        # updated prefix: the stock aux stream (prefix sum plus the last MLP output).
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        end = layers[-1].layer_idx + 1
        for index in self.aux_layers:
            (self.final if index == end else self.before[index]).bind_aux()
        empty = partial(torch.empty, device=self.device, dtype=torch.bfloat16)
        self.hidden = (empty((64, 7168)), empty((64, 7168)))
        self.prefix = empty((64, 7168))
        self.blocks = empty((64, backbone.num_attn_res_blocks, 7168))

    def build_dep(self, rounded):
        """DEP: no full specialized step (its MoE, o-projection and AttnRes providers
        are TP8 / TP16 ones); every decode or uniform verify step of up to the KV cache
        dtype's DEP_ROWS rows runs `run_wide` with the DEP_KINDS attention layers
        specialized at TP1 (runtime.dep.DepAttention), the rest stock."""
        from .dep import (
            DEP_KINDS,
            DEP_ROWS,
            DepAttention,
            DepKda,
            DepMoe,
            dep_moe_supported,
        )

        self.batches = ()
        self.aux_layers = tuple(self.backbone.aux_hidden_state_layers)
        cap = DEP_ROWS.get(self.cache_config.cache_dtype, 0)
        self.wide = tuple(sorted(m for m in rounded if m <= cap))
        self.wide_cap = {kind: cap * (kind in DEP_KINDS) for kind in WIDE}
        self.attention_rows = max(self.wide, default=0)
        self.attention = {}
        self.feed_forward = {}
        self.wide_moe = {}
        if not self.wide:
            return
        providers = {"dsa": DepAttention, "kda": DepKda}
        side = None  # the DEP MoE layers' side stream (one per rank)
        for layer in self.layers:
            kind = attention_kind(layer.self_attn)
            if kind in DEP_KINDS:
                self.attention[layer.layer_idx] = providers[kind](layer, self)
            if layer.is_moe_layer and dep_moe_supported(layer.mlp):
                side = side or torch.cuda.Stream()
                moe = DepMoe(layer.mlp, self.wide, side)
                self.wide_moe[layer.layer_idx] = moe

    def wide_batches(self, kind):
        """The wide step rows whose `kind` attention layers run the specialized
        projection and core."""
        return tuple(b for b in self.wide if b <= self.wide_cap[kind])

    def build_providers(self):
        """Every layer's attention and feed-forward providers, the model's mailboxes
        and the prefetch side stream. All collective allocations, weight packing and
        kernel compiles finish here, before KV binding and CUDA graph capture."""
        config, layers = self.backbone.config, self.layers
        beta = config.activation_situ_beta
        linear_beta = config.activation_situ_linear_beta
        tp, rank, device, batches = self.tp, self.rank, self.device, self.batches
        owner = self.group, device, tp, rank
        packed = {
            d.layer_idx: pack_front(d.mlp, rank, tp) for d in layers if d.is_moe_layer
        }
        for d in layers:  # before the prefetch tables take the weight addresses
            if attention_kind(d.self_attn) == "dsa":
                pack_dsa_projection(d.self_attn)
        self.packed = packed
        prefetch = self.prefetch = PrefetchController(self, layers, packed)
        state = self.moe_state = MoEScratch(tp, device, batches)
        mailbox, tail_box, moe_layers = (
            OprojMailbox(*owner),
            MoeTailBox(*owner),
            list(packed),
        )
        twoshot = TwoShot(*owner)
        front_box = OprojMailbox(*owner, (1, 64, 3584 + 2 * 896))
        small_batch = small_batch_chain.SmallBatchBox(*owner)
        self.attention = {}
        self.feed_forward = {}
        # Wide steps' MoE layers (moe.wide), sharing one scratch.
        self.wide_moe = {}
        wide = wide_batches(self.wide)
        wide_state = WideScratch(tp, device, wide) if wide else None
        for layer in layers:
            index, module = layer.layer_idx, layer.mlp
            boxes = (mailbox, small_batch, twoshot)
            self.attention[index] = Attention(layer, self, prefetch, *boxes)
            if not layer.is_moe_layer:
                reduce = all_reduce if module.down_proj.reduce_results else None
                self.feed_forward[index] = PreparedDenseMLP(module, batches, reduce)
                continue
            args = (module, packed[index], front_box, state.front, batches, beta)
            shared = dict(plan=state.plan, counters=state.shared, q=self.q)
            front = SharedFront(*args, linear_beta, **shared)
            routed = module.experts.routed_experts
            w13, w2 = (
                w.view(torch.uint8) for w in (routed.w13_weight, routed.w2_weight)
            )
            weights = w13, routed.w13_weight_scale, w2, routed.w2_weight_scale
            experts = PersistentExperts(*weights, beta=beta, linear_beta=linear_beta)
            norm = module.routed_expert_norm.weight
            up = module.routed_expert_up_proj.weight
            op = module.experts._k3_latent_moe_tail_op
            placement = tail_box, moe_layers.index(index), len(moe_layers)
            tail = PreparedTail(op, norm, up, batches, placement)
            moe = PreparedMoE(
                front, experts, tail, state, prefetch=prefetch, layer_idx=index
            )
            moe.small_batch = small_batch_chain.SmallBatchChain(
                small_batch, module, experts, tail, index, beta, linear_beta, prefetch
            )
            self.feed_forward[index] = moe
            if wide_state is not None:
                self.wide_moe[index] = WideMoE(
                    module,
                    experts,
                    wide_state,
                    beta,
                    linear_beta,
                    packed=packed[index],
                )

    def bind_model_state(self, state):
        """Install the runner-state side of the deferred RecoverSSM on the model
        runner state (`recoverssm.state.install_model_state`)."""
        install_recoverssm_model_state(state)

    @torch.no_grad()
    def process_weights_after_loading(self):
        """After a weight reload: re-derive, in place, the copies built from the
        weights. Everything else reads the backbone's tensors directly, which a
        reload refills at the same addresses."""
        for layer in self.layers:
            if layer.is_moe_layer:
                fresh = pack_front(layer.mlp, self.rank, self.tp)
                self.packed[layer.layer_idx].copy_(fresh)
            else:
                self.feed_forward[layer.layer_idx].process_weights_after_loading(
                    layer.mlp
                )
            core = self.attention[layer.layer_idx].core
            # KDA keeps derived copies; MLA reads the backbone's tensors
            if isinstance(core, PreparedKda):
                core.process_weights_after_loading(layer.self_attn)

    def prepare_kv(self):
        """Compile each layer's cores for its bound KV cache layout. The memory
        profiling's throwaway cache is left unprepared (its compiles would all be
        thrown away: every KV-dependent kernel is specialized to its cache's block
        count), so its steps run the stock forward; the worker records the real
        cache's block count before binding it, the profiling one only after."""
        self.kv_prepared = False
        if self.cache_config.num_gpu_blocks is None:  # the profiling cache
            return
        for layer in self.layers:
            attention = self.attention.get(layer.layer_idx)
            if attention is not None:  # DEP: the specialized kinds only
                attention.prepare_layout(caches_of(layer.self_attn))
        self.kv_prepared = True

    def embed_input_ids(self, ids):
        rows = ids.shape[0] if ids.ndim == 1 and ids.is_contiguous() else 0
        if self.dep or self.hybrid or not 1 <= rows <= PREPARED_MAX:
            return self.backbone.embed_input_ids(ids)
        table = self.backbone.embed_tokens.weight
        return self.embedding.run(ids, table.new_empty((ids.shape[0], table.shape[1])))

    def plan(self, positions, intermediate_tensors):
        """This step's decode plan, or None to run the stock forward instead.

        Steps that run the stock forward:
        - pipeline-parallel stage inputs (`intermediate_tensors`);
        - token counts without prepared providers (not in `batches`, and not a
          wide verify step: q > 1 and in `wide`, the rows past the step cap
          that some attention layer kind's projection and core take, run_wide);
        - steps before `prepare_kv` bound the KV cache (the memory profiling's
          cache is never prepared);
        - steps outside a CUDA graph capture (profiling, warmup and the warmup run
          before each capture), so every stock kernel runs before its capture;
        - PIECEWISE steps, keyed by token count alone and so captured as stock;
        - steps with prompt rows (prefills, chunked prefills, fresh prompts);
        - speculative steps whose requests do not all verify q tokens (partial
          drafts, verify rows mixed with plain decode rows).

        Every predicate is the same on all TP ranks: the mailboxes of the specialized
        decode need every rank to take the same path.
        """
        b = positions.shape[0]
        wide = b in self.wide
        if intermediate_tensors is not None or not (b in self.batches or wide):
            return None
        if not self.kv_prepared:  # KV cache not prepared (memory profiling)
            return None
        if not torch.cuda.is_current_stream_capturing():
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        by_layer = context.attn_metadata
        rows: list[Any] = list(by_layer.values()) if isinstance(by_layer, dict) else []
        q = self.q if any(getattr(md, "num_spec_decodes", 0) for md in rows) else 1
        # Wide steps verify (TP8 / TP16); DEP steps also decode one token each.
        if b % q or (wide and q == 1 and not self.dep):
            return None
        metadata = [decode_metadata(d.self_attn, context, q, b) for d in self.layers]
        if any(md is None for md in metadata):
            return None
        return Step(q, metadata, context.slot_mapping, wide, positions)

    def run(self, step, input_ids, inputs_embeds):
        """Normalized hidden states (and the aux streams) of a planned step."""
        if self.hybrid is not None:
            return self.hybrid.run(step, input_ids, inputs_embeds)
        if step.wide:
            return self.run_wide(step, input_ids, inputs_embeds)
        x = inputs_embeds
        if x is None:
            x = self.embed_input_ids(input_ids)
        out = torch.empty_like(x)
        # The stock aux streams: the AttnRes prefix entering each aux layer.
        aux_out = [torch.empty_like(x) for _ in self.aux_layers]
        aux = dict(zip(self.aux_layers, aux_out))
        b, q, slots = x.shape[0], step.q, step.slot_mapping
        prefix, blocks = self.prefix[:b], self.blocks[:b]
        first, second = self.hidden[0][:b], self.hidden[1][:b]
        hidden, pre_ready = x, False
        for layer, md in zip(self.layers, step.metadata):
            index, caches = layer.layer_idx, caches_of(layer.self_attn)
            if not pre_ready:
                self.before[index].run(hidden, prefix, blocks, first, aux.get(index))
            self.attention[index].run(
                first, second, prefix, blocks, caches, md, slots, q
            )
            consumer = self.fused_before.get(index)
            pre_ready = consumer is not None
            if consumer is not None:
                target = out if consumer is self.final else first
                stream = aux.get(index + 1)
                consume = partial(consumer.run_ring, prefix, blocks, aux=stream)
                hidden = self.feed_forward[index].run(second, target, consume=consume)
            else:
                hidden = self.feed_forward[index].run(second, first)
                first, second = second, first
        if not pre_ready:
            self.final.run(hidden, prefix, blocks, out, aux.get(index + 1))
        self.prefetch.join()
        self.moe_state.join()
        return (out, aux_out) if aux_out else out

    def run_wide(self, step, input_ids, inputs_embeds):
        """A wide step: the stock decoder layer loop (KimiLinearModel.forward without
        pipeline or sequence parallelism: AttnRes, o_proj + all-reduce, MoE / dense
        MLP), with the attention layers whose kind serves this many rows running the
        specialized input projection and core instead of the stock module's."""
        backbone = self.backbone
        x = inputs_embeds
        if x is None:
            x = backbone.embed_input_ids(input_ids)
        b, q, slots, positions = x.shape[0], step.q, step.slot_mapping, step.positions
        aux = [x] if backbone.start_layer in self.aux_layers else []
        residual = x.new_empty(b, backbone.num_attn_res_blocks, x.shape[1])
        prefix_sum, hidden = x, None
        for layer, md in zip(self.layers, step.metadata):
            hidden, prefix_sum, residual = layer._pre_attn_norm(
                hidden, residual, prefix_sum
            )
            attention = self.attention.get(layer.layer_idx)
            if attention is not None and b in attention.wide:
                caches = caches_of(layer.self_attn)
                hidden = attention.run_wide(hidden, caches, md, slots, q)
            else:
                hidden = layer._run_self_attn(positions, hidden)
            hidden, prefix_sum, residual = layer._post_attn_norm(
                hidden, residual, prefix_sum
            )
            # moe.wide for the rows it prepared (65-WIDE_ROWS); dense MLP and any
            # other wide rows (a step cap below 64): stock.
            moe = self.wide_moe.get(layer.layer_idx)
            if moe is not None and moe.serves(b):
                hidden = moe.run(hidden)
            else:
                hidden = layer.mlp(hidden)
            if layer.layer_idx + 1 in self.aux_layers:
                aux.append(
                    backbone._capture_aux_hidden_stream(
                        layer.layer_idx, prefix_sum, hidden, residual
                    )
                )
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
        return self.normalize((hidden, aux) if aux else hidden)

    def normalize(self, output):
        """A stock step's output with the backbone's final norm applied, as the
        specialized decode's last AttnRes applies it; the aux streams stay pre-norm.

        Runs in chunks of at most 255 rows: the stock RMSNorm kernel reduces with
        1024-thread blocks below 256 rows and 256-thread blocks above, so chunking keeps
        the reduction order `compute_logits` uses for the few sampled rows, and
        prefill logits match stock.
        """
        hidden, aux = output if isinstance(output, tuple) else (output, None)
        norm = self.backbone.norm
        if hidden.ndim == 2 and hidden.shape[0] >= 256:
            chunks = range(0, hidden.shape[0], 255)
            hidden = torch.cat([norm(hidden[i : i + 255], None) for i in chunks])
        else:
            hidden = norm(hidden, None)
        return hidden if aux is None else (hidden, aux)
