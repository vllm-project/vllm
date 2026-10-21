# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor over the stock Kimi K3 backbone: one routing
decision per step, then every layer's prepared providers in one loop."""

from functools import partial
from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.distributed import tensor_model_parallel_all_reduce as all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from ..cutedsl.attn_res.prepared import PreparedAttnRes
from ..cutedsl.common.embedding import PreparedEmbedding
from ..cutedsl.common.twoshot import TwoShot
from ..cutedsl.moe.dense import PreparedDenseMLP
from ..cutedsl.moe.experts_runtime import PersistentExperts
from ..cutedsl.moe.front import SharedFront
from ..cutedsl.moe.prepared import MoEScratch, PreparedMoE
from ..cutedsl.moe.small_batch import chain as small_batch_chain
from ..cutedsl.moe.tail_runtime import MoeTailBox, PreparedTail
from ..cutedsl.oproj.prepared import OprojMailbox
from .prefetch import PrefetchController
from .providers import BATCHES, PREPARED_MAX, Attention, pack_front


class Step(NamedTuple):
    q: int  # tokens per request
    metadata: list  # per layer
    slot_mapping: Any


def layer_name(module):
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
    cache = getattr(attention, "kv_cache", ())
    return [cache] if isinstance(cache, torch.Tensor) else list(cache)


class SpecializedDecode:
    """Built once the weights are loaded; owns every prepared provider, the
    mailboxes and the decode buffers. Reads the backbone, never modifies it."""

    def __init__(self, backbone, config):
        self.backbone, self.buffers = backbone, {}
        q = config.q
        # Verify steps decode q tokens per request at M = requests * q tokens; FULL
        # verify graphs round each capture size up to whole requests.
        rounded = {round_up(s, q) for s in config.capture_sizes}
        verify = {m for m in rounded if m <= PREPARED_MAX}
        self.batches, self.q = tuple(sorted({*BATCHES, *verify})), q
        self.queries = (1, q) if q > 1 else (1,)
        # The longest sequence the deployment serves bounds the MLA split count.
        self.max_context, self.mla_core = config.max_context, config.mla_core
        # Decode context parallel over the TP ranks: MLA KV is sequence-sharded.
        self.dcp, self.dcp_box, self.mla_layers = config.dcp, None, 0
        group = get_tp_group()
        self.tp, self.rank = group.world_size, group.rank_in_group
        self.group = group.device_group
        self.device = torch.device(current_platform.current_device())

        layers = backbone.layers[backbone.start_layer : backbone.end_layer]
        self.layers = layers = tuple(layers)
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
        self.packed = packed
        prefetch = self.prefetch = PrefetchController(self, layers, packed)
        state = MoEScratch(tp, device, batches)
        mailbox, tail_box, moe_layers = (
            OprojMailbox(*owner),
            MoeTailBox(*owner),
            list(packed),
        )
        twoshot = TwoShot(*owner)
        front_box = OprojMailbox(*owner, (1, 64, 3584 + 2 * 896))
        small_batch = small_batch_chain.SmallBatchBox(*owner)
        self.attention: dict[int, Attention] = {}
        self.feed_forward: dict[int, Any] = {}
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

    @torch.no_grad()
    def refresh_weights(self):
        """After a weight reload: re-derive, in place, the copies built from the
        weights. Everything else reads the backbone's tensors directly, which a
        reload refills at the same addresses."""
        for layer in self.layers:
            if layer.is_moe_layer:
                fresh = pack_front(layer.mlp, self.rank, self.tp)
                self.packed[layer.layer_idx].copy_(fresh)
            else:
                self.feed_forward[layer.layer_idx].refresh_weights(layer.mlp)
            core = getattr(self.attention[layer.layer_idx], "core", None)
            refresh = getattr(core, "refresh_weights", None)
            if refresh is not None:
                refresh(layer.self_attn)

    def prepare_kv(self):
        """Compile each layer's cores for its bound KV cache layout (the profiling
        cache, then the real one)."""
        for layer in self.layers:
            caches = caches_of(layer.self_attn)
            self.attention[layer.layer_idx].prepare_layout(caches)

    def embed_input_ids(self, ids):
        rows = ids.shape[0] if ids.ndim == 1 and ids.is_contiguous() else 0
        if not 1 <= rows <= PREPARED_MAX:
            return self.backbone.embed_input_ids(ids)
        table = self.backbone.embed_tokens.weight
        return self.embedding.run(ids, table.new_empty((ids.shape[0], table.shape[1])))

    def plan(self, positions, intermediate_tensors):
        """This step's decode plan, or None to run the stock forward instead.

        Steps that run the stock forward:
        - pipeline-parallel stage inputs (``intermediate_tensors``);
        - token counts without prepared providers (above 64 or not in ``batches``);
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
        if intermediate_tensors is not None or b not in self.batches:
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
        if b % q:
            return None
        metadata = [decode_metadata(d.self_attn, context, q, b) for d in self.layers]
        if any(md is None for md in metadata):
            return None
        return Step(q, metadata, context.slot_mapping)

    def run(self, step, input_ids, inputs_embeds):
        """Normalized hidden states (and the aux streams) of a planned step."""
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
        return (out, aux_out) if aux_out else out

    def normalize(self, output):
        """A stock step's output with the backbone's final norm applied, as the
        specialized decode's last AttnRes applies it; the aux streams stay pre-norm.

        Runs in chunks of at most 255 rows: the stock RMSNorm kernel reduces with
        1024-thread blocks below 256 rows and 256-thread blocks above, so chunking keeps
        the reduction order ``compute_logits`` uses for the few sampled rows, and
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
