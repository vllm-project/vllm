# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident dense decoder for the TP2 MiniCPM5 DSpark checkpoint."""

from weakref import ReferenceType, ref

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.config import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.triton_utils import triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

from ..draft_attention import _expand_draft_pages
from .peer import region

logger = init_logger(__name__)


def _pack_projection_weight(weight):
    n, k = weight.shape
    return weight.view(n // 16, 16, k // 16, 8, 2).permute(0, 2, 3, 1, 4).contiguous()


class DraftMono:
    queries = 7

    def __init__(self, draft, target, max_context, max_requests, tp=2):
        self.target = target
        self.tp = tp
        self.max_context = max_context
        self.max_requests = max_requests
        self.layers = tuple(draft.layers)
        self.attentions = tuple(layer.self_attn.attn for layer in self.layers)
        self.weights = tuple(self._weights(layer, tp) for layer in self.layers)
        self.controls: tuple[torch.Tensor, ...] = ()
        self.markov_shard = None
        self.markov_peer = None
        self.caches: tuple[ReferenceType[torch.Tensor], ...] = ()

    def attach(self, draft):
        """Install the optional XPU forward on this draft instance."""
        fallback = draft.forward

        def forward(input_ids, positions, input_embeds=None):
            if input_embeds is None:
                input_embeds = draft.embed_input_ids(input_ids)
            result = self.run(positions, input_embeds)
            if result is None:
                return fallback(input_ids, positions, input_embeds)
            hidden, residual = result
            hidden, _ = draft.norm(hidden, residual)
            return hidden

        draft.specialized_decode = self
        draft.forward = forward

    @staticmethod
    def _weights(layer, tp=2):
        if tp not in (1, 2):
            return ()
        query_heads, kv_heads = 32 // tp, 2 // tp
        query_width = query_heads * 128
        intermediate = 8192 // tp
        attn = layer.self_attn
        if (
            type(attn.attn.impl) is not FlashAttentionImpl
            or attn.attn.attn_type != AttentionType.DECODER
            or attn.causal
            or attn.sliding_window is not None
            or attn.attention_sink_bias is not None
            or attn.v_scale is not None
            or (attn.num_heads, attn.num_kv_heads, attn.head_dim)
            != (query_heads, kv_heads, 128)
            or not isinstance(layer.mlp.act_fn, SiluAndMul)
            or not attn.rotary_emb.is_neox_style
            or attn.rotary_emb.rotary_dim != 128
        ):
            return ()
        projections = (
            attn.qkv_proj,
            attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        )
        if any(module.bias is not None for module in projections):
            return ()
        norms = (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            attn.q_norm,
            attn.k_norm,
        )
        if any(norm.variance_epsilon != 1e-6 for norm in norms):
            return ()
        weights = tuple(module.weight for module in (*norms, *projections))
        shapes = (
            (2048,),
            (2048,),
            (128,),
            (128,),
            (query_width + 2 * kv_heads * 128, 2048),
            (2048, query_width),
            (2 * intermediate, 2048),
            (2048, intermediate),
        )
        if not all(
            tensor.device == weights[0].device
            and tensor.dtype == torch.float16
            and tensor.is_contiguous()
            and tensor.shape == shape
            for tensor, shape in zip(weights, shapes, strict=True)
        ):
            return ()
        return (*weights[:4], *(_pack_projection_weight(w) for w in weights[4:]))

    def _all_ranks_support(self, supported):
        vote = torch.tensor([supported], dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=self.target.tp.cpu_group)
        return bool(vote.item())

    def bind_caches(self):
        self.caches = ()
        supported = getattr(self.target.ops, "draft_tensor_parallel_sizes", None)
        supported_tps = supported() if supported is not None else (2,)
        if not self._all_ranks_support(
            self.tp in supported_tps
            and 0 < len(self.layers) <= self.target.draft_slots
            and all(self.weights)
        ):
            logger.info_once("MiniCPM5 draft mono requires compatible FP16 weights")
            return

        caches = tuple(attn.kv_cache for attn in self.attentions)
        first = caches[0]
        ropes = tuple(
            layer.self_attn.rotary_emb._match_cos_sin_cache_dtype(weights[0])
            for layer, weights in zip(self.layers, self.weights, strict=True)
        )
        compatible = all(
            cache.device == first.device == weights[0].device
            and cache.dtype == torch.float16
            and cache.ndim == 4
            and (cache.is_contiguous() or cache.permute(0, 2, 1, 3).is_contiguous())
            and cache.shape[1] == 2 // self.tp
            and cache.shape[2] == first.shape[2]
            and cache.shape[2] > 0
            and cache.shape[2] % 64 == 0
            and cache.shape[3] == 256
            and rope.ndim == 2
            and rope.shape[1] == 128
            and rope.shape[0] >= self.max_context
            and rope.is_contiguous()
            for cache, rope, weights in zip(caches, ropes, self.weights, strict=True)
        )
        if not self._all_ranks_support(compatible):
            logger.info_once("MiniCPM5 draft mono requires packed FP16 KV and RoPE")
            return

        rows = self.max_requests * self.queries
        device = first.device
        self.block_size = first.shape[2]
        self.blocks = torch.empty(
            (rows, cdiv(self.max_context, self.block_size)),
            device=device,
            dtype=torch.int32,
        )
        self.lengths = torch.empty(rows, device=device, dtype=torch.int32)
        self.epochs = torch.zeros(1, device=device, dtype=torch.int32)
        if not self.controls:
            self.controls = tuple(
                torch.zeros(
                    self.target.ops.control_words(), device=device, dtype=torch.int32
                )
                for _ in self.layers
            )
            if self.target.failstop is None:
                for control in self.controls:
                    self.target.add_health_control(control)
        self.ropes = ropes
        # The draft attention layers keep the cache tensors alive until
        # the shared runner drops them after graph memory profiling.
        self.caches = tuple(ref(cache) for cache in caches)
        logger.info_once(
            "MiniCPM5 draft mono bound %d dense non-causal layers", len(self.layers)
        )

    def run(self, positions, hidden):
        if not self.caches or not is_forward_context_available():
            return None
        caches = tuple(cache() for cache in self.caches)
        if any(cache is None for cache in caches):
            return None
        context = get_forward_context()
        metadata = context.attn_metadata
        rows = positions.numel()
        if (
            context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            or not isinstance(metadata, dict)
            or rows == 0
            or rows % self.queries
            or rows > self.lengths.numel()
        ):
            return None
        items = [metadata.get(attn.layer_name) for attn in self.attentions]
        first = items[0]
        if (
            not isinstance(first, FlashAttentionMetadata)
            or any(item is not first for item in items)
            or first.causal is not False
            or first.dcp_context_kv_lens is not None
            or first.mm_prefix_query_range_tensor is not None
            or first.rswa_prefix_lens is not None
            or first.max_query_len != self.queries
            or first.max_seq_len > self.max_context
        ):
            return None
        starts, lengths, blocks, slots = (
            first.query_start_loc,
            first.seq_lens,
            first.block_table,
            first.slot_mapping,
        )
        num_reqs = starts.numel() - 1
        columns = self.blocks.shape[1]
        if (
            num_reqs != rows // self.queries
            or lengths.numel() != num_reqs
            or not lengths.is_contiguous()
            or not starts.is_contiguous()
            or blocks.shape[0] < num_reqs
            or not 0 < blocks.shape[1] <= columns
            or blocks.stride(1) != 1
            or slots.numel() < rows
        ):
            return None

        self.target.use_current_stream()
        _expand_draft_pages[(rows,)](
            starts,
            lengths,
            blocks,
            self.lengths,
            self.blocks,
            self.epochs,
            num_reqs,
            blocks.stride(0),
            blocks.shape[1],
            columns,
            self.block_size,
            triton.next_power_of_2(num_reqs),
            triton.next_power_of_2(columns),
        )
        residual = None
        for index, (weights, cache, control, rope) in enumerate(
            zip(self.weights, caches, self.controls, self.ropes, strict=True)
        ):
            out, res = torch.empty_like(hidden), torch.empty_like(hidden)
            for start in range(0, rows, self.queries):
                span = slice(start, start + self.queries)
                self.target.ops.draft_decode(
                    hidden[span],
                    None if residual is None else residual[span],
                    positions[span],
                    self.blocks[span],
                    self.lengths[span],
                    slots[span],
                    out[span],
                    res[span],
                    self.target.scratch,
                    control,
                    weights,
                    cache,
                    self.target.region,
                    len(self.target.layers) + index,
                    rope,
                )
            hidden, residual = out, res
        return hidden, residual

    def configure_sampling(self, model):
        head = model.lm_head
        markov = model.model.markov_head.markov_w2
        processor = model.logits_processor
        size, rank = head.num_embeddings_per_partition, self.target.tp.rank_in_group
        start = head.shard_indices.org_vocab_start_index
        supported = (
            model.draft_id_to_target_id is None
            and head.tp_size == 2
            and start == rank * size
            and head.num_embeddings_padded == head.org_vocab_size == size * 2
            and processor.org_vocab_size == head.org_vocab_size
            and processor.scale == 1
            and processor.soft_cap is None
            and not processor.logits_as_input
            and processor.head_dtype in (None, torch.float16)
            and markov.tp_size == 1
            and isinstance(markov.quant_method, UnquantizedEmbeddingMethod)
            and markov.weight.dtype == torch.float16
            and markov.weight.shape[0] == 2 * size
        )
        if self._all_ranks_support(supported):
            self.markov_shard = markov.weight[start : start + size]
            self.vocab_start = start
            if self._all_ranks_support(
                size == 65280 and hasattr(self.target.ops, "candidate_greedy_bias")
            ):
                self._configure_peer_sampling()

    def _configure_peer_sampling(self):
        assert self.markov_shard is not None
        device = self.markov_shard.device
        ops = self.target.ops
        self.markov_peer = region(1, self.target.tp, ops)
        self.markov_stream = torch.xpu.current_stream()
        self.markov_partial = torch.empty((8, 32, 64), dtype=torch.int64, device=device)
        self.markov_control = torch.zeros(
            ops.control_words(), dtype=torch.int32, device=device
        )
        if self.target.failstop is None:
            self.target.add_health_control(self.markov_control)
        else:
            self.markov_peer.set_health(self.target.region.health())

    def _sample_peer_greedy(self, model, base, prev, output):
        assert self.markov_peer is not None
        current = torch.xpu.current_stream()
        if current != self.markov_stream:
            current.wait_stream(self.markov_stream)
            self.markov_peer.rebind()
            self.markov_stream = current

        device = base.device
        values = torch.empty((1, 128), dtype=torch.float32, device=device)
        indices = torch.empty_like(values, dtype=torch.int32)
        sampled = torch.empty((1, 1), dtype=torch.int64, device=device)
        counts = torch.empty((1,), dtype=torch.int32, device=device)
        rejected = torch.empty_like(counts)
        ops = self.target.ops
        rank = self.target.tp.rank_in_group
        for step in range(output.shape[1]):
            bias = F.linear(model.markov_embed(prev), self.markov_shard)
            ops.candidate_greedy_bias(
                values,
                indices,
                base[:, step],
                rank,
                self.markov_partial,
                self.markov_control,
                self.markov_peer,
                sampled,
                counts,
                rejected,
                bias,
            )
            prev = sampled[:, 0]
            output[:, step].copy_(prev)

    def sample_sequential(self, speculator, num_reqs, head_hidden):
        if (
            speculator.draft_logits is not None
            or speculator.enable_adaptive_verification
            or speculator.use_confidence_head
            or speculator.draft_watermarker is not None
        ):
            return False
        count = num_reqs * speculator.num_speculative_steps
        hidden = head_hidden[speculator.sample_indices[:count]]
        prev = speculator.input_buffers.input_ids[speculator._anchor_idx[:num_reqs]]
        self.sample_greedy(
            speculator.model, hidden, prev, speculator.draft_tokens[:num_reqs]
        )
        return True

    def sample_greedy(self, model, hidden, prev, output):
        # Every token participates. Only the winning (value, token ID) of
        # each rank crosses the link, with rank order breaking global ties.
        rows, steps = output.shape
        base = model.logits_processor(model.lm_head, hidden, skip_gather=True)
        base = base.view(rows, steps, -1)
        if rows == 1 and base.dtype == torch.float16 and self.markov_peer is not None:
            self._sample_peer_greedy(model, base, prev, output)
            return
        for step in range(steps):
            bias = F.linear(model.markov_embed(prev), self.markov_shard)
            scores = base[:, step] + bias
            scores.masked_fill_(scores.isnan(), float("-inf"))
            values, ids = scores.max(dim=-1)
            local = torch.stack((values.float(), (ids + self.vocab_start).float()), -1)
            pairs = tensor_model_parallel_all_gather(local, dim=-1).view(rows, 2, 2)
            winner = pairs[..., 0].argmax(-1, keepdim=True)
            prev = pairs[..., 1].gather(-1, winner).squeeze(-1).long()
            output[:, step].copy_(prev)
