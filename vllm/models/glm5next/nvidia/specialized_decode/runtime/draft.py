# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode of the MTP draft layer: the main model's stages on the
draft's FULL-graph steps, and a vocab-parallel argmax for greedy drafts.

The MTP layer has no mHC streams, so its boundaries are a fused all-reduce,
residual add and RMSNorm (``AllReduceNorm``) where the main model fuses the
all-reduce with mHC. The vocab-parallel embedding's all-reduce takes the same
mailbox, fused with the enorm / hnorm of the eh_proj input. The draft's BF16
projections and LM head shard run the split-K GEMM above two tokens.
"""

from typing import Any

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.glm5next.nvidia.ops.fused_eh_norm import fused_eh_norm

from ..linear import MAX_TOKENS as PROJECTION_MAX_TOKENS
from ..linear import planned_step
from ..policy import Policy
from .decode import (
    DSA_PROJECTIONS,
    INDEXER_PROJECTIONS,
    MAX_TOKENS,
    LayerStages,
    Step,
    is_decode,
)

logger = init_logger(__name__)

# Draft rows whose greedy token the vocab-parallel argmax serves.
ARGMAX_MAX_TOKENS = 64


class SpecializedDraft(LayerStages):
    """Built on the first eager draft forward (the weights are processed by
    then). Reads the MTP layers, never modifies them."""

    def __init__(self, predictor, config, greedy: bool):
        self.config = config
        self.policy = Policy(config.policy)
        self.q = config.q
        self.mtp_layers = list(predictor._mtp_layers)
        self.layers = [layer.mtp_block for layer in self.mtp_layers]
        self.text_config = self.mtp_layers[0].config
        # Every MTP MoE output feeds a fused boundary.
        self.moe_layers = self.layers
        # Draft decode steps run one token per request; draft prefills run q.
        self.batches = sorted(b for b in config.capture_sizes if b <= MAX_TOKENS)
        # Token counts whose BF16 projections take the split-K GEMM.
        self.split_k_tokens = [b for b in self.batches if b > PROJECTION_MAX_TOKENS]
        self.kda: dict[int, Any] = {}
        self.eh_proj: dict[int, Any] = {}
        if self.policy.enabled("dsa_proj"):
            from ..gemm import Projection
            from ..linear import route

            routed = 0
            for mtp, layer in zip(self.mtp_layers, self.layers):
                attn = layer.self_attn
                modules = [getattr(attn, n, None) for n in DSA_PROJECTIONS]
                routed += sum(
                    m is not None and route(m, self.split_k_tokens) for m in modules
                )
                indexer = getattr(attn, "indexer", None)
                modules = [getattr(indexer, n, None) for n in INDEXER_PROJECTIONS]
                routed += sum(m is not None and route(m) for m in modules)
                self.eh_proj[layer.layer_idx] = Projection(
                    mtp.eh_proj.weight, self.split_k_tokens
                )
            logger.info("Specialized draft routes %d DSA GEMMs", routed)
        if self.policy.enabled("dsa_indexer"):
            from ..indexer import IndexerStage

            for layer in self.layers:
                IndexerStage(layer.self_attn.indexer)
        if self.policy.enabled("dsa_attn"):
            from ..dsa_attention import DsaAttentionStage

            for layer in self.layers:
                DsaAttentionStage(
                    layer.self_attn,
                    MAX_TOKENS,
                    self.policy.tokens("dsa_value_up", "cute"),
                )
        self.mega: dict[int, Any] = {}
        if self.policy.enabled("moe"):
            self.mega = self._build_mega()
        self.dense: dict[int, Any] = {}
        self.ar_norm = None
        self.deferred: set[int] = set()
        if self.policy.enabled("boundary"):
            self.ar_norm = self._build_ar_norm()
            if self.ar_norm is not None:
                self.deferred = self._defer_moe_finalize()
        self.embed_tokens = predictor.embed_tokens
        self.fused_embedding = self.ar_norm is not None and self._build_embedding()
        self.head: Any = None
        self.argmax = self._build_argmax() if greedy else None

    def _build_embedding(self) -> bool:
        """Whether the embedding's all-reduce can take the boundary mailbox."""
        assert self.ar_norm is not None
        weight, shard = self.embed_tokens.weight, self.embed_tokens.shard_indices
        norms = {
            n.variance_epsilon for m in self.mtp_layers for n in (m.enorm, m.hnorm)
        }
        if (
            weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or shard.num_added_elements
            or len(norms) != 1
        ):
            logger.warning("Specialized draft: stock embedding all-reduce")
            return False
        self.ar_norm.prepare_embedding(
            weight,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index - shard.org_vocab_start_index,
            norms.pop(),
        )
        return True

    def _build_ar_norm(self):
        from ..all_reduce_norm import AllReduceNorm

        norms = {
            n.variance_epsilon
            for mtp, layer in zip(self.mtp_layers, self.layers)
            for n in (layer.post_attention_layernorm, mtp.shared_head.norm)
        }
        if len(norms) != 1:
            logger.warning("Specialized draft: mixed norm eps, no fused boundary")
            return None
        return AllReduceNorm(
            hidden_size=self.layers[0].hidden_size,
            max_num_tokens=self.boundary_max_tokens,
            top_k=self.text_config.num_experts_per_tok,
            eps=norms.pop(),
            device=self.mtp_layers[0].eh_proj.weight.device,
        )

    def _build_argmax(self):
        from ..draft_argmax import DraftArgmax

        heads = [mtp.shared_head.head for mtp in self.mtp_layers]
        head = heads[0]
        weight = head.weight
        shard = head.shard_indices
        if weight.dtype != torch.bfloat16 or any(
            h.weight.shape != weight.shape for h in heads
        ):
            logger.warning("Specialized draft: unsupported LM head, stock argmax")
            return None
        columns = weight.shape[0]
        if self.policy.enabled("dsa_proj"):
            from ..gemm import Projection

            self.head = Projection(weight, self.split_k_tokens)
        return DraftArgmax(
            columns=columns,
            valid=columns - shard.num_org_vocab_padding,
            offset=shard.org_vocab_start_index,
            capacity=ARGMAX_MAX_TOKENS,
            device=weight.device,
        )

    def partial(self, tokens: int) -> bool:
        return self.ar_norm is not None and bool(self.policy("boundary", tokens))

    def plan(self, positions) -> Step | None:
        """This draft step's plan, or None for the stock forward: planned only
        inside FULL-graph captures of uniform draft prefills (q tokens per
        request) and draft decodes (one token per request)."""
        b = positions.shape[-1]
        if not torch.cuda.is_current_stream_capturing() or b > MAX_TOKENS:
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        by_layer = context.attn_metadata
        if not isinstance(by_layer, dict) or not by_layer:
            return None
        rows = list(by_layer.values())
        for q in dict.fromkeys((self.q, 1)):
            if b % q == 0 and all(is_decode(md, q, b) for md in rows):
                logger.info_once("Specialized draft captures %d-token steps", b)
                return Step(q, positions, by_layer)
        logger.info_once("Specialized draft skips a %d-token capture", b)
        return None

    def reduced_norm(self, x, partial, residual, norm):
        """``norm(x + residual)`` and the new residual of a sublayer output."""
        if partial:
            assert self.ar_norm is not None
            return self.ar_norm(x, residual, norm.weight.data)
        return norm(x, residual)

    def run(
        self, mtp, step: Step, input_ids, inputs_embeds, previous_hidden_states, rows
    ):
        """``Glm5NextMultiTokenPredictorLayer.forward`` of a planned step.

        ``rows``: the speculator's per-request last-token rows. A draft prefill
        (q tokens per request) samples only those rows, so after the attention
        (which writes every token's KV) only they run on; the other output rows
        are left unset."""
        layer = mtp.mtp_block
        tokens = step.positions.shape[0]
        if (
            inputs_embeds is None
            and self.fused_embedding
            and self.partial(tokens)
            and input_ids.dtype == torch.int32
        ):
            assert self.ar_norm is not None
            x = self.ar_norm.embed_norm(
                input_ids,
                step.positions,
                previous_hidden_states,
                self.embed_tokens.weight,
                mtp.enorm.weight,
                mtp.hnorm.weight,
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            x = fused_eh_norm(
                step.positions,
                inputs_embeds,
                previous_hidden_states,
                mtp.enorm.weight,
                mtp.hnorm.weight,
                mtp.enorm.variance_epsilon,
            )
        projection = self.eh_proj.get(layer.layer_idx)
        with planned_step():
            if projection is not None and projection.supports(tokens):
                x = projection(x)
            else:
                x = mtp.eh_proj(x)
            residual = x
            x = layer.input_layernorm(x)
            x, partial = self.attention(layer, x, step)
            if step.q > 1 and rows is not None:
                rows = rows[: tokens // step.q]
                x, residual = x[rows], residual[rows]
            else:
                rows = None
            x, residual = self.reduced_norm(
                x, partial, residual, layer.post_attention_layernorm
            )
            x, partial = self.mlp(layer, x)
            x, _ = self.reduced_norm(x, partial, residual, mtp.shared_head.norm)
        if rows is not None:
            out = x.new_empty((tokens, x.shape[1]))
            out.index_copy_(0, rows, x)
            x = out
        return x, x

    def top_tokens(self, head, logits_processor, hidden_states):
        """Greedy draft tokens, or None for the stock vocab-parallel argmax."""
        if self.argmax is None or not self.argmax.supports(hidden_states.shape[0]):
            return None
        if logits_processor.soft_cap is not None or logits_processor.scale != 1.0:
            return None
        tokens = hidden_states.shape[0]
        if (
            self.head is not None
            and torch.cuda.is_current_stream_capturing()
            and self.head.supports(tokens)
            and head.weight is self.head.weight
            and hidden_states.is_contiguous()
        ):
            logits = self.head(hidden_states)
        else:
            logits = logits_processor._apply_head(head, hidden_states, None)
        if logits.dtype != torch.bfloat16:
            return None
        return self.argmax(logits.contiguous())
