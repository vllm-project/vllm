# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEP target and draft layer loops, opt-in, without a TP mailbox."""

from types import SimpleNamespace

import torch

from vllm.forward_context import is_forward_context_available
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer
from vllm.models.deepseek_v41.common.ops.query_quant import can_fuse_query_quant
from vllm.models.deepseek_v41.nvidia.specialized_decode.engram.prefetch import (
    EngramPrefetch,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.decode import (
    captures_pure_decode,
    finish_layers,
    flashinfer_attentions,
    step_gates,
)

from ..attention.indexer import CandidateOverlap
from ..attention.sparse_mla import SparseIndexTables, SparseMlaDecode
from ..attention.step import AttentionStep
from .attention.projections import ProjectionBank
from .boundary import LocalBoundaries
from .communication import DEPCommunication
from .moe import DEPMoEStep
from .workspace import ExpertWorkspace


class DEPDecode:
    embedding = None
    all_gather = None
    is_dep = True

    def __init__(self, backbone, config, lm_head=None, target=None):
        if backbone.use_sequence_parallel or (
            target is None and backbone.use_native_mega_moe
        ):
            raise ValueError(
                "DEP specialized decode requires modular TRTLLM MoE, no SP"
            )
        self.backbone, self.config, self.lm_head = backbone, config, lm_head
        self.is_draft = target is not None
        layer_ids = (
            range(len(backbone.layers))
            if self.is_draft
            else range(backbone.start_layer, backbone.end_layer)
        )
        layers = [backbone.layers[i] for i in layer_ids]
        attns = flashinfer_attentions(layers)
        self.mhc_stream = torch.cuda.Stream() if target is None else target.mhc_stream
        self.boundaries = LocalBoundaries(layers, self.mhc_stream)
        self.projections = ProjectionBank(config.max_rows, forward_only=True)
        self.tables = SparseIndexTables(attns)
        self.mla = SparseMlaDecode(
            self.tables,
            config.max_rows * max(a.padded_heads for a in attns),
            wide_c8=config.wide_c8_attention and not self.is_draft,
        )
        self.workspace = (
            ExpertWorkspace(attns[0].fused_wqa_wkv.weight.device, config.max_rows)
            if target is None
            else target.workspace
        )
        source = next(
            (
                a
                for a in attns
                if a.indexer is not None and a.layer_id == a.candidate_source_layer
            ),
            None,
        )
        consumers = [
            a
            for a in attns
            if source is not None
            and a is not source
            and a.indexer is not None
            and a.candidate_source_layer == source.layer_id
            and isinstance(a.indexer.indexer_op, SparseMQAIndexer)
        ]
        self.candidates = None
        if consumers:
            indexer = consumers[0].indexer
            assert indexer is not None
            self.candidates = CandidateOverlap(indexer.indexer_op)
        self.engram = EngramPrefetch(
            [layer.engram for layer in layers if layer.engram is not None],
            config.max_rows,
        )
        self.plans = [
            SimpleNamespace(
                layer_id=i,
                layer=layer,
                engram=layer.engram,
                attention=AttentionStep(
                    attn,
                    self.tables,
                    self.mla,
                    publisher=None,
                    mxfp8_input=can_fuse_query_quant([attn.fused_wqa_wkv]),
                    source=self.candidates if attn is source else None,
                    consumer=self.candidates if attn in consumers else None,
                    all_gather=None,
                    projection_bank=self.projections,
                ),
                moe=DEPMoEStep(layer.ffn, self.workspace, config.max_rows),
            )
            for i, layer, attn in zip(layer_ids, layers, attns)
        ]
        self._swa_prefixes = sorted({a.swa_cache_layer.prefix for a in attns})
        self._prepared_rows = set()
        self.communication = DEPCommunication(
            self.plans, config, None if target is None else target.communication
        )

    @classmethod
    def for_target(cls, backbone, config, lm_head):
        return cls(backbone, config, lm_head)

    def for_draft(self, backbone):
        return type(self)(backbone, self.config, target=self)

    def prepare_rows(self, rows):
        if rows in self._prepared_rows:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DEP shape was not prepared before capture")
        from .warmup import prepare_rows

        prepare_rows(self, rows)
        self.communication.prepare_rows(rows)
        self._prepared_rows.add(rows)

    def plan(self, positions, intermediate_tensors=None):
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= self.config.max_rows
            or not is_forward_context_available()
        ):
            return None
        if not torch.cuda.is_current_stream_capturing():
            self.prepare_rows(rows)
            if self.candidates is not None:
                self.candidates.warm()
            return None
        if rows not in self._prepared_rows or not captures_pure_decode(
            self._swa_prefixes
        ):
            return None
        return step_gates(rows, self)

    def run(
        self,
        step,
        x,
        positions,
        input_ids,
        engram_hashes=None,
        engram_mask=None,
        streams=None,
    ):
        if self.is_draft and (
            engram_hashes is not None or engram_mask is not None or streams is not None
        ):
            raise ValueError("unexpected target state at the draft entry")
        aux_layers = () if self.is_draft else self.backbone.aux_hidden_state_layers
        aux = {}
        bound = self.boundaries
        self.tables.join()
        self.tables.reset()
        state = None
        try:
            if engram_hashes is not None:
                self.engram.start(x.shape[0])
            for plan in self.plans:
                quant = None
                if state is None:
                    state, layer_input = bound.first(plan.layer, x, streams)
                elif plan.engram is not None and engram_hashes is not None:
                    streams = bound.post_only(state)
                    if plan.layer_id in aux_layers:
                        aux[plan.layer_id] = streams.mean(dim=1)
                    injected = self.engram.forward(
                        plan.engram,
                        streams,
                        engram_hashes[:, plan.engram.layer_hash_index],
                        engram_mask,
                    )
                    state, layer_input = bound.pre(plan.layer, injected, state)
                else:
                    state, layer_input, quant, layer_aux = bound.sublayer(
                        plan.layer,
                        False,
                        state,
                        quantize=plan.attention.mxfp8_input,
                        capture_aux=plan.layer_id in aux_layers,
                    )
                    if layer_aux is not None:
                        aux[plan.layer_id] = layer_aux
                state.pending = plan.attention.run(layer_input, quant, positions, step)
                bound.join()
                state, ffn_input, _, _ = bound.sublayer(plan.layer, True, state)
                state.pending = plan.moe.run(ffn_input, input_ids, step)
                bound.join()
            if state is None:
                raise RuntimeError("no decoder layers")
            streams = bound.post_only(state)
            if not self.is_draft and self.backbone.end_layer in aux_layers:
                aux[self.backbone.end_layer] = streams.mean(dim=1)
        except BaseException:
            # Also join aux projections on an exception between fork and first
            # reader; otherwise capture can leave unrelated work unjoined.
            self._join_side_work()
            current = torch.cuda.current_stream()
            current.wait_stream(self.mhc_stream)
            streams_to_join = {
                s for p in self.plans for s in p.attention.attn.aux_stream_list or ()
            }
            for stream in streams_to_join:
                current.wait_stream(stream)
            raise
        self._join_side_work()
        if self.is_draft:
            return hc_collapse_triton(streams, state.pre)
        return finish_layers(self.backbone, streams, state.pre, aux)

    def _join_side_work(self):
        self.tables.join()
        if self.candidates is not None:
            self.candidates.join()
        self.engram.join()

    def warm_up(self, vllm_config, drafts):
        from .warmup import warm_up

        warm_up(self, vllm_config, drafts)
