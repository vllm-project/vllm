# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor over the stock Muse Spark backbone: one plan per
captured decode graph, then every layer's providers in one loop.

One step of b rows (one request verifying 1 + 7 tokens):

    embedding  token ids -> embedded tokens: the stock embedding, eagerly before
               forward on the Omni class, in the graph on the text class
    per layer  the first layer: the embedded tokens into the stream (fp32) and
               its pre_qkv_proj_norm into the attention input, one kernel
               input boundary (later layers): gather of the previous MoE's
               unpooler blocks + the dual norm into the stream and this layer's
               attention input
               attention: QKV front -> attn2 -> o_proj partial (TP8: published
               into the attention boundary's mailbox from the GEMM epilogue)
               attention boundary: all-reduce + the dual norm into the stream
               and the MoE input
               MoE: router -> routing -> pooler -> routed experts -> tail ->
               unpooler, published zero-copy into the next boundary's mailbox
               (TP8: the NVFP4 layers' experts run expert TP, and their tail
               is the TP tail instead of the EP tail)
    final      the last gather + dual norm into model.norm

Every provider is built here, in one order that is the same on every rank: the
builds are collective (symmetric-memory rendezvous, warm-ups that all-reduce).
"""

from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..cutedsl.boundary.all_gather_dual_norm import FinalBoundary, GatherBoundary
from ..cutedsl.boundary.layer_front import LayerFront
from ..cutedsl.common import policy
from ..cutedsl.moe.pooler_gather import Pooler, pooler_layer
from ..cutedsl.moe.tail import MoETail, TpTail
from ..cutedsl.oproj.batt import AttentionBoundary
from . import DecodeConfig, flashinfer_ops
from .providers import (
    BATCHES,
    HIDDEN,
    PREPARED_MAX,
    BuildContext,
    QkvAttention,
    RoutedMoE,
    Step,
)

logger = init_logger(__name__)


class Layout(NamedTuple):
    """The provider class of each block, for one TP size."""

    boundary: Any
    attention: Any
    attention_boundary: Any
    moe: Any
    final: Any


# TP4 (one tray): the all-reduce fused into the attention boundary in the
# stock all-reduce's order (MNNVL's by default, TRT-LLM's when set), the attn2
# KV-head grid (4 KV heads per rank). TP8 (two trays): the all-reduce fused
# into the attention boundary in the MNNVL all-reduce's order, 2 KV heads per
# rank. TP16 (four trays): the MNNVL all-reduce through vLLM's wrapper
# (``RoutedExpertsOps.all_reduce``) for the attention boundary, 1 KV head per
# rank.
# The classes take their TP form at build.
LAYOUT = Layout(
    GatherBoundary,
    QkvAttention,
    AttentionBoundary,
    RoutedMoE,
    FinalBoundary,
)
LAYOUTS: dict[int, Layout] = {4: LAYOUT, 8: LAYOUT, 16: LAYOUT}


def provider_requirements(vllm_config) -> str | None:
    """The first failing requirement of the layout's providers, or None."""
    layout = LAYOUTS.get(vllm_config.parallel_config.tensor_parallel_size)
    if layout is None:
        return "TP other than 4/8/16"
    providers = (
        layout.boundary,
        layout.attention,
        layout.attention_boundary,
        layout.moe,
        layout.final,
    )
    reasons = (provider.unsupported(vllm_config) for provider in providers)
    reason = next((reason for reason in reasons if reason is not None), None)
    return reason if reason is not None else flashinfer_ops.unsupported(vllm_config)


def layer_metadata(layer, context):
    """The layer's attention metadata in this forward context, or None."""
    md = context.attn_metadata
    if isinstance(md, list):
        # Speculative decoding: [0] is the target model's.
        md = md[0] if md else None
    return md.get(layer.self_attn.attn.layer_name) if isinstance(md, dict) else None


def decode_metadata(layer, context, q: int, b: int):
    """The layer's metadata if this step is one request verifying all b = q tokens,
    else None."""
    md = layer_metadata(layer, context)
    if md is None or md.num_prefills or md.num_decodes != 1:
        return None
    return md if md.num_decode_tokens == q == b else None


class DecodeLayer(NamedTuple):
    """The providers of one decoder layer."""

    # The input boundary; for the first layer, whose input is the embedded
    # tokens, its front
    # (LayerFront: the embedded tokens into the stream and the attention input).
    boundary: Any
    attention: QkvAttention
    attention_boundary: AttentionBoundary
    moe: RoutedMoE
    # This layer's aux state buffer (the drafter's input), or None.
    aux: torch.Tensor | None


class SpecializedDecode:
    """Built once the weights are loaded and the aux layers set; owns every provider
    and the decode buffers. Reads the backbone, never modifies it."""

    def __init__(self, backbone: torch.nn.Module, config: DecodeConfig):
        self.backbone, self.q = backbone, config.q
        group = get_tp_group()
        self.tp, self.rank = group.world_size, group.rank_in_group
        assert self.tp == config.tp
        self.device = torch.device(current_platform.current_device())
        self.start_layer = backbone.start_layer
        self.layers = tuple(backbone.layers[backbone.start_layer : backbone.end_layer])
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        self.residual_dtype = backbone.residual_dtype
        self.buffers: dict[str, torch.Tensor] = {}
        # Decided once: the fused all-reduces sum in the stock all-reduce's
        # order (collective: creates its workspace on first use).
        all_reduce_order = (
            flashinfer_ops.stock_sum_order(group.device_communicator)
            if self.tp in policy.FUSED_TPS
            else None
        )
        ctx = BuildContext(
            backbone,
            self.tp,
            self.rank,
            self.device,
            group.device_group,
            self.buffers,
            self.aux_layers,
            config.max_context,
            all_reduce_order,
        )
        self.layout = layout = LAYOUTS[self.tp]
        self.build(ctx, layout)
        # Capture-time outcomes already logged; plan() never logs outside capture.
        self.logged: set[str] = set()
        logger.info(
            "Muse Spark specialized decode built: TP%d, %d layers, aux layers %s",
            self.tp,
            len(self.layers),
            self.aux_layers,
        )

    def build(self, ctx: BuildContext, layout: Layout) -> None:
        """Every provider, in the same order on every rank (the builds are
        collective)."""
        tp, rank, device = self.tp, self.rank, self.device

        def buffer(name: str, dtype: torch.dtype) -> torch.Tensor:
            tensor = torch.zeros(PREPARED_MAX, HIDDEN, dtype=dtype, device=device)
            self.buffers[name] = tensor
            return tensor

        self.stream = buffer("stream", torch.float32)
        self.attention_input = buffer("attention_input", torch.bfloat16)
        self.moe_input = buffer("moe_input", torch.bfloat16)
        self.output = buffer("output", torch.bfloat16)

        # The routed expert ops of every layer (expert TP at
        # policy.EXPERT_TP_TPS) and, at TP16, the all-reduce.
        self.ops = ops = flashinfer_ops.build(
            get_tp_group(), self.layers, use_expert_tp=tp in policy.EXPERT_TP_TPS
        )
        all_reduce = (
            self.ops.all_reduce if tp in flashinfer_ops.ALL_REDUCE_TPS else None
        )
        for layer in self.layers:
            flashinfer_ops.check_attention(layer.self_attn.attn)

        # The first layer's front, the input boundary of every layer after it,
        # then the final boundary: the boundaries' mailboxes receive the
        # unpooler blocks of the layer before.
        boundaries = [LayerFront.build(ctx, self.layers[0], max(BATCHES))] + [
            layout.boundary.build(ctx, layer) for layer in self.layers[1:]
        ]
        self.final = layout.final.build(ctx, None)
        mailboxes = [b.mailbox.lamport for b in boundaries[1:]]
        mailboxes.append(self.final.mailbox.lamport)

        pooler = Pooler(
            ctx.group,
            tp=tp,
            rank=rank,
            device=device,
            layers=[
                pooler_layer(layer.feed_forward, tp, rank) for layer in self.layers
            ],
            batches=BATCHES,
        )
        pooler.warm_up()
        # The EP tail for the layers whose experts stay EP (all of them
        # without expert TP), the TP tail for the expert-TP layers; each is
        # built only if some layer takes it.
        expert_tp = [ops.is_expert_tp(layer.layer_idx) for layer in self.layers]
        ep_tail = tp_tail = None
        if not all(expert_tp):
            ep_tail = MoETail(ctx.group, tp=tp, rank=rank, device=device)
        if any(expert_tp):
            tp_tail = TpTail(
                ctx.group, tp=tp, rank=rank, device=device, batches=BATCHES
            )
            logger.info(
                "Muse Spark expert TP tail: TP%d, %d layers, %d bands per token "
                "at %d tokens",
                tp,
                sum(expert_tp),
                tp_tail.split(BATCHES[0]),
                BATCHES[0],
            )

        layers = []
        for i, layer in enumerate(self.layers):
            index = layer.layer_idx
            aux = (
                buffer(f"aux{index}", torch.float32)
                if index in self.aux_layers
                else None
            )
            boundary = boundaries[i]
            writes_aux = boundary.aux if i == 0 else boundary.kernel.aux
            if writes_aux != (aux is not None):
                raise RuntimeError(f"layer {index}: aux state mismatch")
            layers.append(
                DecodeLayer(
                    boundary,
                    layout.attention.build(ctx, layer),
                    layout.attention_boundary.build(ctx, layer, all_reduce=all_reduce),
                    layout.moe.build(
                        ctx,
                        layer,
                        ops=ops,
                        pooler=pooler,
                        pooler_index=i,
                        tail=tp_tail if expert_tp[i] else ep_tail,
                        mailbox=mailboxes[i],
                    ),
                    aux,
                )
            )
        self.decode_layers = tuple(layers)
        end = self.backbone.end_layer
        self.final_aux = (
            buffer(f"aux{end}", torch.float32) if end in self.aux_layers else None
        )
        if self.final.kernel.aux != (self.final_aux is not None):
            raise RuntimeError("final boundary: aux state mismatch")
        self.final_slot = self.final.mailbox.slot(self.final.index, 0)
        # The mailbox slot each input boundary gathers from, for the layers
        # after the first.
        self.slots = tuple(b.mailbox.slot(b.index, 0) for b in boundaries[1:])
        torch.accelerator.synchronize(device)

    def prepare_kv(self) -> None:
        """Prepare every provider for the bound KV caches (the profiling cache, then
        the real one)."""
        for layer in self.decode_layers:
            layer.attention.prepare_kv()

    def plan(self, positions, intermediate_tensors) -> Step | None:
        """This step's decode plan, or None to run the stock forward instead.

        Steps that run the stock forward:
        - pipeline-parallel stage inputs (``intermediate_tensors``);
        - token counts other than ``BATCHES``;
        - steps outside a CUDA graph capture (profiling, warmup and the warmup run
          before each capture), so every stock kernel runs before its capture;
        - PIECEWISE steps, keyed by token count alone and so captured as stock;
        - steps other than one request verifying q tokens (prefills, several
          requests, partial drafts);
        - decode metadata other than the TRT-LLM form attn2 reads;
        - steps whose QKV front call the kernel does not take (positions or slot
          mapping in another form).

        attn2 serves every context length up to max_model_len and each layer's
        sliding window.

        Every predicate reads host state, the same on all TP ranks: the providers'
        collectives need every rank to take the same path.
        """
        b = positions.shape[0]
        if intermediate_tensors is not None or b not in BATCHES:
            return None
        if not torch.cuda.is_current_stream_capturing():
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return self.captured(b, "PIECEWISE graph")
        metadata = tuple(
            decode_metadata(layer, context, self.q, b) for layer in self.layers
        )
        if any(md is None for md in metadata):
            return self.captured(b, "not one request verifying all tokens")
        if not all(flashinfer_ops.trtllm_decode(md) for md in metadata):
            return self.captured(b, "decode metadata other than the TRT-LLM form")
        step = Step(self.q, positions, metadata, context.slot_mapping)
        x = self.attention_input[:b]
        if any(
            layer.attention.front_call(step, x) is None for layer in self.decode_layers
        ):
            return self.captured(b, "a step the QKV front does not take")
        return self.captured(b, None, step)

    def captured(
        self, b: int, fallback: str | None, step: Step | None = None
    ) -> Step | None:
        """Logs each capture-time outcome once and returns ``step``."""
        outcome = f"stock forward ({fallback})" if fallback else "specialized decode"
        message = f"Muse Spark decode graph for {b} tokens: {outcome}"
        if message not in self.logged:
            self.logged.add(message)
            logger.info(message)
        return step

    def run(
        self,
        step: Step,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Normalized hidden states (and the aux streams) of a planned step."""
        x = inputs_embeds
        if x is None:
            assert input_ids is not None
            x = self.backbone.embed_input_ids(input_ids)
        b = x.shape[0]
        stream = self.stream[:b]
        attention_input, moe_input = self.attention_input[:b], self.moe_input[:b]
        aux_out: list[torch.Tensor] = []
        # The first layer's front takes the embedded tokens, each input
        # boundary its slot.
        inputs = (x.contiguous(), *self.slots)
        for layer, source, metadata in zip(self.decode_layers, inputs, step.metadata):
            aux = None if layer.aux is None else layer.aux[:b]
            layer.boundary.run(source, stream, attention_input, aux)
            if aux is not None:
                aux_out.append(aux)
            partial = layer.attention.run(step, metadata, attention_input)
            layer.attention_boundary.run(partial, stream, moe_input)
            layer.moe.run(moe_input)
        output = self.output[:b]
        final_aux = None if self.final_aux is None else self.final_aux[:b]
        self.final.run(self.final_slot, stream, output, final_aux)
        if final_aux is not None:
            aux_out.append(final_aux)
        return (output, aux_out) if aux_out else output
