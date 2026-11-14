# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor.

`SpecializedDecode` runs the decoder layers of `MiniMaxM3Model` for the
steps captured into FULL CUDA graphs of pure decode batches of up to
`max_rows` rows. Its model state builds it once the weights are loaded.
`plan` decides per forward whether it runs; the model's forward then passes
the returned `DecodeStep` to the backbone, which embeds the input as usual
and hands the layers to `DecodeStep.run`. Every other forward (prefill,
mixed, eager, PIECEWISE, larger batches) runs the stock layers, which the
executor leaves unchanged.

Per layer the executor runs the stock modules' weights through the stages of
`stages.py`:

- dense-attention layers (0-2): allreduce_norm.input -> front -> dense_decode
  -> o_proj -> allreduce_norm.post_attention -> dense_mlp;
- sparse-attention layers: allreduce_norm.input -> front -> index_score ->
  page_topk -> sparse_decode -> o_proj -> allreduce_norm.post_attention
  (+ router) -> moe (finalize deferred to the next input boundary when the
  stock MoE defers it and the next front does not fuse that boundary);

then allreduce_norm.final. A front that `fuses_input` replaces
allreduce_norm.input + front with `run_fused`; a MoE whose next front
`polls_push` pushes its output into that front's exchange. Each stage takes
the first of its implementations that runs over the deployment's cache dtypes,
supports the step's rows and is `compatible` with the other selected stages
(`Stages.select`); the last is always the stock one (`stock.py`).

The stages issue TP collectives, so every decision reads host state that all
TP ranks share: the step's rows and the attention metadata's host fields.
"""

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.sequence import IntermediateTensors

from .. import DecodeConfig, max_rows
from ..stages import Stages, StepStages
from ..stock import stock_stages

if TYPE_CHECKING:
    from ...model import MiniMaxM3DecoderLayer, MiniMaxM3Model

DecodeOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]


class DecodeStep(NamedTuple):
    """The host-side choices of one captured decode step, by its rows."""

    rows: int
    # The implementation of every stage for this step.
    stages: StepStages
    executor: "SpecializedDecode"

    def run(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> DecodeOutput:
        """The backbone's output after its embedding."""
        return self.executor.run(self, hidden_states, positions)


class LayerPlan(NamedTuple):
    layer_id: int
    layer: "MiniMaxM3DecoderLayer"
    # Sparse (MSA) attention with the lightning indexer, or dense attention.
    sparse: bool
    moe: bool


def default_stages(
    config: DecodeConfig, backbone: "MiniMaxM3Model | None" = None
) -> Stages:
    """The stage implementations of a deployment: the kernel stages
    (`kernel_stages.py`) that run over its cache dtypes, each ahead of the
    stock stage. Without a backbone, the stock chain."""
    stock = stock_stages()
    # The stock stage object of every slot (the kernel stages delegate to it).
    stock_by_slot = {name: impls[-1] for name, impls in stock._asdict().items()}
    if backbone is None:
        extra: dict[str, list] = {}
    else:
        from ..kernel_stages import kernel_stage_lists

        extra = kernel_stage_lists(backbone, config, stock_by_slot)
    stages = Stages(
        *([*extra.get(name, ()), *impls] for name, impls in stock._asdict().items())
    )
    return stages.for_cache_dtypes(config.cache_dtypes)


def captures_pure_decode(prefixes: list[str]) -> bool:
    """Whether the forward being captured into a CUDA graph is a FULL graph of
    a pure decode step: not a PIECEWISE capture (Model Runner V2 captures FULL
    graphs with the forward context's mode NONE, and PIECEWISE ones, which
    report capturing too, with PIECEWISE), and attention metadata without
    prefills for every attention layer and index cache in `prefixes`."""
    context = get_forward_context()
    attn_metadata = context.attn_metadata
    if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE or not isinstance(
        attn_metadata, dict
    ):
        return False
    for prefix in prefixes:
        metadata = attn_metadata.get(prefix)
        if metadata is None or getattr(metadata, "num_prefills", 0) > 0:
            return False
    return True


class SpecializedDecode:
    """The decode executor over one backbone's local layers."""

    def __init__(
        self,
        backbone: "MiniMaxM3Model",
        config: DecodeConfig,
        lm_head: nn.Module,
        stages: Stages | None = None,
    ) -> None:
        from ...model import MiniMaxM3DecoderLayer, MiniMaxM3SparseAttention

        self.backbone = backbone
        self.config = config
        self.lm_head = lm_head
        self.max_rows = max_rows(config.query_len)
        self.stages = default_stages(config, backbone) if stages is None else stages
        if any(not impls for impls in self.stages):
            raise ValueError(
                "decode_backend=latency_specialized has no implementation of "
                f"some stage for the {config.cache_dtypes} caches"
            )
        if backbone.use_mega_moe:
            raise ValueError(
                "decode_backend=latency_specialized does not run the MegaMoE experts"
            )
        self.plans: list[LayerPlan] = []
        prefixes: set[str] = set()
        for layer_id in range(backbone.start_layer, backbone.end_layer):
            layer = backbone.layers[layer_id]
            assert isinstance(layer, MiniMaxM3DecoderLayer)
            attn = layer.self_attn
            sparse = isinstance(attn, MiniMaxM3SparseAttention)
            if sparse:
                self._check_sparse_attention(attn)
                prefixes.add(attn.layer_name)
                prefixes.add(attn.indexer.index_cache.prefix)
            else:
                prefixes.add(attn.attn.layer_name)
            self.plans.append(
                LayerPlan(
                    layer_id=layer_id,
                    layer=layer,
                    sparse=sparse,
                    moe=layer.is_moe_layer,
                )
            )
        self._prefixes = sorted(prefixes)

    @staticmethod
    def _check_sparse_attention(attn) -> None:
        from ...indexer_msa import MiniMaxM3IndexerMSAImpl
        from ...sparse_attention_msa import MiniMaxM3SparseMSAImpl

        # unsupported() refuses the NVFP4 caches that select the other impls.
        if not isinstance(attn.indexer.impl, MiniMaxM3IndexerMSAImpl):
            raise ValueError(
                "decode_backend=latency_specialized needs the MSA indexer, not "
                f"{type(attn.indexer.impl).__name__}"
            )
        if not isinstance(attn.impl, MiniMaxM3SparseMSAImpl):
            raise ValueError(
                "decode_backend=latency_specialized needs the MSA sparse "
                f"attention, not {type(attn.impl).__name__}"
            )

    @classmethod
    def for_target(
        cls, backbone: "MiniMaxM3Model", config: DecodeConfig, lm_head: nn.Module
    ) -> "SpecializedDecode":
        """The executor of the target model's backbone."""
        return cls(backbone, config, lm_head)

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        """The step this forward runs, or None to run the stock layers: only
        FULL CUDA graph captures of pure decode steps of up to `max_rows`
        rows run the executor."""
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= self.max_rows
            or not is_forward_context_available()
        ):
            return None
        if not torch.cuda.is_current_stream_capturing():
            # The eager run before each capture (and every eager step) runs the stock
            # layers, which also compiles the stock kernels.
            return None
        if not captures_pure_decode(self._prefixes):
            return None
        return DecodeStep(rows=rows, stages=self.stages.select(rows), executor=self)

    def run(
        self, step: DecodeStep, x: torch.Tensor, positions: torch.Tensor
    ) -> DecodeOutput:
        """The decoder layers and the final norm of the step, on the embedding
        `x`; returns what `MiniMaxM3Model.forward` returns."""
        backbone = self.backbone
        aux_layers = backbone.aux_hidden_state_layers
        aux_hidden_states: list[torch.Tensor] = []
        s = step.stages
        for stage in s.distinct():
            stage.begin_step(step, positions)
        pending: torch.Tensor | object = x
        residual: torch.Tensor | None = None
        plans = self.plans
        for i, plan in enumerate(plans):
            layer = plan.layer
            attn = layer.self_attn
            if residual is not None and s.front.fuses_input(layer):
                front, residual = s.front.run_fused(
                    layer,
                    pending,  # type: ignore[arg-type]
                    residual,
                    positions,
                )
            else:
                bound = s.allreduce_norm.input(layer, pending, residual)  # type: ignore[arg-type]
                residual = bound.residual
                front = s.front.run(attn, bound.hidden, positions)
            # Aux hidden state k is the residual stream entering layer k, read once the
            # input boundary has reduced it, and cloned (as the stock forward): the
            # stages update the residual buffers in place.
            if plan.layer_id in aux_layers:
                aux_hidden_states.append(residual.clone())
            if plan.sparse:
                assert front.index_q is not None
                scores = s.index_score.run(attn, front.index_q)
                topk = s.page_topk.run(attn, scores)
                attn_out = s.sparse_decode.run(attn, front.q, topk)
            else:
                attn_out = s.dense_decode.run(attn, front)
            partial = s.o_proj.run(attn, attn_out)
            bound = s.allreduce_norm.post_attention(layer, partial, residual)
            residual = bound.residual
            hidden = bound.hidden
            assert isinstance(hidden, torch.Tensor)
            if plan.moe:
                moe = layer.block_sparse_moe
                rows = hidden.shape[0]
                successor = plans[i + 1].layer if i + 1 < len(plans) else None
                # A front that fuses the next input boundary lands a BF16 partial, the
                # push, or (``fuses_finalize``) the unfinalized MoE output.
                fused_next = successor is not None and s.front.fuses_input(successor)
                may_defer = not fused_next or s.front.fuses_finalize(successor)  # type: ignore[arg-type]
                push = (
                    fused_next
                    and s.front.polls_push(successor)  # type: ignore[arg-type]
                    and s.moe.can_push(moe, rows)
                )
                pending = s.moe.run(
                    moe,
                    hidden,
                    bound.router_logits,
                    defer_finalize=may_defer and moe.defers_finalize(rows),
                    push=push,
                )
            else:
                pending = s.dense_mlp.run(layer.mlp, hidden)
        assert residual is not None
        bound = s.allreduce_norm.final(backbone, pending, residual)  # type: ignore[arg-type]
        hidden_states = bound.hidden
        assert isinstance(hidden_states, torch.Tensor)
        if backbone.end_layer in aux_layers:
            aux_hidden_states.append(bound.residual.clone())
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
