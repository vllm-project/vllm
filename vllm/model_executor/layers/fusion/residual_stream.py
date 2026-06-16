# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``ResidualStream``: the single manual-fusion entry point for a decoder layer.

The transformer residual stream is the running sum each sublayer reads from (via
RMSNorm) and writes back into. ``ResidualStream`` owns that read point: at each
sublayer boundary it performs the one transition the stream's distribution state
requires -- (all-reduce +) residual-add + RMSNorm + optional activation-quant --
and hands the (possibly pre-quantized) activation to the next consumer linear.
This is the manual-fusion endgame shape for RFC #43224: model ``forward`` becomes
``prepare_attn -> attn -> prepare_mlp -> mlp`` with no ``do_allreduce`` booleans
or ``fuse_allreduce`` branch, and is the only abstraction model code touches (it
subsumes the standalone ``fused_ar_rms_norm_quant`` kernel-dispatch helper).

It reads the layer's modules *live*, so it tracks late changes (e.g. eagle
replacing its layer-0 norm with ``nn.Identity()`` after ``__init__``). Whether the
layer defers its reduce is read from the row-linear's existing ``reduce_results``
flag -- the single source of truth -- so there is no separate fusion flag:
- linears defer (``reduce_results=False``): inputs arrive ``PARTIAL``; the stream
  reduces them once while normalizing.
- linears reduce themselves (``reduce_results=True``): inputs arrive ``FULL``; the
  stream only normalizes. This is how MoE / unfused reusers share the same forward
  with no special-casing.
A second all-reduce is impossible by construction (``FULL`` never reduces), and a
non-RMSNorm ``norm`` (eagle Identity) is tolerated.

Scope: plain all-reduce only. Sequence-parallelism is sketched (``Scatter.SHARD``
+ reduce-scatter/all-gather in ``_reduce_norm``) but not implemented.
"""

from enum import Enum, auto

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fusion.ar_rms_quant import fused_ar_rms_norm_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm

NormOut = tuple[torch.Tensor | QuantizedActivation, torch.Tensor]


class Scatter(Enum):
    """Distribution state of a hidden-states tensor across the TP group."""

    FULL = auto()  # replicated / already all-reduced
    PARTIAL = auto()  # per-rank partial sum; owes exactly one all-reduce
    # SHARD = auto()  # sequence-parallel token shard (future: RS/AG transitions)


class ResidualStream:
    """Owns (AR +) residual-add + RMSNorm(+quant) for one decoder layer.

    Holds the decoder ``layer`` and reads its norms / consumer linears / reduce
    mode live, so model ``forward`` is just ``prepare_attn`` -> attn ->
    ``prepare_mlp`` -> mlp, with no ``tp_size``/``do_allreduce``/fuse branch.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        self.layer = layer
        self.tp_size = get_tensor_model_parallel_world_size()

    @property
    def defer(self) -> bool:
        # Single source of truth: the attention o_proj's reduce_results. A layer
        # whose row-linears skip their reduce leaves PARTIAL output for us.
        return not getattr(self.layer.self_attn.o_proj, "reduce_results", True)

    def _reduce_norm(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        norm: torch.nn.Module,
        incoming: Scatter,
        consumer_linear: torch.nn.Module | None,
    ) -> NormOut:
        do_allreduce = incoming is Scatter.PARTIAL and self.tp_size > 1
        # PARTIAL -> all-reduce + add + norm + (quant); FULL -> norm only.
        # An SP build would branch here on Scatter.SHARD to a
        # reduce-scatter+norm (and all-gather) transition instead.
        if type(norm) is RMSNorm:
            return fused_ar_rms_norm_quant(
                hidden,
                residual,
                norm,
                consumer_linear=consumer_linear,
                do_allreduce=do_allreduce,
            )
        # Non-fusable norm (e.g. eagle's Identity layer-0 norm): reduce if owed,
        # then run the plain residual-add + norm in eager ops.
        if do_allreduce:
            hidden = tensor_model_parallel_all_reduce(hidden)
        if residual is None:
            return norm(hidden), hidden
        return norm(hidden, residual)

    def prepare_attn(
        self, hidden: torch.Tensor, residual: torch.Tensor | None
    ) -> NormOut:
        # Layer 0 receives the already-reduced embedding (FULL). Later layers
        # receive the previous layer's output: PARTIAL iff this model defers.
        incoming = (
            Scatter.PARTIAL if (self.defer and residual is not None) else Scatter.FULL
        )
        return self._reduce_norm(
            hidden,
            residual,
            self.layer.input_layernorm,
            incoming,
            self.layer.self_attn.qkv_proj,
        )

    def prepare_mlp(self, hidden: torch.Tensor, residual: torch.Tensor) -> NormOut:
        # Attention o_proj output is PARTIAL iff it ran with reduce_results=False.
        incoming = Scatter.PARTIAL if self.defer else Scatter.FULL
        return self._reduce_norm(
            hidden,
            residual,
            self.layer.post_attention_layernorm,
            incoming,
            getattr(self.layer.mlp, "gate_up_proj", None),
        )


def finalize_norm(
    norm: RMSNorm,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    *,
    incoming: Scatter,
) -> torch.Tensor:
    """Final RMSNorm before the LM head, owning the last decoder's deferred AR.

    ``incoming`` is the last decoder layer's declared ``output_scatter`` (PARTIAL
    when it defers, FULL otherwise), so the gate is a declared state rather than a
    ``fuse_allreduce`` flag check.
    """
    out, _ = fused_ar_rms_norm_quant(
        hidden,
        residual,
        norm,
        consumer_linear=None,
        do_allreduce=(
            incoming is Scatter.PARTIAL and get_tensor_model_parallel_world_size() > 1
        ),
    )
    return out
