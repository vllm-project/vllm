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

It receives the layer's norms and consumer linears directly at construction, so
each model spells out which module is ``qkv_proj``/``gate_up_proj``/... instead of
this file guessing by name. A layer that swaps one of those modules *after*
building its stream (e.g. eagle replacing its layer-0 norm with ``nn.Identity()``,
aria swapping in an MoE ``mlp``) must rebuild the stream so it captures the
replacement. Whether the layer defers its reduce is read from the row-linear's
existing ``reduce_results`` flag -- the single source of truth -- so there is no
separate fusion flag:
- linears defer (``reduce_results=False``): inputs arrive ``PARTIAL``; the stream
  reduces them once while normalizing.
- linears reduce themselves (``reduce_results=True``): inputs arrive ``FULL``; the
  stream only normalizes. This is how MoE / unfused reusers share the same forward
  with no special-casing.
A second all-reduce is impossible by construction (``FULL`` never reduces), and a
non-RMSNorm ``norm`` (eagle Identity) is tolerated.

Scope: plain all-reduce only. Sequence-parallelism (a future token-shard state
with reduce-scatter/all-gather transitions) is out of scope.
"""

from enum import Enum, auto

import torch

from vllm.config import VllmConfig
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


class ResidualStream:
    """Owns (AR +) residual-add + RMSNorm(+quant) for one decoder layer.

    Receives the layer's norms and consumer linears directly, so model
    ``forward`` is just ``prepare_attn`` -> attn -> ``prepare_mlp`` -> mlp, with
    no ``tp_size``/``do_allreduce``/fuse branch. The modules are captured at
    construction: a layer that swaps one *after* building its stream (eagle's
    Identity norm, aria's MoE ``mlp``) must rebuild the stream.

    ``qkv_proj``/``gate_up_proj`` may be None (e.g. an MoE ``mlp`` has no
    ``gate_up_proj``); the consumer-linear quant fusion is then simply skipped.
    The linears are captured before any LoRA wrapping, so they are already the
    base layers -- no unwrapping needed.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        qkv_proj: torch.nn.Module | None,
        o_proj: torch.nn.Module | None,
        gate_up_proj: torch.nn.Module | None,
        down_proj: torch.nn.Module | None,
    ) -> None:
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_lora_enable = vllm_config.lora_config is not None
        self.is_pp_enable = vllm_config.parallel_config.pipeline_parallel_size > 1

    @property
    def defer_attn(self) -> bool:
        # Single source of truth: the attention o_proj's reduce_results. A layer
        # whose row-linears skip their reduce leaves PARTIAL output for us.
        return (
            not getattr(self.o_proj, "reduce_results", True) and not self.is_pp_enable
        )

    @property
    def defer_mlp(self) -> bool:
        # Single source of truth: the mlp down_proj's reduce_results. A layer
        # whose row-linears skip their reduce leaves PARTIAL output for us.
        # NOTE maybe we can run ar fusion before mlp when pp is used
        return (
            not getattr(self.down_proj, "reduce_results", True)
            and not self.is_pp_enable
        )

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
        # An SP build would add a token-shard branch here for a
        # reduce-scatter+norm (and all-gather) transition instead.
        quant_consumer_linear = None if self.is_lora_enable else consumer_linear
        if type(norm) is RMSNorm:
            return fused_ar_rms_norm_quant(
                hidden,
                residual,
                norm,
                consumer_linear=quant_consumer_linear,
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
            Scatter.PARTIAL
            if (self.defer_attn and residual is not None)
            else Scatter.FULL
        )
        return self._reduce_norm(
            hidden,
            residual,
            self.input_layernorm,
            incoming,
            self.qkv_proj,
        )

    def prepare_mlp(self, hidden: torch.Tensor, residual: torch.Tensor) -> NormOut:
        # Attention o_proj output is PARTIAL iff it ran with reduce_results=False.
        incoming = Scatter.PARTIAL if self.defer_mlp else Scatter.FULL
        return self._reduce_norm(
            hidden,
            residual,
            self.post_attention_layernorm,
            incoming,
            self.gate_up_proj,
        )


def finalize_norm(
    norm: torch.nn.Module,
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
    do_allreduce = (
        incoming is Scatter.PARTIAL and get_tensor_model_parallel_world_size() > 1
    )
    # Mirror ``_reduce_norm``: only a plain RMSNorm can go through the fused
    # kernel (which asserts ``type(norm) is RMSNorm``). A variant final norm
    # (e.g. GemmaRMSNorm) reduces if owed, then runs eager -- so finalize_norm
    # is safe for any model, not just RMSNorm ones.
    if type(norm) is RMSNorm:
        out, _ = fused_ar_rms_norm_quant(
            hidden, residual, norm, consumer_linear=None, do_allreduce=do_allreduce
        )
        return out
    if do_allreduce:
        hidden = tensor_model_parallel_all_reduce(hidden)
    out, _ = norm(hidden, residual)
    return out
