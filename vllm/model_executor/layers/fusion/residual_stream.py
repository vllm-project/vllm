# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``ResidualStream``: the single manual-fusion entry point for a decoder layer.

The transformer residual stream is the running sum each sublayer reads from (via
RMSNorm) and writes back into. ``ResidualStream`` owns that read point: at each
sublayer boundary it performs the one transition the stream's distribution state
requires -- (all-reduce +) residual-add + RMSNorm + optional activation-quant --
and hands the (possibly pre-quantized) activation to the next consumer linear.

It receives the layer's norms and consumer linears at each :meth:`prepare`
call, so each model spells out which module is ``qkv_proj``/``gate_up_proj``/
... instead of this file guessing by name, and a layer may swap any of those
modules between forwards. Whether the layer defers its reduce is read from
the row-linear's existing ``reduce_results`` flag -- the single source of
truth -- so there is no separate fusion flag:
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
    """Per-forward state machine holding a decoder stack's residual stream.

    One instance is created at the model's forward entry (``residual=None``, or
    via :meth:`from_handoff` on a non-first pipeline-parallel rank) and threaded
    through every migrated decoder layer, which reads from and writes to it at
    each sublayer boundary:

    - read: :meth:`prepare` consumes the pending sublayer contribution and
      performs the transition the stream's distribution state requires --
      (all-reduce +) residual-add + RMSNorm (+ activation-quant) -- handing the
      activation to the next consumer linear.
    - write: :meth:`record` parks a sublayer output with its distribution tag,
      and is consumed by the next :meth:`prepare`, which installs the new
      residual.

    A layer tags what it produced: ``PARTIAL`` when its row-linear defers its
    reduce (see :meth:`defer_attn` / :meth:`defer_mlp`), ``FULL`` otherwise. A
    ``PARTIAL`` value owes exactly one all-reduce, which the next read performs;
    a ``FULL`` value is never reduced, so a double all-reduce is impossible by
    construction. After the last layer, :func:`finalize_norm` performs the final
    transition outside the stream.

    At most one contribution may be pending at a time, and the tensors passed to
    :meth:`prepare` / :meth:`snapshot` must be the ones the model is forwarding
    (verified by identity), because the fused kernel updates the pending value and
    the residual in place.

    Args:
        residual: Running residual sum, or ``None`` at the stack entry.
        vllm_config: Read for the LoRA setting, which keeps the activation
            unquantized so adapter patching stays valid.

    """

    def __init__(
        self,
        *,
        residual: torch.Tensor | None = None,
        vllm_config: VllmConfig,
    ) -> None:
        self.residual = residual
        self.pending: tuple[torch.Tensor, Scatter] | None = None
        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_lora_enable = vllm_config.lora_config is not None

    @classmethod
    def from_handoff(
        cls, value, residual, incoming: Scatter, vllm_config: VllmConfig
    ) -> "ResidualStream":
        stream = cls.__new__(cls)
        stream.residual = residual
        stream.pending = (value, incoming)
        stream.tp_size = get_tensor_model_parallel_world_size()
        stream.is_lora_enable = vllm_config.lora_config is not None
        return stream

    def defer_attn(self, o_proj: torch.nn.Module) -> bool:
        # Single source of truth: the attention o_proj's reduce_results. A layer
        # whose row-linears skip their reduce leaves PARTIAL output for us.
        return not getattr(o_proj, "reduce_results", True)

    def defer_mlp(self, down_proj: torch.nn.Module) -> bool:
        # Single source of truth: the mlp down_proj's reduce_results. A layer
        # whose row-linears skip their reduce leaves PARTIAL output for us.
        # TODO: NOTE maybe we can run ar fusion before mlp when pp is used
        return not getattr(down_proj, "reduce_results", True)

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

    def prepare(
        self,
        hidden: torch.Tensor,
        norm: torch.nn.Module,
        consumer_linear: torch.nn.Module | None,
    ) -> NormOut:
        # input — read pending + residual, no side effects
        (value, incoming), residual = self.input(hidden)
        # entry.prepare — the fused transition: (AR +) residual-add + RMSNorm (+ quant)
        out, new_residual = self._reduce_norm(
            value, residual, norm, incoming, consumer_linear
        )
        # write — consume pending, install the new residual
        self._write(new_residual)
        return out, new_residual

    def record(self, output: torch.Tensor, incoming: Scatter) -> None:
        """Producer side: park a sublayer output with its distribution state."""
        if self.pending is not None:
            raise RuntimeError("cannot replace an unconsumed producer contribution")
        self.pending = (output, incoming)

    def check(self, hidden: torch.Tensor) -> None:
        """The pending value must be the tensor the model is forwarding."""
        if self.pending is None:
            if self.residual is not None and hidden is not self.residual:
                raise RuntimeError("hidden does not match the written residual")
        else:
            if hidden is not self.pending[0]:
                raise RuntimeError(
                    "output does not belong to this residual stream; "
                    "change layer outputs through the boundary accessors"
                )

    def input(
        self, hidden: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, Scatter], torch.Tensor | None]:
        """READ ONLY: pending (value, tag) + running residual. Does not consume."""
        self.check(hidden)
        if self.pending is not None:
            return self.pending, self.residual
        # Layer 0 / PP entry: raw tensor, nothing owed, no residual yet.
        return (hidden, Scatter.FULL), None

    def _write(self, residual: torch.Tensor) -> None:
        """CONSUME: install the new residual; release the pending contribution."""
        self.pending = None
        self.residual = residual

    def snapshot(self, hidden: torch.Tensor) -> torch.Tensor:
        """Materialize the current sum WITHOUT consuming the pending.

        For aux hidden-state capture (Eagle3/DFLASH): the captured value must be
        the complete hidden state, but the pending contribution may still owe its
        all-reduce to the next boundary. Complete the owed AR on a clone, so the
        real consumer still sees the partial and reduces it exactly once.
        """
        (value, incoming), residual = self.input(hidden)
        if incoming is Scatter.PARTIAL and self.tp_size > 1:
            # Clone: the all-reduce may run in place, and the pending value the
            # next boundary consumes must stay partial (a second AR would scale
            # it by tp_size).
            value = tensor_model_parallel_all_reduce(value.clone())
        return value if residual is None else value + residual


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
