# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference (Stage 1) int8 quantization utilities for the Mamba/SSM
recurrent state cache.

Motivation
----------
vLLM already ships mature quantization for the *attention* KV cache
(FP8, TurboQuant Hadamard+Lloyd-Max, NVFP4, per-token-head scales - see
``vllm/utils/torch_utils.py``'s ``STR_DTYPE_TO_TORCH_DTYPE`` and the
``fp8``/``turboquant_*``/``nvfp4`` entries there) because that cache is
read and rewritten every decode step and is memory-bandwidth bound.

The Mamba/SSM recurrent state (conv state + temporal/SSM state used by
Mamba1, Mamba2, GatedDeltaNet, KDA, ShortConv and the hybrid models that
wrap them - Jamba, NemotronH, Zamba2, Qwen3.5, FalconH1, etc.) has the
exact same profile: every decode step, every layer, the state is read,
updated via the selective-scan recurrence, and rewritten. Today it can
only be stored as a floating dtype - see ``MambaDType`` in
``vllm/config/cache.py`` (``Literal["auto", "float32", "float16",
"bfloat16"]``) and ``MambaStateDtypeCalculator._mamba_state_dtype`` in
``vllm/model_executor/layers/mamba/mamba_utils.py``, which resolves the
temporal state dtype via a plain ``STR_DTYPE_TO_TORCH_DTYPE`` lookup
with no quantize/dequantize step anywhere around it. Interestingly,
``STR_DTYPE_TO_TORCH_DTYPE`` already contains ``"int8"`` and ``"fp8*"``
entries (reused from the KV-cache dtype strings) and
``MambaBase.bind_kv_cache`` (``vllm/model_executor/layers/mamba/
abstract.py``) already slices the raw per-block byte page and
reinterprets it per declared dtype/shape - so a smaller-dtype state
would already be sized and laid out correctly by the existing
allocator. What is missing is the actual quantize-before-write /
dequantize-before-compute step, since the selective-scan recurrence
itself must run in floating point.

Scope of this module (Stage 1)
-------------------------------
This module provides a small, self-contained, pure-PyTorch reference
implementation of that missing piece: per-channel dynamic int8
quantize/dequantize helpers for an SSM state tensor, plus a
``QuantizedSSMState`` wrapper that models the read/compute/write pattern
a real integration would use at every decode step. It is deliberately
**not** wired into ``MambaDType``, ``CacheConfig``, or any real Mamba
mixer/kernel in this change - doing that safely requires touching the
compiled CPU/Triton/CUDA selective-scan kernels and the per-architecture
mixer forward passes (``mamba_mixer.py``, ``mamba_mixer2.py``, the
``gdn``/``kda``/``short_conv`` variants), which cannot be responsibly
verified through source-browsing alone without a local test run.

Deferred follow-up work (not implemented here):
    * Add an ``"int8"`` (and/or ``"fp8"``) option to ``MambaDType`` and
      thread it through ``CacheConfig`` validation.
    * Call ``quantize_state_int8``/``dequantize_state_int8`` (or a fused
      kernel equivalent) around the temporal-state read/write in each
      mixer's ``forward()``, so storage stays int8 while the recurrence
      math still runs in float32/bf16.
    * Replace the pure-PyTorch quantizer with a fused Triton/CUDA kernel
      once the above is validated numerically on real models.

None of this is wired up yet; see the tests in
``tests/kernels/mamba/test_state_quant.py`` for what is verified today
(pure numerical round-trip and drift behaviour of the quantizer itself).
"""

from dataclasses import dataclass

import torch


def quantize_state_int8(
    state: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel dynamic symmetric int8 quantization of a float state.

    A single absmax-derived scale is shared across ``dim`` and computed
    independently for every other position in the tensor (e.g. for an
    SSM state of shape ``(num_heads, head_dim, state_size)`` with the
    default ``dim=-1``, every ``(head, channel)`` row gets its own scale
    computed over the ``state_size`` axis).

    Args:
        state: Floating-point state tensor to quantize.
        dim: Dimension the scale is shared across (collapsed to size 1
            in the returned scale tensor).
        eps: Minimum scale value, to avoid division by zero for
            all-zero rows.

    Returns:
        ``(qdata, scale)`` where ``qdata`` is ``torch.int8`` with the
        same shape as ``state`` and values in ``[-127, 127]``, and
        ``scale`` is ``torch.float32`` with the same shape as ``state``
        except size 1 along ``dim``. ``qdata.float() * scale``
        approximately reconstructs ``state``.
    """
    if state.numel() == 0:
        raise ValueError("Cannot quantize an empty state tensor.")

    state_f32 = state.to(torch.float32)
    amax = state_f32.abs().amax(dim=dim, keepdim=True)
    scale = (amax / 127.0).clamp_min(eps)
    qdata = torch.clamp(torch.round(state_f32 / scale), -127, 127).to(torch.int8)
    return qdata, scale


def dequantize_state_int8(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_state_int8`.

    Args:
        qdata: ``torch.int8`` tensor produced by
            :func:`quantize_state_int8`.
        scale: The matching per-channel ``float32`` scale tensor.

    Returns:
        A ``torch.float32`` tensor with the same shape as ``qdata``,
        reconstructing the original state up to int8 rounding error.
    """
    if qdata.dtype != torch.int8:
        raise ValueError(f"Expected torch.int8 qdata, got {qdata.dtype}.")
    return qdata.to(torch.float32) * scale


@dataclass
class QuantizedSSMState:
    """A quantized-at-rest SSM/Mamba state: int8 data plus its scale.

    This models what a real integration would keep resident in the
    per-layer state cache instead of a full-precision float tensor.
    """

    qdata: torch.Tensor
    scale: torch.Tensor
    dim: int = -1

    @classmethod
    def from_float(
        cls,
        state: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8,
    ) -> "QuantizedSSMState":
        """Quantize a floating-point state into a new instance."""
        qdata, scale = quantize_state_int8(state, dim=dim, eps=eps)
        return cls(qdata=qdata, scale=scale, dim=dim)

    def to_float(self) -> torch.Tensor:
        """Dequantize back to a ``torch.float32`` tensor."""
        return dequantize_state_int8(self.qdata, self.scale)

    def update_(self, new_state: torch.Tensor, eps: float = 1e-8) -> None:
        """Re-quantize ``new_state`` and replace this instance's data.

        Models the "read (dequantize) -> compute one recurrence step in
        float -> write back (quantize)" pattern a real per-step
        integration would need; see
        :func:`simulate_quantized_recurrence_step` for a worked example
        of the full step.
        """
        qdata, scale = quantize_state_int8(new_state, dim=self.dim, eps=eps)
        self.qdata = qdata
        self.scale = scale

    @property
    def shape(self) -> torch.Size:
        return self.qdata.shape

    def memory_bytes(self) -> int:
        """Total bytes used by the int8 data plus its float32 scale.

        Useful for sanity-checking the memory savings this is meant to
        provide versus an equivalent float16/float32 state tensor.
        """
        return (
            self.qdata.numel() * self.qdata.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


def simulate_quantized_recurrence_step(
    state_q: QuantizedSSMState,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
) -> QuantizedSSMState:
    """Reference example of one diagonal SSM recurrence step against a
    quantized-at-rest state.

    Computes ``h_t = a * h_{t-1} + b * x_t`` by dequantizing
    ``state_q`` to float, doing the step in float32, and re-quantizing
    the result. This is a simplified stand-in for the real selective-scan
    recurrence (which additionally involves a per-step discretization of
    ``a``/``b`` and, for Mamba2/GDN, a low-rank ``B``/``C`` structure),
    used here only to measure how much numerical drift accumulates if a
    state is quantized after every single step.

    Args:
        state_q: The previous quantized state, ``h_{t-1}``.
        a: Per-step decay term, same shape as the dequantized state.
        b: Per-step input gate term, same shape as the dequantized state.
        x: Per-step input, same shape as the dequantized state.

    Returns:
        A new :class:`QuantizedSSMState` holding ``h_t``.
    """
    h_prev = state_q.to_float()
    h_next = a * h_prev + b * x
    return QuantizedSSMState.from_float(h_next, dim=state_q.dim)
