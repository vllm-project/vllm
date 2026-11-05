# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-phase activation formats for linear layers (``QuantSpec.phase``).

A kernel serving several activation formats from one weight copy implements
``DualActivationKernel``. The kernel oracle sets its ``phase``, and the kernel
then picks the format per call from the input row count. Under torch.compile
the choice runs inside the ``vllm::phase_split_linear`` op: vLLM compiles with
dynamic shapes, so a Python branch on the row count in traced code would be
fixed for every size.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.quantization import PhaseSpec
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4DynamicTensor,
)
from vllm.platforms import current_platform
from vllm.utils import torch_utils
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

ActivationMode = QuantKey | None
"""An activation format a kernel computes with; None is unquantized."""

_KernelT = TypeVar("_KernelT")
_PlanT = TypeVar("_PlanT")

_OPTION = "quantization_config.linear.phase"

# Phase-split layers by name, for vllm::phase_split_linear. The name reaches
# the op as a hoisted LayerName graph input, like the layer names of vLLM's
# attention and MoE ops, so a compiled region that vLLM reuses for several
# layers (e.g. multimodal encoder blocks) resolves each one. Unlike the forward
# context, the registry also covers graphs that run outside a forward pass.
_LAYERS: weakref.WeakValueDictionary[str, torch.nn.Module] = (
    weakref.WeakValueDictionary()
)


def _register_layer(layer: torch.nn.Module) -> str:
    """Name ``layer`` uniquely among live phase-split layers: its prefix, or
    its class name without one, plus the first free ``#n`` suffix when another
    live layer holds that name."""
    name = layer.__dict__.get("phase_split_layer_name")
    if name is not None and _LAYERS.get(name) is layer:
        return name
    base = getattr(layer, "prefix", "") or type(layer).__name__
    name, n = base, 0
    while _LAYERS.get(name, layer) is not layer:
        n += 1
        name = f"{base}#{n}"
    _LAYERS[name] = layer
    return name


def has_phase_split_layers() -> bool:
    """Whether any live layer is served by a phase split."""
    return len(_LAYERS) > 0


class DualActivationKernel:
    """Mixin for linear kernels that serve every format in
    ``activation_modes()`` from one weight copy.

    The kernel oracle sets ``phase`` before ``process_weights_after_loading``,
    which prepares each phase's shapes and calls ``register_phase_layer``.
    With a phase, ``apply_weights`` runs ``apply_phase``.
    """

    phase: PhaseSpec | None = None

    @classmethod
    def activation_modes(cls) -> frozenset[ActivationMode]:
        raise NotImplementedError

    def apply_mode(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        mode: ActivationMode,
    ) -> torch.Tensor:
        """Run the GEMM with activations in ``mode``."""
        raise NotImplementedError

    def phase_mode(self, rows: int) -> ActivationMode:
        """Activations of a call with ``rows`` rows: ``phase.decode`` up to
        ``phase.max_decode_tokens``, or for every call when it is unset. Under
        CUDA graphs the rows are the padded capture size."""
        assert self.phase is not None
        decode, prefill = self.phase.activation_modes()
        cutoff = self.phase.max_decode_tokens
        return decode if cutoff is None or rows <= cutoff else prefill

    def register_phase_layer(self, layer: torch.nn.Module) -> None:
        """Let ``vllm::phase_split_linear`` find ``layer``."""
        layer.phase_split_kernel = self
        layer.phase_split_layer_name = _register_layer(layer)

    def apply_phase(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``apply_weights`` with a phase."""
        out_features = layer.output_size_per_partition
        x_2d = x.reshape(-1, x.shape[-1])
        if torch.compiler.is_compiling():
            output = torch.ops.vllm.phase_split_linear(
                x_2d,
                bias,
                _encode_layer_name(layer.phase_split_layer_name),
                out_features,
            )
        else:
            # Eager calls, including ones outside a forward pass such as MLA
            # weight dequantization, pick the phase here.
            mode = self.phase_mode(x_2d.shape[0])
            output = self.apply_mode(layer, x_2d, bias, mode=mode)
        return output.view(*x.shape[:-1], out_features)


@dataclass(frozen=True)
class PhasePlans(Generic[_PlanT]):
    """A kernel's prepared plans for one weight under a phase.

    ``modes`` are the (decode, prefill) activations the plans serve; a phase
    that quantizes can still get None for a layer the kernel cannot quantize.
    Calls with at most ``cutoff`` rows, or every call when it is None, are
    decode calls. ``capacity`` is the largest planned call, and ``key`` says
    what the plans were prepared for, so a reload with an equal key keeps them.
    """

    key: object
    capacity: int
    modes: tuple[ActivationMode, ActivationMode]
    cutoff: int | None
    plan: _PlanT
    # For kernels whose plan cannot serve quantized decode calls together with
    # unquantized prefill calls.
    decode_plan: _PlanT | None = None

    @property
    def plans(self) -> tuple[_PlanT, ...]:
        return (
            (self.plan,) if self.decode_plan is None else (self.plan, self.decode_plan)
        )

    @property
    def quantized_mode(self) -> ActivationMode:
        """The quantized activations the plans serve, if any."""
        decode, prefill = self.modes
        return decode if decode is not None else prefill

    def select(self, rows: int, mode: ActivationMode) -> tuple[_PlanT, ActivationMode]:
        """The plan of a call with ``rows`` rows in ``mode``, and the
        activations it runs with: None where the plans do not quantize."""
        decode = self.cutoff is None or rows <= self.cutoff
        if self.modes[0 if decode else 1] is None:
            mode = None
        if mode is not None and self.decode_plan is not None:
            return self.decode_plan, mode
        return self.plan, mode


def _check_batch_invariance(phase: PhaseSpec) -> None:
    """Checked before kernel support: VLLM_BATCH_INVARIANT selects its own
    NVFP4 kernels, which would otherwise be reported as unable to serve the
    phase."""
    decode, prefill = phase.activation_modes()
    if envs.VLLM_BATCH_INVARIANT and (
        decode != prefill or kNvfp4DynamicTensor in (decode, prefill)
    ):
        raise ValueError(
            f"{_OPTION} makes outputs depend on the rows in a step and is not "
            "supported with VLLM_BATCH_INVARIANT"
        )


def _check_phase_split_config(phase: PhaseSpec) -> None:
    """NVFP4 activations share one scale per call, so uninitialized padding
    rows (CUDA graphs, data parallelism) must never reach it."""
    decode, prefill = phase.activation_modes()
    shares_scale = kNvfp4DynamicTensor in (decode, prefill)
    config = get_current_vllm_config()
    if not torch_utils._USE_LAYERNAME:
        # A plain-string layer name is baked into the graph, so a compiled
        # graph reused for other layers (encoder blocks, or another engine's
        # model loaded from the compile cache) would resolve the wrong ones.
        raise ValueError(
            f"{_OPTION} requires LayerName layer names "
            "(VLLM_USE_LAYERNAME=1, the default on torch >= 2.11)"
        )
    if not shares_scale:
        return
    compilation = config.compilation_config
    max_capture_size = max(
        [
            *(compilation.cudagraph_capture_sizes or []),
            *compilation.decoder_replay_cudagraph_capture_sizes,
        ],
        default=0,
    )
    if decode != prefill and prefill == kNvfp4DynamicTensor:
        assert phase.max_decode_tokens is not None
        if phase.max_decode_tokens < max_capture_size:
            raise ValueError(
                f"{_OPTION}.max_decode_tokens={phase.max_decode_tokens} is below "
                f"the largest CUDA graph capture size {max_capture_size}"
            )
    # NVFP4 decode calls run in CUDA graphs and take their scale from the rows
    # the runner marks as real; V2 only marks them with this flag.
    if decode == kNvfp4DynamicTensor and not envs.VLLM_MOE_SKIP_PADDING:
        raise ValueError(
            f"{_OPTION} with decode=nvfp4 requires VLLM_MOE_SKIP_PADDING=1, which "
            "marks the CUDA graph padding rows"
        )
    if (
        config.parallel_config.data_parallel_size > 1
        or config.parallel_config.enable_elastic_ep
        or config.compilation_config.cudagraph_mm_encoder
    ):
        raise ValueError(
            f"{_OPTION} does not support data parallelism or encoder CUDA graphs"
        )
    # One request's activations set the scale for every request in the step,
    # so clients must not supply them directly.
    model_config = config.model_config
    mm_config = getattr(model_config, "multimodal_config", None)
    if getattr(model_config, "enable_prompt_embeds", False) or getattr(
        mm_config, "enable_mm_embeds", False
    ):
        raise ValueError(
            f"{_OPTION} with NVFP4 does not support client-supplied prompt or "
            "multimodal embeddings"
        )


def padding_rows(num_rows: int) -> torch.Tensor | None:
    """Mask of the CUDA graph and data parallel padding rows of a call over
    the tokens of a step, which may hold uninitialized memory. None when the
    runner does not mark them, or the call has other rows. The mask is a
    persistent buffer, so CUDA graphs read the current step's values."""
    if not (envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available()):
        return None
    is_padding = get_forward_context().is_padding
    if is_padding is None or is_padding.shape[0] != num_rows:
        return None
    return is_padding


def _serves(kernel_cls: type, phase: PhaseSpec) -> bool:
    return (
        issubclass(kernel_cls, DualActivationKernel)
        and set(phase.activation_modes()) <= kernel_cls.activation_modes()
    )


def maybe_phase_split(
    kernel: _KernelT,
    phase: PhaseSpec | None,
    weight: str,
    candidates: Iterable[type] = (),
) -> _KernelT:
    """Set ``kernel``'s phase, or raise if it cannot serve both phases from
    one weight copy. ``candidates`` are named in the error when they can."""
    if phase is None:
        return kernel
    _check_batch_invariance(phase)
    if not (isinstance(kernel, DualActivationKernel) and _serves(type(kernel), phase)):
        capable = [cls.__name__ for cls in candidates if _serves(cls, phase)]
        selected = (
            " (selected by VLLM_BATCH_INVARIANT)" if envs.VLLM_BATCH_INVARIANT else ""
        )
        raise NotImplementedError(
            f"{_OPTION} ({phase}) with weight={weight} on "
            f"{current_platform.device_name}: {type(kernel).__name__}{selected} "
            "cannot serve it from one weight copy. Kernels that can, where supported "
            f"(--linear-backend b12x): {', '.join(capable) or 'none'}."
        )
    _check_phase_split_config(phase)
    kernel.phase = phase
    return kernel


def _phase_split_linear(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: LayerNameType,
    out_features: int,
) -> torch.Tensor:
    layer = _LAYERS[_resolve_layer_name(layer_name)]
    kernel = layer.phase_split_kernel
    output = kernel.apply_mode(layer, x, bias, mode=kernel.phase_mode(x.shape[0]))
    if output.shape != (x.shape[0], out_features):
        raise RuntimeError(
            f"{type(kernel).__name__} returned shape "
            f"{tuple(output.shape)}, expected {(x.shape[0], out_features)}"
        )
    return output


def _phase_split_linear_fake(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: LayerNameType,
    out_features: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], out_features))


direct_register_custom_op(
    op_name="phase_split_linear",
    op_func=_phase_split_linear,
    fake_impl=_phase_split_linear_fake,
)
