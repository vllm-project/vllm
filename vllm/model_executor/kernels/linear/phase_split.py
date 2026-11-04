# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-phase activation formats for linear layers (``QuantSpec.phase``).

A kernel serving several activation formats from one weight copy implements
``DualActivationKernel``. ``PhaseSplitLinearKernel`` wraps it and picks the
format per call from the input row count. Under torch.compile the choice runs
inside the ``vllm::phase_split_linear`` op: vLLM compiles with dynamic shapes,
so a Python branch on the row count in traced code would be fixed for every
size.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from typing import TypeVar

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

    ``phase`` is set before ``process_weights_after_loading`` so the kernel can
    prepare each phase's shapes.
    """

    phase: PhaseSpec | None = None

    @classmethod
    def activation_modes(cls) -> frozenset[ActivationMode]:
        raise NotImplementedError

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        mode: ActivationMode = None,
    ) -> torch.Tensor:
        """Run the GEMM with activations in ``mode``."""
        raise NotImplementedError


def _check_phase_split_config(phase: PhaseSpec, option: str) -> None:
    """NVFP4 activations share one scale per call, so uninitialized padding
    rows (CUDA graphs, data parallelism) must never reach it. ``option`` names
    the row-count setting the user set, for errors."""
    decode, prefill = phase.activation_modes()
    name = option.removesuffix(".max_decode_tokens")
    shares_scale = kNvfp4DynamicTensor in (decode, prefill)
    if envs.VLLM_BATCH_INVARIANT and (decode != prefill or shares_scale):
        raise ValueError(
            f"{name} makes outputs depend on the rows in a step and is not "
            "supported with VLLM_BATCH_INVARIANT"
        )
    config = get_current_vllm_config()
    if not torch_utils._USE_LAYERNAME:
        # A plain-string layer name is baked into the graph, so a compiled
        # graph reused for other layers (encoder blocks, or another engine's
        # model loaded from the compile cache) would resolve the wrong ones.
        raise ValueError(
            f"{name} requires LayerName layer names "
            "(VLLM_USE_LAYERNAME=1, the default on torch >= 2.11)"
        )
    if not shares_scale:
        return
    max_capture_size = max(config.compilation_config.cudagraph_capture_sizes or [0])
    if decode != prefill and prefill == kNvfp4DynamicTensor:
        assert phase.max_decode_tokens is not None
        if phase.max_decode_tokens < max_capture_size:
            raise ValueError(
                f"{option}={phase.max_decode_tokens} is below the largest CUDA "
                f"graph capture size {max_capture_size}"
            )
    # NVFP4 decode calls run in CUDA graphs and take their scale from the rows
    # the runner marks as real; V2 only marks them with this flag.
    if decode == kNvfp4DynamicTensor and not envs.VLLM_MOE_SKIP_PADDING:
        raise ValueError(
            f"{name} with decode=nvfp4 requires VLLM_MOE_SKIP_PADDING=1, which "
            "marks the CUDA graph padding rows"
        )
    if (
        config.parallel_config.data_parallel_size > 1
        or config.parallel_config.enable_elastic_ep
        or config.compilation_config.cudagraph_mm_encoder
    ):
        raise ValueError(
            f"{name} does not support data parallelism or encoder CUDA graphs"
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


class PhaseSplitLinearKernel:
    """Serves ``phase.decode`` and ``phase.prefill`` from one
    ``DualActivationKernel``.

    A call with at most ``phase.max_decode_tokens`` rows, or every call when it
    is unset, uses the decode format. Under CUDA graphs the rows are the padded
    capture size.
    """

    # Loading also prepares each phase's shapes and registers the layer.
    supports_pre_processed_weights = False

    def __init__(self, kernel: DualActivationKernel, phase: PhaseSpec) -> None:
        self.kernel = kernel
        self.phase = phase
        self.max_decode_tokens = phase.max_decode_tokens
        self.decode_mode, self.prefill_mode = phase.activation_modes()
        kernel.phase = phase

    def input_quant_key(self) -> QuantKey | None:
        # Each phase quantizes its own input.
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)  # type: ignore[attr-defined]
        layer.phase_split_kernel = self
        layer.phase_split_layer_name = _register_layer(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            output = self.forward(layer, x_2d, bias)
        return output.view(*x.shape[:-1], out_features)

    def forward(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Eager body of ``vllm::phase_split_linear``."""
        if self.max_decode_tokens is None or x.shape[0] <= self.max_decode_tokens:
            mode = self.decode_mode
        else:
            mode = self.prefill_mode
        return self.kernel.apply_weights(layer, x, bias, mode=mode)


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
    option: str = "quantization_config.linear.phase.max_decode_tokens",
) -> _KernelT | PhaseSplitLinearKernel:
    """Wrap ``kernel`` for ``phase``, or raise if it cannot serve both phases
    from one weight copy. ``candidates`` are named in the error when they can;
    ``option`` names the row-count setting the user set."""
    if phase is None:
        return kernel
    _check_phase_split_config(phase, option)
    if isinstance(kernel, DualActivationKernel) and _serves(type(kernel), phase):
        return PhaseSplitLinearKernel(kernel, phase)
    capable = [cls.__name__ for cls in candidates if _serves(cls, phase)]
    raise NotImplementedError(
        f"quantization_config.linear.phase ({phase}) with weight={weight} on "
        f"{current_platform.device_name}: {type(kernel).__name__} cannot serve "
        "it from one weight copy. Kernels that can, where supported "
        f"(--linear-backend b12x): {', '.join(capable) or 'none'}."
    )


def _phase_split_linear(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: LayerNameType,
    out_features: int,
) -> torch.Tensor:
    layer = _LAYERS[_resolve_layer_name(layer_name)]
    output = layer.phase_split_kernel.forward(layer, x, bias)
    assert output.shape == (x.shape[0], out_features), output.shape
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
