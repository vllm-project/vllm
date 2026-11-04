# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.model_executor.kernels.linear.phase_split import (
    ActivationMode,
    DualActivationKernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xWarmupUnit,
    plan_b12x_linear,
    reuse_packed_weight_storage,
)
from vllm.utils.b12x import (
    get_b12x_blockscaled as _import_b12x_blockscaled,
)
from vllm.utils.b12x import (
    get_b12x_mxfp8_linear as _import_b12x_mxfp8,
)
from vllm.utils.torch_utils import current_stream

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def _apply_b12x_mxfp8_packed_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    packed_weight = layer.b12x_mxfp8_packed_weight

    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output_shape = [*x.shape[:-1], int(packed_weight.out_features)]

    mxfp8 = _import_b12x_mxfp8()
    assert mxfp8 is not None
    output = mxfp8.mm(
        input_2d,
        packed_weight,
        bias=bias,
        expected_m=max(1, int(input_2d.shape[0])),
    )
    return output.view(*output_shape)


class B12xMxfp8LinearKernel(Mxfp8LinearKernel, DualActivationKernel):
    """ModelOpt MXFP8 linear through the native b12x SM120 dense GEMM path.

    b12x picks BF16 or MXFP8 activations per call by row count. With a phase
    split, each phase's calls use its activations instead."""

    @classmethod
    def activation_modes(cls) -> frozenset[ActivationMode]:
        return frozenset((None, kMxfp8Dynamic))

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x MXFP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x MXFP8 kernels require a Blackwell 12x device"
        mxfp8 = _import_b12x_mxfp8()
        if mxfp8 is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not mxfp8.is_supported():
            return False, "b12x.gemm.mxfp8_linear is not supported"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        del c
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        assert weight.dtype == MXFP8_VALUE_DTYPE, (
            f"b12x MXFP8 requires {MXFP8_VALUE_DTYPE}, got {weight.dtype}"
        )
        assert weight.ndim == 2, f"b12x MXFP8 weight must be 2D, got {weight.ndim}D"
        assert hasattr(layer, "weight_scale"), "b12x MXFP8 linear requires weight_scale"

        out_features, in_features = map(int, weight.shape)
        assert in_features % MXFP8_BLOCK_SIZE == 0, (
            "b12x MXFP8 requires input features divisible by "
            f"{MXFP8_BLOCK_SIZE}, got {in_features}"
        )
        weight_scale = layer.weight_scale.data
        assert weight_scale.dtype == MXFP8_SCALE_DTYPE, (
            f"b12x MXFP8 requires {MXFP8_SCALE_DTYPE} weight_scale, "
            f"got {weight_scale.dtype}"
        )
        assert weight_scale.ndim == 2, (
            f"b12x MXFP8 weight_scale must be 2D, got {weight_scale.ndim}D"
        )

        mxfp8 = _import_b12x_mxfp8()
        assert mxfp8 is not None
        scale_k = in_features // MXFP8_BLOCK_SIZE
        packed_weight = mxfp8.pack_weight(
            weight[:out_features, :in_features].detach(),
            weight_scale[:out_features, :scale_k].detach(),
        )
        layer.b12x_mxfp8_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_mxfp8_packed_weight", None),
            packed_weight,
        )
        replace_parameter(layer, "weight", weight.new_empty((0,)))
        replace_parameter(layer, "weight_scale", weight_scale.new_empty((0,)))
        layer.b12x_warmup_provider = self
        if self.phase is not None:
            blockscaled = _import_b12x_blockscaled()
            assert blockscaled is not None
            layer.b12x_mxfp8_plans = plan_b12x_linear(
                blockscaled,
                layer.b12x_mxfp8_packed_weight,
                "mxfp8",
                self.phase,
                getattr(layer, "b12x_mxfp8_plans", None),
            )

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        if self.phase is not None:
            return self._get_phase_warmup_unit(layer, token_counts, output_dtype)
        packed_weight = layer.b12x_mxfp8_packed_weight
        device = torch.device(packed_weight.weight.values.device)

        def compile() -> None:
            mxfp8 = _import_b12x_mxfp8()
            assert mxfp8 is not None
            for tokens in token_counts:
                source = torch.zeros(
                    (tokens, int(packed_weight.in_features)),
                    dtype=output_dtype,
                    device=device,
                )
                mxfp8.mm(
                    source,
                    packed_weight,
                    expected_m=max(1, int(tokens)),
                    stream=current_stream().cuda_stream,
                )

        return B12xWarmupUnit(
            name="MXFP8",
            key=(
                type(self),
                device,
                int(packed_weight.in_features),
                int(packed_weight.padded_in_features),
                int(packed_weight.out_features),
                output_dtype,
            ),
            compile=compile,
        )

    def _get_phase_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        plans = layer.b12x_mxfp8_plans
        mode = kMxfp8Dynamic if any(plans.quantized) else None
        planned = {m for plan in plans.plans for m in plan.token_counts}
        packed_weight = layer.b12x_mxfp8_packed_weight

        def compile() -> None:
            for tokens in sorted(set(token_counts) | planned):
                source = torch.zeros(
                    (tokens, int(packed_weight.in_features)),
                    dtype=output_dtype,
                    device=packed_weight.weight.values.device,
                )
                self.apply_weights(layer, source, mode=mode)

        return B12xWarmupUnit(
            name="MXFP8",
            # Every layer owns a plan that must be prepared before capture.
            key=(type(self), *(plan.handle for plan in plans.plans), output_dtype),
            compile=compile,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        mode: ActivationMode = None,
    ) -> torch.Tensor:
        if self.phase is None:
            return _apply_b12x_mxfp8_packed_linear(layer, x, bias)
        if mode is not None and mode != kMxfp8Dynamic:
            raise ValueError(f"b12x MXFP8 does not support activations {mode}")
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        # Rows beyond the planned capacity (e.g. an encoder batch) run in
        # capacity-sized chunks; an unplanned call would prepare on first use.
        capacity = layer.b12x_mxfp8_plans.query.num_tokens
        quantize = mode is not None
        if x_2d.shape[0] <= capacity:
            output = self._mm(layer, x_2d, quantize)
        else:
            output = torch.cat(
                [self._mm(layer, c, quantize) for c in x_2d.split(capacity)]
            )
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], output.shape[-1])

    @staticmethod
    def _mm(layer: torch.nn.Module, x: torch.Tensor, quantize: bool) -> torch.Tensor:
        mxfp8 = _import_b12x_mxfp8()
        assert mxfp8 is not None
        plan, quantized = layer.b12x_mxfp8_plans.select(x.shape[0], quantize)
        return mxfp8.mm(
            x,
            layer.b12x_mxfp8_packed_weight,
            plan=plan,
            required_mode="quantized" if quantized else "a16",
        )
