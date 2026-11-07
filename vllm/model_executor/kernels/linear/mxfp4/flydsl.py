# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    get_fp8_min_max,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kMxfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, WarmupChoices
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class _FlydslMxfp4Gemm(VllmJitKernel["_FlydslMxfp4Gemm.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        signature: tuple

    def dispatch(self, signature: tuple) -> CompileKey:  # type: ignore[override]
        return self.CompileKey(signature=signature)

    def get_warmup_keys(
        self,
        *,
        n: int,
        k: int,
        mode: str,
        normal_only: bool,
        out_dtype: torch.dtype,
        device_id: int,
        max_m: int,
        num_processors: int,
        cache_size_bytes: int,
        lds_size_bytes: int,
    ) -> list[CompileKey]:
        from flydsl.kernels.gemm.rdna4_mxfp4 import get_warmup_keys

        signatures = get_warmup_keys(
            n,
            k,
            max_m,
            mode=mode,
            normal_only=normal_only,
            dtype=out_dtype,
            device_id=device_id,
            cache_policy="rotating",
            num_processors=num_processors,
            cache_size_bytes=cache_size_bytes,
            lds_size_bytes=lds_size_bytes,
        )
        return self._trace_dispatch(self.dispatch)(signature=WarmupChoices(*signatures))

    def compile(self, compile_key: CompileKey) -> None:
        from flydsl.kernels.gemm.rdna4_mxfp4 import compile_gemm

        compile_gemm(compile_key.signature)

    def __call__(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        x_scale: torch.Tensor,
        base_exponents: torch.Tensor | None,
        table: torch.Tensor | None,
        n: int,
        k: int,
        mode: str,
        normal_only: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        from flydsl.kernels.gemm.rdna4_mxfp4 import PreparedMXFP4, gemm

        if x.shape[0] == 0:
            return torch.empty((0, n), device=x.device, dtype=out_dtype)
        prepared = PreparedMXFP4(
            packed,
            scales,
            n,
            k,
            mode,
            base_exponents,
            table,
            normal_only=normal_only,
            nibble_layout="split",
        )
        # Full-model execution visits different layers between uses of a weight.
        return gemm(x, prepared, x_scale, dtype=out_dtype, cache_policy="rotating")


_FLYDSL_MXFP4_GEMM = _FlydslMxfp4Gemm()


def _flydsl_mxfp4_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    x_scale: torch.Tensor,
    base_exponents: torch.Tensor | None,
    table: torch.Tensor | None,
    n: int,
    k: int,
    mode: str,
    normal_only: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return _FLYDSL_MXFP4_GEMM(
        x,
        packed,
        scales,
        x_scale,
        base_exponents,
        table,
        n,
        k,
        mode,
        normal_only,
        out_dtype,
    )


def _flydsl_mxfp4_gemm_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    x_scale: torch.Tensor,
    base_exponents: torch.Tensor | None,
    table: torch.Tensor | None,
    n: int,
    k: int,
    mode: str,
    normal_only: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty((x.shape[0], n), device=x.device, dtype=out_dtype)


direct_register_custom_op(
    op_name="flydsl_mxfp4_gemm",
    op_func=_flydsl_mxfp4_gemm,
    mutates_args=[],
    fake_impl=_flydsl_mxfp4_gemm_fake,
)


class Rdna4Mxfp4LinearKernel(MxFp4LinearKernel):
    """Lossless MXFP4 weights and FP8 activations on ROCm RDNA4."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "FlyDSL MXFP4 requires ROCm RDNA4"
        from vllm.platforms.rocm import on_rdna4

        if not on_rdna4():
            return False, "FlyDSL MXFP4 requires gfx1200/gfx1201"
        try:
            available = (
                importlib.util.find_spec("flydsl.kernels.gemm.rdna4_mxfp4") is not None
            )
        except (ImportError, ModuleNotFoundError):
            available = False
        if not available:
            return False, "FlyDSL with the RDNA4 MXFP4 kernels is not installed"
        return True, None

    @classmethod
    def _supports_quant_scheme(
        cls, weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return weight_key == kMxfp4Static and activation_key in (
            kFp8StaticTensorSym,
            kFp8DynamicTokenSym,
        )

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key not in (
            kFp8StaticTensorSym,
            kFp8DynamicTokenSym,
        ):
            return (
                False,
                "FlyDSL MXFP4 requires static tensor or dynamic token FP8 activations",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flydsl.kernels.gemm.rdna4_mxfp4 import (
            get_device_capabilities,
            prepare_weights,
        )

        prepared = prepare_weights(layer.weight, layer.weight_scale)
        replace_parameter(layer, "weight", prepared.packed)
        replace_parameter(layer, "weight_scale", prepared.scales)
        layer.register_buffer(
            "_flydsl_mxfp4_refs", prepared.base_exponents, persistent=False
        )
        layer.register_buffer("_flydsl_mxfp4_table", prepared.table, persistent=False)
        layer._flydsl_mxfp4_n = prepared.n
        layer._flydsl_mxfp4_k = prepared.k
        layer._flydsl_mxfp4_mode = prepared.mode
        layer._flydsl_mxfp4_normal_only = prepared.normal_only

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            params_dtype = getattr(layer, "params_dtype", None)
            output_dtypes = (
                (params_dtype,)
                if params_dtype is not None
                else (torch.bfloat16, torch.float16)
            )
            for out_dtype in output_dtypes:
                _FLYDSL_MXFP4_GEMM.register_warmup(
                    n=prepared.n,
                    k=prepared.k,
                    mode=prepared.mode,
                    normal_only=prepared.normal_only,
                    out_dtype=out_dtype,
                    device_id=layer.weight.device.index,
                    max_m=vllm_config.scheduler_config.max_num_batched_tokens,
                    **get_device_capabilities(layer.weight.device.index),
                )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        if x_2d.shape[0] == 0:
            return torch.empty(
                (*shape[:-1], layer._flydsl_mxfp4_n), device=x.device, dtype=x.dtype
            )
        if self.config.activation_quant_key == kFp8DynamicTokenSym:
            from vllm import _custom_ops as ops

            x_fp8, x_scale = ops.scaled_fp8_quant(
                x_2d.contiguous(), use_per_token_if_dynamic=True
            )
            x_scale = x_scale.view(-1)
        else:
            fp8_min, fp8_max = get_fp8_min_max()
            x_fp8 = (
                (x_2d / layer.input_scale)
                .clamp(fp8_min, fp8_max)
                .to(torch.float8_e4m3fn)
                .contiguous()
            )
            x_scale = (
                layer.input_scale.reshape(())
                .expand(x_2d.shape[0])
                .to(device=x.device, dtype=torch.float32)
                .contiguous()
            )
        y = torch.ops.vllm.flydsl_mxfp4_gemm(
            x_fp8,
            layer.weight,
            layer.weight_scale,
            x_scale,
            layer._flydsl_mxfp4_refs,
            layer._flydsl_mxfp4_table,
            layer._flydsl_mxfp4_n,
            layer._flydsl_mxfp4_k,
            layer._flydsl_mxfp4_mode,
            layer._flydsl_mxfp4_normal_only,
            x.dtype,
        )
        if bias is not None:
            y = y + bias
        return y.reshape(*shape[:-1], layer._flydsl_mxfp4_n)
