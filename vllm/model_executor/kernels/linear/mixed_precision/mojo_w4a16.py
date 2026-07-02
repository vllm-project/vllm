# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe Mojo W4A16 GEMM integration for ROCm.

The vLLM layer owns torch custom-op registration and AWQ linear plumbing. The
portable Mojo runtime beside the kernels owns policy selection, generated
extension builds, runner caching, scratch preparation, and HIP-stream launches.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .triton_w4a16 import (
    TRITON_W4A16_SUPPORTED_GROUP_SIZES,
    TRITON_W4A16_SUPPORTED_QUANT_TYPES,
    TritonW4A16LinearKernel,
)

_RUNTIME_PATH = (
    Path(__file__).resolve().parents[5]
    / "csrc"
    / "rocm"
    / "mojo_gemm_w4a16"
    / "runtime.py"
)


@lru_cache(maxsize=1)
def _load_mojo_runtime() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "vllm_mojo_gemm_w4a16_runtime", _RUNTIME_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Mojo W4A16 runtime from {_RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mojo_runtime = _load_mojo_runtime()

MojoRunConfig = _mojo_runtime.MojoRunConfig

_is_compiling = _mojo_runtime._is_compiling
_debug_log = _mojo_runtime._debug_log
_stream_debug = _mojo_runtime._stream_debug
_validate_inputs = _mojo_runtime._validate_inputs
_select_config_for_tensors = _mojo_runtime.select_config_for_tensors
_get_qweight_kpacked = _mojo_runtime.get_qweight_kpacked
_make_qweight_kpacked = _mojo_runtime.make_qweight_kpacked
_cache_prepared_qweight_kpacked = _mojo_runtime.cache_prepared_qweight_kpacked
prepare_mojo_w4a16_gemm = _mojo_runtime.prepare_mojo_w4a16_gemm
_mojo_w4a16_gemm_out_impl = _mojo_runtime.mojo_w4a16_gemm_out_impl
_mojo_w4a16_gemm_out_fake = _mojo_runtime.mojo_w4a16_gemm_out_fake
_mojo_w4a16_gemm_impl = _mojo_runtime.mojo_w4a16_gemm_impl
_mojo_w4a16_gemm_fake = _mojo_runtime.mojo_w4a16_gemm_fake


direct_register_custom_op(
    op_name="mojo_w4a16_gemm_out",
    op_func=_mojo_w4a16_gemm_out_impl,
    mutates_args=["out"],
    fake_impl=_mojo_w4a16_gemm_out_fake,
    tags=_mojo_runtime.pt2_tags(),
)
direct_register_custom_op(
    op_name="mojo_w4a16_gemm",
    op_func=_mojo_w4a16_gemm_impl,
    mutates_args=[],
    fake_impl=_mojo_w4a16_gemm_fake,
    tags=_mojo_runtime.pt2_tags(),
)


def mojo_w4a16_gemm_op(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
    use_v2_format: bool = False,
    qweight_kpacked: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_inputs(a, qweight, scales, qzeros, group_size, zp_bias)
    is_compiling = _is_compiling
    _debug_log(
        "op "
        f"a={tuple(a.shape)} qweight={tuple(qweight.shape)} "
        f"scales={tuple(scales.shape)} has_qzeros={qzeros is not None} "
        f"group={group_size} zp_bias={zp_bias} {_stream_debug(a)}"
    )
    if not is_compiling() and not torch.cuda.is_current_stream_capturing():
        prepare_mojo_w4a16_gemm(
            a, qweight, scales, qzeros, group_size, zp_bias, use_v2_format
        )
    if is_compiling():
        out = torch.empty(
            (a.shape[0], qweight.shape[1] * 8),
            dtype=a.dtype,
            device=a.device,
        )
        torch.ops.vllm.mojo_w4a16_gemm_out(
            out,
            a,
            qweight,
            qweight_kpacked if qweight_kpacked is not None else qweight,
            scales,
            qzeros if qzeros is not None else qweight,
            group_size,
            zp_bias,
            qzeros is not None,
            use_v2_format,
        )
        return out
    cfg = _select_config_for_tensors(
        a, qweight, scales, qzeros, group_size, zp_bias, use_v2_format
    )
    qweight_kpacked_or_dummy = (
        qweight_kpacked
        if qweight_kpacked is not None
        else _get_qweight_kpacked(cfg, qweight)
    )
    return torch.ops.vllm.mojo_w4a16_gemm(
        a,
        qweight,
        qweight_kpacked_or_dummy,
        scales,
        qzeros if qzeros is not None else qweight,
        group_size,
        zp_bias,
        qzeros is not None,
        use_v2_format,
    )


def mojo_w4a16_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None = None,
    group_size: int | None = None,
    zp_bias: int = 8,
    use_v2_format: bool = False,
    qweight_kpacked: torch.Tensor | None = None,
) -> torch.Tensor:
    if group_size is None:
        group_size = int(a.shape[1] // scales.shape[0])
    return mojo_w4a16_gemm_op(
        a,
        qweight,
        scales,
        qzeros,
        group_size,
        zp_bias,
        use_v2_format,
        qweight_kpacked,
    )


class MojoW4A16LinearKernel(TritonW4A16LinearKernel):
    """Mojo W4A16 GEMM for ROCm RDNA.

    Weight preprocessing is inherited from ``TritonW4A16LinearKernel``. Kernel
    selection is explicit via ``--linear-backend mojo``.
    """

    SUPPORTED_QUANT_TYPES = TRITON_W4A16_SUPPORTED_QUANT_TYPES

    def __init__(
        self,
        c: MPLinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
        w_zp_param_name: str | None = None,
        w_gidx_param_name: str | None = None,
    ) -> None:
        super().__init__(
            c, w_q_param_name, w_s_param_name, w_zp_param_name, w_gidx_param_name
        )
        safe_q_name = w_q_param_name.replace(".", "_")
        self.w_q_kpacked_name = f"{safe_q_name}_mojo_kpacked"

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        linear_backend = (
            config.kernel_config.linear_backend if config is not None else "auto"
        )
        if linear_backend != "mojo":
            return False, "Mojo W4A16 is opt-in via --linear-backend mojo"

        dependencies_ok, dependency_error = _mojo_runtime.check_dependencies()
        if not dependencies_ok:
            return False, dependency_error
        if not current_platform.is_rocm():
            return False, "MojoW4A16LinearKernel is ROCm-only"
        from vllm.platforms.rocm import on_gfx1100, on_gfx1151

        if not (on_gfx1100() or on_gfx1151()):
            return False, "Mojo W4A16 currently targets RDNA3/RDNA3.5"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Mojo W4A16 supports fp16 and bf16"
        if c.partition_weight_shape[1] % 8 != 0:
            return False, "Output features must be divisible by 8"
        if c.has_g_idx:
            return False, "Mojo W4A16 path does not support g_idx yet"

        k = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else k
        if c.group_size not in TRITON_W4A16_SUPPORTED_GROUP_SIZES and c.group_size != k:
            return False, f"Group size {c.group_size} is not supported"
        if k % group_size:
            return False, f"Input features {k} not divisible by group size {group_size}"
        if not c.zero_points and (
            not c.weight_type.has_bias() or c.weight_type.bias != 8
        ):
            return (
                False,
                "Mojo constant zero-point path currently supports zp_bias=8",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        w_q, _, _, _ = self._get_weight_params(layer)
        if w_q.shape[0] % 8:
            return

        qweight_kpacked = _make_qweight_kpacked(w_q)
        _cache_prepared_qweight_kpacked(w_q, qweight_kpacked)
        if self.w_q_kpacked_name in layer._buffers:
            layer._buffers[self.w_q_kpacked_name] = qweight_kpacked
        else:
            layer.register_buffer(
                self.w_q_kpacked_name,
                qweight_kpacked,
                persistent=False,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        k = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else k
        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0
        w_q_kpacked = getattr(layer, self.w_q_kpacked_name, None)

        output = mojo_w4a16_gemm_op(
            x_2d,
            w_q,
            w_s,
            w_zp,
            group_size,
            zp_bias,
            True,
            w_q_kpacked,
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


__all__ = [
    "MojoW4A16LinearKernel",
    "mojo_w4a16_gemm",
    "mojo_w4a16_gemm_op",
    "prepare_mojo_w4a16_gemm",
]
