# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

from .base import MambaSSUKernel, SSULayer

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


def triton_stochastic_rounding_reasons(cfg: LinearAttnSelectorConfig) -> list[str]:
    """Why the Triton SSU kernels' `convert_rs_fp16x2` can't round the SSM
    state stochastically here, if it is enabled."""
    if not cfg.use_stochastic_rounding:
        return []
    reasons = []
    spec = cfg.kv_cache_spec
    if isinstance(spec, MambaSpec) and spec.dtypes[1] != torch.float16:
        reasons.append(
            f"stochastic rounding requires a float16 SSM cache, got {spec.dtypes[1]}"
        )
    # The `cvt.rs` PTX instruction (SM10x), or gfx950's (capability 9.5).
    capability = cfg.device_capability
    if capability is None or not (
        (cfg.platform == PlatformEnum.CUDA and capability.major == 10)
        or (
            cfg.platform == PlatformEnum.ROCM
            and (capability.major, capability.minor) == (9, 5)
        )
    ):
        reasons.append(
            "Triton stochastic rounding requires compute capability 10.x "
            "on CUDA or gfx950 on ROCm"
        )
    return reasons


class TritonSSUKernel(MambaSSUKernel):
    backend = MambaBackendEnum.TRITON

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform != PlatformEnum.CPU

    @classmethod
    def supports_stochastic_rounding(cls) -> bool:
        return True

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    @classmethod
    def supports_pipeline_parallel(cls) -> bool:
        return True

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        return triton_stochastic_rounding_reasons(cfg)

    @staticmethod
    @cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update,
        )

        return selective_state_update

    @staticmethod
    @cache
    def _is_blackwell() -> bool:
        return current_platform.is_device_capability_family(100)

    @classmethod
    def forward(
        cls,
        layer: SSULayer,
        metadata: "BaseMambaAttentionMetadata",
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> None:
        cls._fn()(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=metadata.state_indices_tensor_d,
            dst_state_batch_indices=metadata.state_indices_tensor_d,
            null_block_id=NULL_BLOCK_ID,
            out=out,
            num_accepted_tokens=metadata.num_accepted_tokens,
            cu_seqlens=metadata.query_start_loc_d,
            is_blackwell=cls._is_blackwell(),
            enable_stochastic_rounding=layer.mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=layer.mamba_config.stochastic_rounding_philox_rounds,
        )
