# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.platforms.interface import PlatformEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

from .base import MambaSSUKernel, SSULayer

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class FlashInferSSUKernel(MambaSSUKernel):
    backend = MambaBackendEnum.FLASHINFER

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CUDA

    @classmethod
    def supports_stochastic_rounding(cls) -> bool:
        return True

    @classmethod
    def supports_ssu_algorithm(cls) -> bool:
        return True

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    @classmethod
    def supports_pipeline_parallel(cls) -> bool:
        return True

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.mamba import selective_state_update  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        spec = cfg.kv_cache_spec
        if not isinstance(spec, MambaSpec):
            return []
        reasons = []
        state_size, state_dtype = spec.shapes[1][-1], spec.dtypes[1]
        # FlashInfer (0.7.4) splits the state over a warp's 32 lanes, loads it in
        # 64-byte stages, and needs 128 bytes once rounded up to a power of two.
        smallest = max(32, 128 // state_dtype.itemsize)
        multiple = 64 // state_dtype.itemsize
        if state_size < smallest or state_size % multiple:
            reasons.append(
                f"FlashInfer SSU requires a state size of at least {smallest}, "
                f"a multiple of {multiple}, for a {state_dtype} SSM state; got "
                f"{state_size}"
            )
        if cfg.use_stochastic_rounding and state_dtype != torch.float16:
            reasons.append(
                f"stochastic rounding requires a float16 SSM cache, got {state_dtype}"
            )
        return reasons

    @staticmethod
    @cache
    def _fn():
        from flashinfer.mamba import selective_state_update

        return selective_state_update

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
        mamba_config = layer.mamba_config
        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device)
            if mamba_config.enable_stochastic_rounding
            else None
        )
        indices = metadata.state_indices_tensor_d
        cu_seqlens = metadata.query_start_loc_d
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
            state_batch_indices=indices,
            dst_state_batch_indices=indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=metadata.num_accepted_tokens,
            cache_steps=indices.size(-1)
            if cu_seqlens is not None and indices is not None
            else 0,
            pad_slot_id=NULL_BLOCK_ID,
            out=out,
            rand_seed=rand_seed,
            philox_rounds=mamba_config.stochastic_rounding_philox_rounds or 10,
            algorithm=mamba_config.ssu_algorithm or "auto",
        )
