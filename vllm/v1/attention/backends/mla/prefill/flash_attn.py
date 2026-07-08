# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for MLA prefill."""

import functools
from typing import TYPE_CHECKING, ClassVar

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.model_executor.warmup.fa4_cutedsl_config import (
    FA4MLAPrefillCompileContext,
    iter_fa4_mla_prefill_compile_requests,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import (
    compile_flash_attn_varlen_func_from_specs,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


class FlashAttnPrefillBackend(MLAPrefillBackend):
    """FlashAttention backend for MLA prefill."""

    supported_mla_dimensions: ClassVar[list[MLADimensions]] = [
        MLADimensions(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        return is_flash_attn_varlen_func_available()

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        # Handle the differences between the flash_attn_varlen from
        # flash_attn and the one from vllm_flash_attn
        assert flash_attn_varlen_func is not None, (
            "FlashAttnPrefillBackend requires flash_attn_varlen_func. "
            "Ensure FlashAttnPrefillBackend.is_available() is checked first."
        )
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version(
            head_size=qk_head_dim, head_size_v=v_head_dim
        )
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = functools.partial(
                flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
            )

        # Determine if we need to pad V
        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims.
        # FA3 on Hopper (SM90) and FA4 natively handle diff headdims.
        device_capability = current_platform.get_device_capability()
        self.requires_v_padding = self.vllm_flash_attn_version is None or not (
            (
                self.vllm_flash_attn_version == 3
                and device_capability is not None
                and device_capability[0] == 9
            )
            or self.vllm_flash_attn_version == 4
        )

        # Track whether we're using vllm's FA or upstream (for ROCm)
        self._is_vllm_fa = current_platform.is_cuda() or current_platform.is_xpu()
        if self.vllm_flash_attn_version == 4:
            register_cutedsl_warmup_provider(self)

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        if self.vllm_flash_attn_version != 4:
            return ()
        if compile_flash_attn_varlen_func_from_specs is None:
            raise RuntimeError(
                "FA4 compile-only API is unavailable; CuTeDSL warmup does not "
                "fall back to synthetic forward passes."
            )

        dtype = self.vllm_config.model_config.dtype
        if dtype not in self.supported_dtypes:
            dtype = torch.bfloat16

        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        ctx = FA4MLAPrefillCompileContext(
            dtype=dtype,
            num_heads=self.num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_nope_head_dim=self.qk_nope_head_dim + self.v_head_dim,
            requires_v_padding=self.requires_v_padding,
            scale=self.scale,
            num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
            fa_version=self.vllm_flash_attn_version,
        )
        compile_requests = tuple(iter_fa4_mla_prefill_compile_requests(ctx))
        if not compile_requests:
            return ()

        return tuple(
            CuTeDSLCompileUnit(
                name="fa4_mla_prefill",
                key=request.key,
                compile=request.compile,
            )
            for request in compile_requests
        )

    def supports_quant_output(self, quant_key: "QuantKey") -> bool:
        device_capability = current_platform.get_device_capability()
        return (
            self.vllm_flash_attn_version == 4
            and self._is_vllm_fa
            and device_capability is not None
            and device_capability[0] in (10, 11)
            and quant_key == kFp8StaticTensorSym
        )

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
            kwargs["out"] = out
            kwargs["output_scale"] = output_scale
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
            assert out is None and output_scale is None
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        attn_out = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Unpad output back to v_head_dim if we padded V
        if self.requires_v_padding:
            attn_out = attn_out[..., : v.shape[-1]]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
            out=out,
            output_scale=output_scale,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_metadata.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )
