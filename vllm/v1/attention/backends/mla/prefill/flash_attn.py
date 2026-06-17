# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for MLA prefill."""

import functools
import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import (
    FlashAttentionCuTeDSLCompileSpec,
    compile_flash_attn_varlen_func_from_specs,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _tensor_values_for_log(value: object) -> object:
    if not isinstance(value, torch.Tensor):
        return value
    if value.numel() > 32:
        return {
            "shape": tuple(value.shape),
            "head": value[:16].detach().cpu().tolist(),
            "tail": value[-16:].detach().cpu().tolist(),
        }
    return value.detach().cpu().tolist()


if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


class FlashAttnPrefillBackend(MLAPrefillBackend):
    """FlashAttention backend for MLA prefill."""

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
        self.vllm_flash_attn_version = get_flash_attn_version(head_size=qk_head_dim)
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

    def get_cutedsl_warmup_plan(self, runner: object) -> object | None:
        if self.vllm_flash_attn_version != 4:
            return None

        from vllm.model_executor.warmup.cutedsl_warmup import (
            CuTeDSLCompileUnit,
            CuTeDSLWarmupPlan,
            get_cutedsl_warmup_token_sizes,
        )

        specs = tuple(
            self._iter_cutedsl_compile_specs(
                runner,
                get_cutedsl_warmup_token_sizes(runner),
            )
        )

        return CuTeDSLWarmupPlan(
            provider="fa4_mla_prefill",
            compile_units=tuple(
                CuTeDSLCompileUnit(
                    name="fa4_mla_prefill",
                    key=spec,
                    compile=spec.compile,
                )
                for spec in specs
            ),
        )

    def _iter_cutedsl_compile_specs(
        self,
        runner: object,
        token_sizes: Sequence[int],
    ) -> Iterable[FlashAttentionCuTeDSLCompileSpec]:
        if compile_flash_attn_varlen_func_from_specs is None:
            raise RuntimeError(
                "FA4 compile-only warmup API is unavailable; CuTeDSL warmup "
                "does not run synthetic forward passes."
            )

        dtype = getattr(runner, "dtype", torch.bfloat16)
        if dtype not in self.supported_dtypes:
            dtype = torch.bfloat16
        scheduler_config = getattr(runner, "scheduler_config", None)
        max_num_seqs = getattr(scheduler_config, "max_num_seqs", 1)

        context_compiled = False
        for num_tokens in token_sizes:
            for seq_lens in self._get_cutedsl_warmup_seq_lens(
                num_tokens, max_num_seqs
            ):
                for return_softmax_lse in (False, True):
                    yield self._get_cutedsl_prefill_new_tokens_compile_spec(
                        seq_lens=seq_lens,
                        dtype=dtype,
                        return_softmax_lse=return_softmax_lse,
                    )

                if context_compiled:
                    continue
                context_compiled = True
                yield self._get_cutedsl_context_chunk_compile_spec(
                    seq_lens=seq_lens,
                    dtype=dtype,
                )

    def _get_cutedsl_warmup_seq_lens(
        self,
        num_tokens: int,
        max_num_seqs: int,
    ) -> list[tuple[int, ...]]:
        if num_tokens <= 0:
            return []

        seq_lens: list[tuple[int, ...]] = [(num_tokens,)]

        if num_tokens > 2 and max_num_seqs > 1:
            ragged: list[int] = [1]
            remaining = num_tokens - 1
            if max_num_seqs > 2 and remaining > 2:
                mid = min(remaining - 1, max(2, num_tokens // 3))
                ragged.append(mid)
                remaining -= mid
            ragged.append(remaining)
            seq_lens.append(tuple(ragged[:max_num_seqs]))

        return list(dict.fromkeys(seq_lens))

    def _get_cutedsl_warmup_qkv_specs(
        self,
        num_tokens: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del dtype
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_head_dim = qk_head_dim if self.requires_v_padding else self.v_head_dim
        q_shape = (num_tokens, self.num_heads, qk_head_dim)
        v_shape = (
            num_tokens,
            self.num_heads,
            v_head_dim,
        )
        return q_shape, q_shape, v_shape

    def _get_cutedsl_warmup_v_stride(
        self,
        v_shape: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        if self.requires_v_padding:
            return None
        kv_nope_head_dim = self.qk_nope_head_dim + self.v_head_dim
        return (
            self.num_heads * kv_nope_head_dim,
            kv_nope_head_dim,
            1,
        )

    def _get_cutedsl_context_chunk_compile_spec(
        self,
        *,
        seq_lens: tuple[int, ...],
        dtype: torch.dtype,
    ) -> FlashAttentionCuTeDSLCompileSpec:
        context_lens = [max(seq_lens) + 1]
        context_lens.extend(0 for _ in seq_lens[1:])
        context_tokens = sum(context_lens)
        assert context_tokens > 0

        q_shape, _, _ = self._get_cutedsl_warmup_qkv_specs(
            sum(seq_lens), dtype
        )
        _, context_k_shape, context_v_shape = (
            self._get_cutedsl_warmup_qkv_specs(context_tokens, dtype)
        )
        return self._get_cutedsl_flash_attn_varlen_compile_spec(
            q_shape=q_shape,
            k_shape=context_k_shape,
            v_shape=context_v_shape,
            v_stride=self._get_cutedsl_warmup_v_stride(context_v_shape),
            dtype=dtype,
            cu_seqlens_q_shape=(len(seq_lens) + 1,),
            cu_seqlens_k_shape=(len(seq_lens) + 1,),
            max_seqlen_q=max(seq_lens),
            max_seqlen_k=max(context_lens),
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )

    def _get_cutedsl_prefill_new_tokens_compile_spec(
        self,
        *,
        seq_lens: tuple[int, ...],
        dtype: torch.dtype,
        return_softmax_lse: bool,
    ) -> FlashAttentionCuTeDSLCompileSpec:
        q_shape, k_shape, v_shape = self._get_cutedsl_warmup_qkv_specs(
            sum(seq_lens), dtype
        )
        return self._get_cutedsl_flash_attn_varlen_compile_spec(
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            v_stride=self._get_cutedsl_warmup_v_stride(v_shape),
            dtype=dtype,
            cu_seqlens_q_shape=(len(seq_lens) + 1,),
            cu_seqlens_k_shape=(len(seq_lens) + 1,),
            max_seqlen_q=max(seq_lens),
            max_seqlen_k=max(seq_lens),
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        if os.environ.get("VLLM_CUTEDSL_LOG_FA4_METADATA") == "1":
            self._log_cutedsl_fa4_metadata(
                q=q,
                k=k,
                v=maybe_padded_v,
                return_softmax_lse=return_softmax_lse,
                kwargs=kwargs,
            )

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

    def _get_cutedsl_flash_attn_varlen_compile_spec(
        self,
        *,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        v_stride: tuple[int, ...] | None = None,
        dtype: torch.dtype,
        cu_seqlens_q_shape: tuple[int, ...],
        cu_seqlens_k_shape: tuple[int, ...],
        max_seqlen_q: int,
        max_seqlen_k: int,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        causal: bool,
    ) -> FlashAttentionCuTeDSLCompileSpec:
        assert compile_flash_attn_varlen_func_from_specs is not None

        num_splits = 0
        if envs.VLLM_BATCH_INVARIANT:
            num_splits = 1

        spec = FlashAttentionCuTeDSLCompileSpec(
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            v_stride=v_stride,
            q_dtype=dtype,
            cu_seqlens_q_shape=cu_seqlens_q_shape,
            cu_seqlens_k_shape=cu_seqlens_k_shape,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            num_splits=num_splits,
            fa_version=self.vllm_flash_attn_version,
        )
        if os.environ.get("VLLM_CUTEDSL_LOG_FA4_METADATA") == "1":
            self._log_cutedsl_fa4_compile_spec(spec)
        return spec

    def _log_cutedsl_fa4_metadata(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        kwargs: dict[str, object],
    ) -> None:
        cu_q = kwargs.get("cu_seqlens_q")
        cu_k = kwargs.get("cu_seqlens_k")
        query_start_loc = _tensor_values_for_log(cu_q)
        key_start_loc = _tensor_values_for_log(cu_k)
        logger.info(
            "CUTEDSL_FA4_METADATA "
            "q_shape=%s k_shape=%s v_shape=%s "
            "q_stride=%s k_stride=%s v_stride=%s dtype=%s "
            "return_softmax_lse=%s max_seqlen_q=%s max_seqlen_k=%s "
            "causal=%s cu_seqlens_q=%s cu_seqlens_k=%s "
            "fa_version=%s requires_v_padding=%s",
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            q.stride(),
            k.stride(),
            v.stride(),
            q.dtype,
            return_softmax_lse,
            kwargs.get("max_seqlen_q"),
            kwargs.get("max_seqlen_k"),
            kwargs.get("causal"),
            query_start_loc,
            key_start_loc,
            self.vllm_flash_attn_version,
            self.requires_v_padding,
        )

    def _log_cutedsl_fa4_compile_spec(
        self,
        spec: FlashAttentionCuTeDSLCompileSpec,
    ) -> None:
        logger.info(
            "CUTEDSL_FA4_METADATA "
            "q_shape=%s k_shape=%s v_shape=%s "
            "q_stride=%s k_stride=%s v_stride=%s dtype=%s "
            "return_softmax_lse=%s max_seqlen_q=%s max_seqlen_k=%s "
            "causal=%s cu_seqlens_q_shape=%s cu_seqlens_k_shape=%s "
            "fa_version=%s requires_v_padding=%s",
            spec.q_shape,
            spec.k_shape,
            spec.v_shape,
            spec.q_stride,
            spec.k_stride,
            spec.v_stride,
            spec.q_dtype,
            spec.return_softmax_lse,
            spec.max_seqlen_q,
            spec.max_seqlen_k,
            spec.causal,
            spec.cu_seqlens_q_shape,
            spec.cu_seqlens_k_shape,
            spec.fa_version,
            self.requires_v_padding,
        )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
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
