# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aiter Triton MLA 后端模块。

本模块实现了基于 ROCm Aiter Triton 的 MLA 后端，负责：
- 继承 AiterMLABackend 并重命名为 AITER_TRITON_MLA
- 使用 Triton MHA kernel 执行 MLA 计算
- 支持 LSE 转置以适配 Triton MHA 输出格式

主要类：
- AiterTritonMLABackend: Aiter Triton MLA 后端类
- AiterTritonMLAImpl: Aiter Triton MLA 实现类
"""
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend, AiterMLAImpl


class AiterTritonMLABackend(AiterMLABackend):
    @staticmethod
    def get_name() -> str:
        return "AITER_TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterTritonMLAImpl"]:
        return AiterTritonMLAImpl


class AiterTritonMLAImpl(AiterMLAImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        from aiter.ops.triton.mha import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        result = self.flash_attn_varlen_func(  # type: ignore[call-arg]
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )
        # Transpose the LSE if Triton MHA is used:
        # (q.shape[0], num_q_heads) to (num_q_heads, q.shape[0])
        if type(result) is tuple and return_softmax_lse:
            output, lse = result
            lse = lse.T.contiguous()
            return (output, lse)
        return result
