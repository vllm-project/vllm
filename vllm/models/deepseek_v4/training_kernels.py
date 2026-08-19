"""Thin calls into the vLLM DeepSeek-V4 kernels used by mLite training."""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn



def _symbol(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(f"missing vLLM kernel {module}.{name}") from exc


def _op(namespace: str, name: str):
    try:
        return getattr(getattr(torch.ops, namespace), name)
    except AttributeError as exc:
        raise NotImplementedError(f"missing torch.ops.{namespace}.{name}") from exc


class MHCKernel(str, Enum):
    PRE = "pre"
    PRE_BROADCAST = "pre_broadcast"
    POST = "post"
    POST_PRE = "post_pre"
    HEAD = "head"


_MHC_ENTRIES = {
    MHCKernel.PRE: "mhc_pre_tilelang",
    MHCKernel.PRE_BROADCAST: "mhc_pre_broadcast_tilelang",
    MHCKernel.POST: "mhc_post_tilelang",
    MHCKernel.POST_PRE: "mhc_fused_post_pre_tilelang",
    MHCKernel.HEAD: "hc_head_fused_kernel_tilelang",
}


class MHCTileLangAdapter(nn.Module):
    def __init__(self, kernel: MHCKernel | str) -> None:
        super().__init__()
        self.kernel = MHCKernel(kernel)

    def forward(self, *args, **kwargs):
        return _symbol(
            "vllm.model_executor.kernels.mhc.tilelang",
            _MHC_ENTRIES[self.kernel],
        )(*args, **kwargs)


class FusedQKVRMSNormAdapter(nn.Module):
    def forward(self, q, kv, q_weight, kv_weight, eps):
        return _symbol("vllm.models.common.ops", "fused_q_kv_rmsnorm")(
            q, kv, q_weight, kv_weight, eps
        )


class KVCacheLayout(str, Enum):
    FP8_DS_MLA = "fp8_ds_mla"


class DS4KVInsertAdapter(nn.Module):
    """Invoke vLLM's fused QNorm/RoPE/fresh-KV quantization boundary."""

    def __init__(self, layout: KVCacheLayout | str) -> None:
        super().__init__()
        if KVCacheLayout(layout) is not KVCacheLayout.FP8_DS_MLA:
            raise ValueError("mLite.vllm supports only the fp8_ds_mla prefill layout")

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        cache: Tensor,
        slot_mapping: Tensor,
        positions: Tensor,
        cos_sin_cache: Tensor,
        *,
        eps: float,
        block_size: int,
        padded_heads: int | None = None,
        q_out: Tensor | None = None,
        **_unused,
    ) -> Tensor:
        if padded_heads is None:
            raise ValueError("padded_heads is required")
        if q_out is not None:
            _op("_C", "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out")(
                q,
                kv,
                q_out,
                cache,
                slot_mapping,
                positions,
                cos_sin_cache,
                padded_heads,
                eps,
                block_size,
            )
            return q_out
        return _op("_C", "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")(
            q,
            kv,
            cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            padded_heads,
            eps,
            block_size,
        )


class FlashMLAAdapter(nn.Module):
    def sparse(
        self,
        q: Tensor,
        kv: Tensor,
        indices: Tensor,
        *,
        sm_scale: float,
        attn_sink: Tensor | None = None,
        topk_length: Tensor | None = None,
        out: Tensor | None = None,
    ):
        return _symbol("vllm.v1.attention.ops.flashmla", "flash_mla_sparse_fwd")(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
        )


class OProjectionAdapter(nn.Module):
    """Pack mLite masters and invoke vLLM's official grouped FP8 O projection."""

    def __init__(self, quantize_weight) -> None:
        super().__init__()
        self.quantize_weight = quantize_weight

    def forward(
        self,
        o: Tensor,
        positions: Tensor,
        cos_sin_cache: Tensor,
        wo_a: Tensor,
        wo_b: Tensor,
        *,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
        o_lora_rank: int,
    ) -> Tensor:
        post_process = _symbol(
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        )
        official = _symbol(
            "vllm.models.deepseek_v4.nvidia.ops.o_proj", "deep_gemm_fp8_o_proj"
        )
        recipe_fn = _symbol(
            "vllm.models.deepseek_v4.nvidia.ops.o_proj", "compute_fp8_einsum_recipe"
        )

        with torch.inference_mode():
            canonical_wa = self.quantize_weight(wo_a)
            wa_q, wa_s = post_process(
                wq=canonical_wa.qweight,
                ws=canonical_wa.scales,
                quant_block_shape=(128, 128),
                use_e8m0=True,
                is_bmm=True,
                bmm_batch_size=n_groups,
            )
            packed_wa = type("_PackedGroupedWeight", (), {})()
            packed_wa.weight = wa_q
            packed_wa.weight_scale = wa_s

            canonical_wb = self.quantize_weight(wo_b)
            wb_q, wb_s = post_process(
                wq=canonical_wb.qweight,
                ws=canonical_wb.scales,
                quant_block_shape=(128, 128),
                use_e8m0=True,
            )

            def packed_wb(value: Tensor) -> Tensor:
                activation_quant = _symbol(
                    "vllm.model_executor.layers.quantization.utils.fp8_utils",
                    "per_token_group_quant_fp8",
                )
                aligned = bool(
                    _symbol("vllm.envs", "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES")
                )
                aq, a_s = activation_quant(
                    value,
                    128,
                    use_ue8m0=True,
                    column_major_scales=True,
                    tma_aligned_scales=aligned,
                )
                output = torch.empty(
                    value.shape[0],
                    wb_q.shape[0],
                    dtype=torch.bfloat16,
                    device=value.device,
                )
                _symbol("vllm.utils.deep_gemm", "fp8_gemm_nt")(
                    (aq, a_s),
                    (wb_q, wb_s),
                    output,
                    is_deep_gemm_e8m0_used=True,
                )
                return output

            recipe, aligned = recipe_fn()
            return official(
                o,
                positions,
                cos_sin_cache,
                packed_wa,
                packed_wb,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                o_lora_rank=o_lora_rank,
                einsum_recipe=recipe,
                tma_aligned_scales=aligned,
            )


__all__ = [
    "DS4KVInsertAdapter",
    "FlashMLAAdapter",
    "FusedQKVRMSNormAdapter",
    "KVCacheLayout",
    "MHCKernel",
    "MHCTileLangAdapter",
    "OProjectionAdapter",
]
