# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRITON_ATTN with a FlyDSL decode/verify kernel for Kimi-K3.1's GQA drafts.

The release DSpark draft (5 GQA layers, 12 / 2 heads a TP rank, head dim 64,
a 2048-token causal sliding window) attends a few query rows a request (the
draft block) over its own cache. ``ops.flydsl_swa_gqa`` serves that on gfx950;
everything else is TRITON_ATTN's: metadata, cache layout and update, and the
forward for any call outside the kernel's slice, which logs why once
(``VLLM_K31_DRAFT_FLYDSL_ATTN=0`` keeps TRITON_ATTN).
"""

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import (
    flydsl_swa_gqa_call_reason,
    flydsl_swa_gqa_config_reason,
    flydsl_swa_gqa_fwd,
    flydsl_swa_gqa_warmup,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)


class KimiK3FlyDSLSwaGqaBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_FLYDSL_SWA_GQA"

    @staticmethod
    def get_impl_cls() -> type["KimiK3FlyDSLSwaGqaImpl"]:
        return KimiK3FlyDSLSwaGqaImpl


class KimiK3FlyDSLSwaGqaImpl(TritonAttentionImpl):
    """TritonAttentionImpl whose forward runs the FlyDSL kernel when it can."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window = self.sliding_window[0] + 1
        # The kernel reads a bf16 query and dequantizes an fp8 cache itself:
        # keep the Attention layer from quantizing q for an fp8 cache.
        self.supports_quant_query_input = False
        # None: the FlyDSL kernel serves this layer (decided at warmup).
        self._off_reason: str | None = "not warmed up"

    def _layer_reason(self, act_dtype: torch.dtype) -> str | None:
        if self.attn_type != AttentionType.DECODER:
            return f"attention type {self.attn_type}"
        if self.sliding_window[0] < 0 or self.sliding_window[1] != 0:
            return "not a causal sliding window"
        if (
            self.alibi_slopes is not None
            or self.logits_soft_cap
            or self.sinks is not None
            or self.chunk_lookback > -1
            or self.use_alibi_sqrt
        ):
            return "ALiBi / soft cap / sinks / chunked attention"
        if self._kv_quant_mode == KVQuantMode.NONE:
            kv_dtype = act_dtype
        elif self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR:
            kv_dtype = current_platform.fp8_dtype()
        else:
            return f"KV cache quantization {self._kv_quant_mode.name}"
        if act_dtype != torch.bfloat16:
            return f"model dtype {act_dtype}"
        return flydsl_swa_gqa_config_reason(
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            window=self._window,
            kv_dtype=kv_dtype,
        ) or self._warmup(kv_dtype)

    def _warmup(self, kv_dtype: torch.dtype) -> str | None:
        # JIT every query-row config now, not inside a CUDA-graph capture.
        try:
            flydsl_swa_gqa_warmup(
                torch.device("cuda", torch.accelerator.current_device_index()),
                self.num_kv_heads,
                self._window,
                kv_dtype,
            )
        except Exception as e:  # noqa: BLE001
            return f"warmup failed: {type(e).__name__}: {e}"
        return None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)
        self._off_reason = self._layer_reason(act_dtype)
        if self._off_reason is None:
            logger.info_once(
                "Kimi-K3 GQA draft attention: FlyDSL SWA kernel (window %d).",
                self._window,
            )
        else:
            logger.warning_once(
                "Kimi-K3 GQA draft attention: FlyDSL SWA kernel off (%s); "
                "using TRITON_ATTN.",
                self._off_reason,
            )

    def _call_reason(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> str | None:
        if output_scale is not None or output_block_scale is not None:
            return "fused output quantization"
        if attn_metadata.causal is not True:
            return "non-causal or per-request causal metadata"
        if (
            attn_metadata.rswa_prefix_lens is not None
            or attn_metadata.mm_prefix_range_tensor is not None
        ):
            return "R-SWA / multimodal prefix ranges"
        if query.dtype != torch.bfloat16:
            return f"query dtype {query.dtype}"
        return flydsl_swa_gqa_call_reason(
            num_reqs=attn_metadata.query_start_loc.shape[0] - 1,
            max_query_len=attn_metadata.max_query_len,
            window=self._window,
            kv_cache=kv_cache,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._off_reason is None and attn_metadata is not None:
            reason = self._call_reason(
                query, kv_cache, attn_metadata, output_scale, output_block_scale
            )
            if reason is None:
                fp8 = self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                try:
                    return flydsl_swa_gqa_fwd(
                        query,
                        kv_cache,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens,
                        attn_metadata.query_start_loc,
                        attn_metadata.max_query_len,
                        output,
                        self.scale,
                        self._window,
                        layer._k_scale if fp8 else None,
                        layer._v_scale if fp8 else None,
                    )
                except Exception as e:  # noqa: BLE001
                    if torch.cuda.is_current_stream_capturing():
                        raise
                    # Off for good: Triton serves this layer from now on.
                    self._off_reason = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "Kimi-K3 GQA draft attention: FlyDSL SWA kernel failed "
                        "(%s); using TRITON_ATTN from now on.",
                        self._off_reason,
                    )
            else:
                logger.warning_once(
                    "Kimi-K3 GQA draft attention: a call is outside the FlyDSL "
                    "SWA kernel (%s); TRITON_ATTN serves it.",
                    reason,
                )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
