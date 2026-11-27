# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft attention beside an UltraQuant target.

UltraQuant pages are LBNHC. Shipping ``ROCM_AITER_FA`` is LBHNC/LHBNC, so the
two backends have no layout in common and the draft cannot be allocated next
to the target. This backend accepts LBNHC and reads a StreamingLLM span: the
first 4 tokens plus the last 2048. That is the DSpark draft path. Target
verification stays on UltraQuant and still sees the full context.

Selected by default when the target KV dtype is ``ultraquant_4bit`` and the
speculative config does not name a draft attention backend.
"""

from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
    AiterFlashAttentionMetadata,
)
from vllm.v1.attention.ops.aiter_noncausal_decode import noncausal_decode_attention
from vllm.v1.attention.ops.dspark_draft_window import (
    DSPARK_DRAFT_SINK,
    DSPARK_DRAFT_WINDOW,
    build_draft_window_sink_tables,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)

# 3136/4. A bf16 draft page at this block stays under a 3136-token UltraQuant
# page, and the scheduler LCM stays on the hybrid manager block.
_DENSE_DRAFT_BLOCK = 784


def _window_decode_metadata(
    attn_metadata: AiterFlashAttentionMetadata,
    num_decodes: int,
    page_win: torch.Tensor,
    seq_win: torch.Tensor,
) -> AiterFlashAttentionMetadata:
    """Point decode rows at the packed window. Prefill rows stay full-context."""
    if num_decodes == attn_metadata.block_table.shape[0]:
        return replace(attn_metadata, block_table=page_win, seq_lens=seq_win)

    rest = attn_metadata.block_table[num_decodes:]
    win_w = page_win.shape[1]
    rest_w = rest.shape[1] if rest.numel() else 0
    out_w = max(win_w, rest_w)
    new_bt = torch.zeros(
        (attn_metadata.block_table.shape[0], out_w),
        device=page_win.device,
        dtype=page_win.dtype,
    )
    new_bt[:num_decodes, :win_w] = page_win
    if rest_w > 0:
        new_bt[num_decodes:, :rest_w] = rest

    src_seq = attn_metadata.seq_lens
    if src_seq.device != page_win.device:
        if page_win.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return attn_metadata
        src_seq = src_seq.to(device=page_win.device)
    new_seq = src_seq.clone()
    new_seq[:num_decodes] = seq_win
    return replace(attn_metadata, block_table=new_bt, seq_lens=new_seq)


class AiterFlashAttentionDSparkImpl(AiterFlashAttentionImpl):
    def _fp8_noncausal_decode(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
    ) -> bool:
        """The parent FA path rejects a bf16 query against an fp8 cache.

        The split-KV kernel takes that pair. It is the FP8 draft path. An
        unquantized cache stays on the parent, which is the UltraQuant draft.
        """
        decode = getattr(attn_metadata, "decode_metadata", None)
        return (
            is_quantized_kv_cache(self.kv_cache_dtype)
            and attn_metadata.causal is False
            and attn_metadata.num_decodes > 0
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_extends == 0
            and decode is not None
            and self.sliding_window[0] == -1
            and self.sinks is None
            and self.alibi_slopes is None
            and not self.logits_soft_cap
            and self.kv_sharing_target_layer_name is None
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_decodes = int(getattr(attn_metadata, "num_decodes", 0) or 0)
        if attn_metadata is not None and num_decodes > 0:
            # kv_cache: [num_blocks, 2, block_size, ...]
            block_size = int(kv_cache.shape[2])
            page_win, seq_win = build_draft_window_sink_tables(
                attn_metadata.block_table[:num_decodes],
                attn_metadata.seq_lens[:num_decodes],
                block_size=block_size,
                window=DSPARK_DRAFT_WINDOW,
                sink=DSPARK_DRAFT_SINK,
            )
            logger.info_once(
                "DSpark draft attention is AITER FA with an UltraQuant "
                "layout: a %d-token sink plus the last %d tokens. Target "
                "verification is unchanged.",
                DSPARK_DRAFT_SINK,
                DSPARK_DRAFT_WINDOW,
            )
            attn_metadata = _window_decode_metadata(
                attn_metadata, num_decodes, page_win, seq_win
            )
            if self._fp8_noncausal_decode(attn_metadata):
                key_cache, value_cache = self._split_kv_cache(kv_cache)
                key_cache = key_cache.view(current_platform.fp8_dtype())
                value_cache = value_cache.view(current_platform.fp8_dtype())
                num_decode_tokens = attn_metadata.num_decode_tokens
                noncausal_decode_attention(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=attn_metadata.query_start_loc[: num_decodes + 1],
                    max_seqlen_q=attn_metadata.decode_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens[:num_decodes],
                    block_table=attn_metadata.block_table[:num_decodes],
                    softmax_scale=self.scale,
                    k_descale=layer._k_scale,
                    v_descale=layer._v_scale,
                )
                return output
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )


class AiterFlashAttentionDSparkBackend(AiterFlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_FA_DSPARK"

    @staticmethod
    def get_impl_cls() -> type[AiterFlashAttentionDSparkImpl]:
        return AiterFlashAttentionDSparkImpl

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # LBNHC first so the draft allocates next to an UltraQuant target.
        return (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        del default_block_size
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size in (16, 32, _DENSE_DRAFT_BLOCK) or block_size % 64 == 0

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        del kv_cache_spec
        # 16 and 32 match an FP8 target. 64 matches an UltraQuant page. 784 is
        # the dense draft page: it divides a 3136-token UltraQuant block, and
        # a bf16 page at that size does not pad the 4-bit pages.
        return [16, 32, 64, _DENSE_DRAFT_BLOCK]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # The hybrid planner raises every attention block to the manager
        # block. A bf16 draft page at that size is several times an
        # UltraQuant page, and every target page is then padded up to it.
        spec = super().customize_spec(spec)
        if (
            spec.block_size > _DENSE_DRAFT_BLOCK
            and spec.block_size % _DENSE_DRAFT_BLOCK == 0
        ):
            return replace(spec, block_size=_DENSE_DRAFT_BLOCK)
        return spec
