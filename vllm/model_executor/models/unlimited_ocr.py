# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Unlimited-OCR model compatible with HuggingFace weights.

Unlimited-OCR (``baidu/Unlimited-OCR`` / ``PaddlePaddle/Unlimited-OCR``) shares
the exact DeepSeek-OCR (gundam, ``base_size=1024`` / ``image_size=640`` / crop)
vision stack: a DeepEncoder (SAM-ViT-B + CLIP-L) followed by a linear MLP
projector, with the same image-token tiling layout. The only difference is the
language backbone, which is a DeepSeek-V2 *MoE* (64 routed + 2 shared experts,
``first_k_dense_replace=1``) that uses plain multi-head attention
(``use_mla=False``, ``qk_nope_head_dim == qk_rope_head_dim == 0``) instead of
the dense MLA decoder used by DeepSeek-OCR.

vLLM's ``DeepseekV2DecoderLayer`` already dispatches to the plain-MHA
``DeepseekAttention`` whenever ``qk_nope_head_dim == qk_rope_head_dim == 0`` and
builds the MoE blocks straight from the config, so the whole DeepSeek-OCR
multimodal wrapper can be reused verbatim. The released checkpoint, however,
mislabels the language backbone in ``language_config.architectures`` as
``"DeepseekOCRForCausalLM"`` (the multimodal wrapper itself); we repoint it at
the registered ``DeepseekV2ForCausalLM`` text model before initialization.

Attention backend: the reference applies Reference Sliding Window Attention
(R-SWA) -- the prompt/image tokens form a globally-visible prefix while the
*generated* tokens additionally attend only a fixed sliding window (128) of
recent tokens. We reproduce this (Level 1: full KV cache + custom mask) by
forcing the language model onto the FlexAttention backend and installing an
R-SWA ``mask_mod``. FlexAttention is the only backend able to express the
"global prefix + sliding window" mask; FlashAttention-3 / Triton only support a
uniform window (and additionally crash or compute incorrectly on this decoder's
10-head,
head_dim-128 shape), and FlashInfer's paged decode exposes no custom mask. The
window size is published via the ``rswa_window`` attribute, which the model
runner reads to plumb per-request prefix lengths into the FlexAttention mask.

The *vision encoder* (DeepEncoder's CLIP stage, head_dim 64) is unaffected and
does not use R-SWA: it runs a single full-attention prefill pass. FlashAttention,
Triton and torch SDPA all produce correct, equally fast results; only FlashInfer
is incompatible (its ViT path asserts on the varlen cu_seqlens metadata that this
CLIP encoder never builds). We default the encoder to FlashAttention and
transparently fall back to it if FlashInfer is requested.

To additionally suppress repetition on long documents, enable
``UnlimitedOCRNoRepeatNGramLogitsProcessor`` (see below).
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

from .deepseek_ocr import (
    DeepseekOCRDummyInputsBuilder,
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=DeepseekOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        text_config = vllm_config.model_config.hf_config.text_config
        # The checkpoint declares the language backbone as the multimodal
        # wrapper ("DeepseekOCRForCausalLM"). The actual text model is a
        # standard DeepSeek-V2 (MoE + plain MHA), so point it at the registered
        # DeepSeek-V2 language model to avoid recursing into the wrapper.
        text_config.architectures = ["DeepseekV2ForCausalLM"]

        # Language model: reproduce R-SWA (global prefix + sliding window over
        # generated tokens) via FlexAttention's custom mask. FlexAttention is
        # the only backend that can express this mask; FlashAttention-3 / Triton
        # only do a uniform window (and crash or compute incorrectly on this
        # decoder's 10-head, head_dim-128 shape) and FlashInfer's paged decode
        # has no
        # custom-mask path. Force FlexAttention unconditionally, overriding any
        # user-selected backend.
        attn_config = vllm_config.attention_config
        if attn_config.backend is None:
            attn_config.backend = AttentionBackendEnum.FLEX_ATTENTION
            logger.info_once(
                "Unlimited-OCR: forcing FlexAttention backend for the language "
                "model to implement Reference Sliding Window Attention (R-SWA)."
            )
        elif attn_config.backend != AttentionBackendEnum.FLEX_ATTENTION:
            logger.warning_once(
                "Unlimited-OCR: language-model attention backend %s cannot "
                "express the R-SWA mask (only FlexAttention can); overriding to "
                "FlexAttention.",
                attn_config.backend,
            )
            attn_config.backend = AttentionBackendEnum.FLEX_ATTENTION

        # Vision encoder (DeepEncoder's CLIP stage, head_dim 64): FlashAttention,
        # Triton and torch SDPA all produce correct results, but FlashInfer's ViT
        # path requires the varlen cu_seqlens metadata that this CLIP encoder
        # never builds (it runs a single full-attention pass) and crashes with an
        # assertion. Default to FlashAttention and reject FlashInfer.
        mm_config = getattr(vllm_config.model_config, "multimodal_config", None)
        if mm_config is not None:
            if mm_config.mm_encoder_attn_backend is None:
                mm_config.mm_encoder_attn_backend = AttentionBackendEnum.FLASH_ATTN
            elif mm_config.mm_encoder_attn_backend == AttentionBackendEnum.FLASHINFER:
                logger.warning_once(
                    "Unlimited-OCR: FlashInfer is not supported for the vision "
                    "encoder (the CLIP stage runs full attention without "
                    "cu_seqlens); falling back to FlashAttention."
                )
                mm_config.mm_encoder_attn_backend = AttentionBackendEnum.FLASH_ATTN

        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Publish the R-SWA sliding-window size (matches the reference). The
        # model runner reads this to enable the FlexAttention R-SWA mask and to
        # plumb per-request prefix (prompt/image) lengths into it.
        self.rswa_window = 128


class UnlimitedOCRNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Sliding-window no-repeat n-gram logits processor for Unlimited-OCR.

    Mirrors the reference ``SlidingWindowNoRepeatNgramProcessor`` (aligned with
    SGLang's ``DeepseekOCRNoRepeatNGramLogitProcessor``): within the last
    ``ngram_window`` tokens, forbid any token that would complete an
    ``no_repeat_ngram_size``-gram already present in that window. This is what
    stops the greedy decoder from looping on long documents.

    Enable it by passing the class to ``LLM(logits_processors=[...])`` (or the
    ``--logits-processors`` CLI flag) and then activating it per request via
    ``SamplingParams.extra_args``::

        from vllm.model_executor.models.unlimited_ocr import (
            UnlimitedOCRNoRepeatNGramLogitsProcessor,
        )

        llm = LLM(
            model=..., logits_processors=[UnlimitedOCRNoRepeatNGramLogitsProcessor]
        )
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args={"no_repeat_ngram_size": 35, "ngram_window": 128},
        )

    A request is left untouched unless it sets ``no_repeat_ngram_size`` (a
    non-positive ``ngram_window`` falls back to an unbounded window, i.e. the
    classic no-repeat-ngram behaviour).
    """

    def __init__(self, vllm_config: VllmConfig, device, is_pin_memory: bool):
        # req batch index -> (ngram_size, window, prompt_tok_ids, output_tok_ids)
        self.req_state: dict[int, tuple[int, int, list[int], list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        # Banning tokens can change the greedy argmax.
        return False

    @staticmethod
    def _new_state(params, prompt_tok_ids, output_tok_ids):
        extra = params.extra_args or {}
        ngram_size = extra.get("no_repeat_ngram_size") or 0
        if ngram_size is None or int(ngram_size) < 1:
            return None
        window = extra.get("ngram_window") or 0
        window = int(window) if window and int(window) > 0 else (1 << 30)
        return (int(ngram_size), window, prompt_tok_ids or [], output_tok_ids)

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.req_state, batch_update, self._new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_state:
            return logits
        for index, (n, window, prompt_ids, out_ids) in self.req_state.items():
            total = len(prompt_ids) + len(out_ids)
            if total < n:
                continue
            need = min(window, total)
            # Last `need` tokens of (prompt + output) without copying the whole
            # (large, image-token-heavy) prompt every decode step.
            if len(out_ids) >= need:
                seq = list(out_ids[-need:])
            else:
                head = prompt_ids[len(prompt_ids) - (need - len(out_ids)) :]
                seq = list(head) + list(out_ids)
            seq_len = len(seq)
            if seq_len < n:
                continue
            prefix = tuple(seq[seq_len - (n - 1) :]) if n > 1 else ()
            banned: set[int] = set()
            for i in range(0, seq_len - n + 1):
                if n == 1 or tuple(seq[i : i + n - 1]) == prefix:
                    banned.add(seq[i + n - 1])
            if banned:
                logits[index, list(banned)] = float("-inf")
        return logits
