# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Unlimited-OCR model compatible with HuggingFace weights.

Unlimited-OCR (``baidu/Unlimited-OCR``) shares
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
multimodal wrapper can be reused verbatim. Model-specific config (language
backbone architecture, FlexAttention for R-SWA, vision encoder backend, and
``rswa_window``) is applied in ``UnlimitedOCRForCausalLMConfig``.

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
window size is published via ``model_config.rswa_window``, which the model
runner reads to plumb per-request prefix lengths into the FlexAttention mask.

The *vision encoder* (DeepEncoder's CLIP stage, head_dim 64) is unaffected and
does not use R-SWA: it runs a single full-attention prefill pass. FlashAttention,
Triton and torch SDPA all produce correct, equally fast results; only FlashInfer
is incompatible (its ViT path asserts on the varlen cu_seqlens metadata that this
CLIP encoder never builds). We default the encoder to FlashAttention and
transparently fall back to it if FlashInfer is requested.

To suppress repetition on long documents, use ``NGramPerReqLogitsProcessor`` from
this module (same request-level processor as DeepSeek-OCR) with::

    SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args={"ngram_size": 35, "window_size": 128},
    )
"""

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .deepseek_ocr import (
    DeepseekOCRDummyInputsBuilder,
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
    NGramPerReqLogitsProcessor,
)

__all__ = [
    "NGramPerReqLogitsProcessor",
    "UnlimitedOCRForCausalLM",
]


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=DeepseekOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
