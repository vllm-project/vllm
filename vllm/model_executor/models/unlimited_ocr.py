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
recent tokens. We reproduce this with backend-specific custom masks: FA4 uses a
``mask_mod``, FlexAttention uses a Triton block mask, and TritonAttention uses a
unified-attention decode mask. FlashInfer's paged decode exposes no custom mask.
The window size is published via ``model_config.rswa_window``, which the model
runner reads to plumb per-request prefix lengths into the attention metadata.

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

Image processing
----------------
Unlimited-OCR supports up to 32 local crops (vs 6 for DeepSeek-OCR), i.e.
``dynamic_preprocess`` runs with ``max_num=32``.

Multi-image requests fall back to non-crop mode: crop ("gundam") mode is only
used for single-image input. DeepSeek-OCR does *not* have this restriction.

Because that fallback makes the per-image processor output depend on *how many*
images are in the request, it breaks the assumption behind vLLM's per-item
multimodal processing cache (``MultiModalProcessorOnlyCache``). We handle this
the same way ``DeepseekVL2MultiModalProcessor`` does: only the single-image case
(which always crops) is cached, while multi-image requests bypass the cache and
are recomputed fresh -- see ``_cached_apply_hf_processor`` below. This keeps the
processing cache consistent (verified by ``test_processing_correctness``).
"""

import math
from collections.abc import Mapping, Sequence

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import MultiModalProcessingInfo
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    count_tiles,
)

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

# Unlimited-OCR supports up to 32 local crops (vs 6 for DeepSeek-OCR).
_UNLIMITED_OCR_MAX_CROPS = 32


class UnlimitedOCRProcessingInfo(DeepseekOCRProcessingInfo):
    """ProcessingInfo for Unlimited-OCR: same as DeepSeek-OCR but with
    max_crops=32 instead of 6.  The higher crop count allows tiling very large
    document pages into up to 32 640×640 patches (dynamic_preprocess max_num=32).
    """

    def get_hf_config(self):
        from vllm.transformers_utils.configs.unlimited_ocr import UnlimitedOCRConfig

        return self.ctx.get_hf_config(UnlimitedOCRConfig)

    def get_hf_processor(self, **kwargs: object):
        from vllm.transformers_utils.processors.unlimited_ocr import (
            UnlimitedOCRProcessor,
        )

        v1_processor_config = dict(
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
            crop_mode=CROP_MODE,
            strategy="v1",
            max_crops=_UNLIMITED_OCR_MAX_CROPS,
        )
        return self.ctx.get_hf_processor(
            UnlimitedOCRProcessor,
            **{**v1_processor_config, **kwargs},
        )

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        patch_size = 16
        downsample_ratio = 4

        # Honour the caller-supplied `cropping` flag: multi-image callers pass
        # cropping=False to match UnlimitedOCRProcessor.tokenize_with_images.
        if cropping:
            if image_width <= IMAGE_SIZE and image_height <= IMAGE_SIZE:
                crop_ratio = [1, 1]
            else:
                crop_ratio = count_tiles(
                    image_width,
                    image_height,
                    max_num=_UNLIMITED_OCR_MAX_CROPS,
                    image_size=IMAGE_SIZE,
                )
            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((BASE_SIZE // patch_size) / downsample_ratio)
        h2 = w2 = math.ceil((IMAGE_SIZE // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        # With max_crops=32, the widest possible grid is 4×8 (aspect ratio 1:2).
        # A 2560×5120 image (4×640 × 8×640) selects exactly 4×8=32 tiles and
        # produces the maximum token count.
        return ImageSize(width=640 * 4, height=640 * 8)


class UnlimitedOCRMultiModalProcessor(DeepseekOCRMultiModalProcessor):
    """Multimodal processor for Unlimited-OCR.

    Disables crop mode for multi-image requests (to stay consistent with
    ``UnlimitedOCRProcessor.tokenize_with_images``), and -- since that makes the
    per-image output depend on the request's image count -- bypasses the
    per-item processing cache for multi-image requests, exactly like
    ``DeepseekVL2MultiModalProcessor``.

    DeepSeek-OCR does *not* apply either of these.
    """

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_unlimited_ocr(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                size = images.get_image_size(item_idx)

                # Disable crop mode for multi-image input.
                # UnlimitedOCRProcessor.tokenize_with_images applies the same
                # fallback, so both paths must agree on the effective crop flag.
                effective_cropping = CROP_MODE and len(images) == 1

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=size.width,
                    image_height=size.height,
                    cropping=effective_cropping,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_unlimited_ocr,
            )
        ]

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        # The processor logic differs for single-image (crop) vs multi-image
        # (no crop) requests. The processing cache assumes per-item output is
        # invariant of how many images are passed per prompt, so we only cache
        # the single-image case and recompute multi-image requests fresh.
        if inputs.mm_data_items.get_count("image", strict=False) > 1:
            return self._apply_hf_processor(inputs, timing_ctx)

        return super()._cached_apply_hf_processor(inputs, timing_ctx)


@MULTIMODAL_REGISTRY.register_processor(
    UnlimitedOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
