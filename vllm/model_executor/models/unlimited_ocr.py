# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Unlimited-OCR model compatible with HuggingFace weights.
Unlimited-OCR-2 uses DASH attention, 16 crops and independent per-page cropping.
The processor behavior described below applies to Unlimited-OCR v1.

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
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.inputs import MultiModalHashes
from vllm.model_executor.layers.dash_attention import DASHAttention
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.hasher import MultiModalHasher
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
from vllm.multimodal.processing.processor import MultiModalProcessingResult
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
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
from .utils import make_empty_intermediate_tensors_factory, make_layers

__all__ = [
    "NGramPerReqLogitsProcessor",
    "UnlimitedOCRForCausalLM",
]

# Unlimited-OCR supports up to 32 local crops (vs 6 for DeepSeek-OCR).
_UNLIMITED_OCR_MAX_CROPS = 32
_UNLIMITED_OCR2_MAX_CROPS = 16


class UnlimitedOCRProcessingInfo(DeepseekOCRProcessingInfo):
    """ProcessingInfo for Unlimited-OCR: same as DeepSeek-OCR but with
    max_crops=32 instead of 6.  The higher crop count allows tiling very large
    document pages into up to 32 640×640 patches (dynamic_preprocess max_num=32).
    """

    def get_hf_config(self):
        from vllm.transformers_utils.configs.unlimited_ocr import UnlimitedOCRConfig

        return self.ctx.get_hf_config(UnlimitedOCRConfig)

    def _get_max_crops(self) -> int:
        if getattr(self.get_hf_config().text_config, "unlimited_ocr2", False):
            return _UNLIMITED_OCR2_MAX_CROPS
        return _UNLIMITED_OCR_MAX_CROPS

    def get_hf_processor(self, **kwargs: object):
        from vllm.transformers_utils.processors.unlimited_ocr import (
            UnlimitedOCR2Processor,
            UnlimitedOCRProcessor,
        )

        v1_processor_config = dict(
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
            crop_mode=CROP_MODE,
            strategy="v1",
            max_crops=self._get_max_crops(),
        )
        processor_cls = (
            UnlimitedOCR2Processor
            if getattr(self.get_hf_config().text_config, "unlimited_ocr2", False)
            else UnlimitedOCRProcessor
        )
        return self.ctx.get_hf_processor(
            processor_cls,
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
                    max_num=self._get_max_crops(),
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
        # A vertical grid maximizes both crop count and newline tokens.
        return ImageSize(width=IMAGE_SIZE, height=IMAGE_SIZE * self._get_max_crops())


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

        mm_kwargs = self.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
        crop_mode = mm_kwargs.get("crop_mode", CROP_MODE)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_unlimited_ocr(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                assert isinstance(images, ImageProcessorItems)
                size = images.get_image_size(item_idx)

                # Only v1 disables cropping for multi-image input. Match the
                # selected HF processor when expanding image placeholders.
                effective_cropping = bool(crop_mode) and (
                    len(images) == 1
                    or getattr(
                        self.info.get_hf_config().text_config, "unlimited_ocr2", False
                    )
                )

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

    def _get_mm_hashes(self, inputs: ProcessorInputs) -> MultiModalHashes:
        hashes = dict(super()._get_mm_hashes(inputs))
        images = inputs.mm_data_items.get("image")
        if isinstance(images, ImageProcessorItems):
            mm_kwargs = self.info.ctx.get_merged_mm_kwargs(
                inputs.hf_processor_mm_kwargs
            )
            cropping = bool(mm_kwargs.get("crop_mode", CROP_MODE)) and (
                len(images) == 1
                or getattr(
                    self.info.get_hf_config().text_config, "unlimited_ocr2", False
                )
            )
            algorithm = self.info.ctx.get_mm_config().mm_hasher_algorithm
            hashes["image"] = [
                MultiModalHasher.hash_kwargs(
                    algorithm, mm_hash=mm_hash, cropping=cropping
                )
                for mm_hash in hashes["image"]
            ]
        return hashes

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalProcessingResult:
        # The processor logic differs for single-image (crop) vs multi-image
        # (no crop) requests. The processing cache assumes per-item output is
        # invariant of how many images are passed per prompt, so we only cache
        # the single-image case and recompute multi-image requests fresh.
        if inputs.mm_data_items.get_count("image", strict=False) > 1:
            return self._apply_hf_processor(inputs, timing_ctx)

        return super()._cached_apply_hf_processor(inputs, timing_ctx)


class UnlimitedOCR2DecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.split(".")[-1])
        self.use_mha = True
        is_moe_layer = (
            config.n_routed_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        )
        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe and is_moe_layer
        )
        self.self_attn = DASHAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            reduce_results=not self.use_sequence_parallel_moe,
        )
        if is_moe_layer:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                apply_routed_scale_to_output=not rocm_aiter_ops.is_fused_moe_enabled(),
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


@support_torch_compile
class UnlimitedOCR2Model(DeepseekV2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Build only DASH layers; the inherited forward and MoE loader are reused.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.is_v32 = False
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            self.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: UnlimitedOCR2DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers, DeepseekV2MoE, "mlp"
        )
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()
        self.use_mha = True
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()

        def backbone_weights():
            for name, weight in weights:
                if ".dash_router." in name:
                    # Router projections are replicated, not main QKV shards.
                    default_weight_loader(params[name], weight)
                    loaded.add(name)
                else:
                    yield name, weight

        loaded.update(super().load_weights(backbone_weights()))
        return loaded


class UnlimitedOCR2TextForCausalLM(DeepseekV2ForCausalLM):
    model_cls = UnlimitedOCR2Model


@MULTIMODAL_REGISTRY.register_processor(
    UnlimitedOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    requires_raw_input_tokens = True
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        from vllm.model_executor.layers.dash_attention import prompt_page_positions

        return prompt_page_positions(
            input_tokens, self.text_config.sparse_attention_config
        ), 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        if getattr(self.text_config, "unlimited_ocr2", False):
            # Skip the unused warmup indexer and deterministic CLIP positions.
            weights = (
                (name, weight)
                for name, weight in weights
                if ".indexer." not in name
                and name != "model.vision_model.embeddings.position_ids"
            )
        return super().load_weights(weights)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if not getattr(self.text_config, "unlimited_ocr2", False):
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )

        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.dash_attention import (
            PAGE_POSITIONS_KEY,
            TOKEN_ROLES_KEY,
        )

        context = get_forward_context()
        if context.attn_metadata is not None and input_ids is None:
            raise ValueError("Unlimited-OCR-2 requires raw input token IDs")
        sp = self.text_config.sparse_attention_config
        if input_ids is not None:
            roles = (input_ids == sp["image_token_id"]).long()
            roles = roles + 2 * (input_ids == sp["page_token_id"]).long()
            context.additional_kwargs[TOKEN_ROLES_KEY] = roles
        context.additional_kwargs[PAGE_POSITIONS_KEY] = positions
        try:
            return super().forward(
                input_ids, positions[0], intermediate_tensors, inputs_embeds, **kwargs
            )
        finally:
            context.additional_kwargs.pop(TOKEN_ROLES_KEY, None)
            context.additional_kwargs.pop(PAGE_POSITIONS_KEY, None)
