# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.models.internvl import (
    BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo,
    InternVLImageEmbeddingInputs,
    InternVLImageInputs,
    InternVLImagePixelInputs,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_image_processor_from_config
from vllm.transformers_utils.processors.nemotron_vl import (
    LlamaNemotronNanoVLProcessor,
    LlamaNemotronVLEmbedImageProcessor,
    LlamaNemotronVLEmbedProcessor,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from .interfaces import (
    MultiModalEmbeddings,
    SupportsCrossEncoding,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .interfaces_base import VllmModelForPooling
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class NemotronVLProcessingInfo(BaseInternVLProcessingInfo):
    """Processing info for Nemotron VL models."""

    def get_image_processor(self, **kwargs: object):
        return cached_image_processor_from_config(self.ctx.model_config, **kwargs)

    def get_hf_processor(self, **kwargs: object) -> LlamaNemotronNanoVLProcessor:
        config = self.get_hf_config()
        vision_config = config.vision_config

        image_processor = self.get_image_processor(**kwargs)
        image_size = image_processor.image_size
        patch_size = int(kwargs.get("patch_size", vision_config.patch_size))
        downsample_ratio = float(
            kwargs.get("downsample_ratio", config.downsample_ratio)
        )
        image_seq_length = int((image_size // patch_size) ** 2 * (downsample_ratio**2))

        return self.ctx.init_processor(
            LlamaNemotronNanoVLProcessor,
            tokenizer=self.get_tokenizer(),
            image_processor=image_processor,
            image_seq_length=image_seq_length,
        )


@MULTIMODAL_REGISTRY.register_processor(
    BaseInternVLMultiModalProcessor[NemotronVLProcessingInfo],
    info=NemotronVLProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder[NemotronVLProcessingInfo],
)
class LlamaNemotronVLChatModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = self._init_vision_model(
                config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.mlp1 = self._init_mlp1(config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.get_text_config(),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.img_context_token_id = None

        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _patch_quant_config(
        self, config: PretrainedConfig, quant_config: QuantizationConfig
    ):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.get_text_config()
            llm_quant_config = getattr(text_config, "quantization_config", None)
            if (not quant_config.modules_to_not_convert) and (
                llm_quant_config is not None
            ):
                quant_config.modules_to_not_convert.append("vision_model")

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ):
        return AutoModel.from_config(
            config.vision_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def _init_mlp1(
        self,
        config: PretrainedConfig,
        vit_hidden_size: int | None = None,
        vision_projection_hidden_size: int | None = None,
    ) -> nn.Module:
        if vit_hidden_size is None:
            vit_hidden_size = config.vit_hidden_size
        if vision_projection_hidden_size is None:
            vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.get_text_config().hidden_size

        return nn.Sequential(
            nn.LayerNorm(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, bias=True
            ),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                vision_projection_hidden_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _call_vision_model(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Call vision model and return embeddings.

        Override this method in subclasses to handle different vision model
        interfaces (e.g., SigLIP vs C-RADIO).
        """
        vit_embeds = self.vision_model(x=pixel_values).features
        return vit_embeds.to(dtype=torch.bfloat16)

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1/blob/main/modeling.py#L177
        vit_embeds = self._call_vision_model(pixel_values)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> InternVLImageInputs | None:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        image_token_id = kwargs["image_token_id"]
        if isinstance(image_token_id, torch.Tensor):
            image_token_id = image_token_id.flatten().unique().item()

        assert isinstance(image_token_id, int)
        self.img_context_token_id = image_token_id

        if pixel_values_flat is not None:
            return InternVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=pixel_values_flat,
                num_patches=image_num_patches,
                resolve_bindings={
                    "h": self.config.force_image_size,
                    "w": self.config.force_image_size,
                },
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: InternVLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]
        hidden_size = self.config.get_text_config().hidden_size

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values_flat", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)

        return modalities

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        self.visual_token_mask = None

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            self._set_visual_token_mask(input_ids)

        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update({"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ## Ignore registered_buffers
        ## see https://huggingface.co/nvidia/C-RADIOv2-H/blob/main/input_conditioner.py#L28 # noqa: E501
        skip_substrs = ["norm_mean", "norm_std"]
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mlp1",
            tower_model="vision_model",
        )


# --------------------------------------------------------
# LlamaNemotronVL Embedding Model (nvidia/llama-nemotron-embed-vl-1b-v2)
# Extends LlamaNemotronVLChatModel for embedding/pooling tasks:
#   - SigLIP vision encoder (instead of C-RADIO)
#   - Bidirectional (non-causal) LLaMA language model
#   - Pooler output instead of generative logits
# --------------------------------------------------------


class LlamaNemotronVLEmbedProcessingInfo(BaseInternVLProcessingInfo):
    """Processing info for LlamaNemotronVL embedding model."""

    def get_image_processor(self, **kwargs):
        model_config = self.ctx.model_config

        config = self.get_hf_config()
        processor_config = (
            get_hf_file_to_dict(
                "processor_config.json",
                model_config.model,
                model_config.revision,
            )
            or {}
        )

        min_dynamic_patch = processor_config.get(
            "min_input_tiles",
            getattr(config, "min_dynamic_patch", 1),
        )
        max_dynamic_patch = processor_config.get(
            "max_input_tiles",
            getattr(config, "max_dynamic_patch", 1),
        )
        dynamic_image_size = processor_config.get(
            "dynamic_image_size",
            getattr(config, "dynamic_image_size", True),
        )

        kwargs.setdefault("image_size", config.force_image_size)
        kwargs.setdefault("min_dynamic_patch", min_dynamic_patch)
        kwargs.setdefault("max_dynamic_patch", max_dynamic_patch)
        kwargs.setdefault("dynamic_image_size", dynamic_image_size)
        kwargs.setdefault("use_thumbnail", True)

        return LlamaNemotronVLEmbedImageProcessor(**kwargs)

    def get_hf_processor(self, **kwargs: object) -> LlamaNemotronVLEmbedProcessor:
        config = self.get_hf_config()
        vision_config = config.vision_config

        image_processor = self.get_image_processor(**kwargs)
        image_size = image_processor.image_size
        patch_size = int(kwargs.get("patch_size", vision_config.patch_size))
        downsample_ratio = float(
            kwargs.get("downsample_ratio", config.downsample_ratio)
        )
        image_seq_length = int((image_size // patch_size) ** 2 * (downsample_ratio**2))

        return self.ctx.init_processor(
            LlamaNemotronVLEmbedProcessor,
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(**kwargs),
            image_seq_length=image_seq_length,
        )


@MULTIMODAL_REGISTRY.register_processor(
    BaseInternVLMultiModalProcessor[LlamaNemotronVLEmbedProcessingInfo],
    info=LlamaNemotronVLEmbedProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder[LlamaNemotronVLEmbedProcessingInfo],
)
class LlamaNemotronVLForEmbedding(LlamaNemotronVLChatModel, VllmModelForPooling):
    """
    LlamaNemotronVL model for embeddings.

    Inherits from LlamaNemotronVLChatModel and specializes it for embedding tasks:
    - Uses SigLIP vision encoder instead of C-RADIO
    - Uses bidirectional LLaMA (via llm_config) instead of causal LLaMA
    - Adds pooler for embedding output instead of generating logits
    """

    is_pooling_model = True

    # Weight mapping from checkpoint format to vLLM format
    # Different from parent class due to different vision model structure
    weight_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Language model mapping
            "language_model.layers.": "language_model.model.layers.",
            "language_model.embed_tokens.": "language_model.model.embed_tokens.",
            "language_model.norm.": "language_model.model.norm.",
            # Vision model mapping (SiglipVisionModel has nested vision_model)
            "vision_model.encoder.": "vision_model.vision_model.encoder.",
            "vision_model.embeddings.": "vision_model.vision_model.embeddings.",
            "vision_model.post_layernorm.": "vision_model.vision_model.post_layernorm.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config

        # Override: get img_context_token_id from config (parent sets None)
        self.img_context_token_id = getattr(config, "img_context_token_id", None)

        # Initialize pooler for embedding output
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config,
        *,
        prefix: str,
    ) -> nn.Module:
        """Override to use SigLIP instead of C-RADIO."""
        return SiglipVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=prefix,
            use_head=False,
        )

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Module:
        """Override to use different MLP structure for embedding model."""
        return super()._init_mlp1(
            config,
            vit_hidden_size=config.vision_config.hidden_size,
            vision_projection_hidden_size=config.get_text_config().hidden_size,
        )

    def _call_vision_model(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Override to handle SigLIP interface."""
        return self.vision_model(pixel_values)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Override to use different weight mapping for SigLIP."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.weight_mapper)


class LlamaNemotronVLForSequenceClassification(
    LlamaNemotronVLForEmbedding, SupportsCrossEncoding
):
    """LlamaNemotronVL model variant for sequence classification / reranking."""

    # Reranker checkpoint places base model weights under `model.*`,
    # while `score.*` remains at the top level.
    weight_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""}) | (
        LlamaNemotronVLForEmbedding.weight_mapper
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        text_config = vllm_config.model_config.hf_config.get_text_config()
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        self.score = ReplicatedLinear(
            model_config.get_hidden_size(),
            text_config.num_labels,
            bias=False,
            params_dtype=model_config.head_dtype,
            quant_config=quant_config,
            return_bias=False,
            prefix=maybe_prefix(prefix, "score"),
        )

        pooler_config = model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_seq_cls(pooler_config, classifier=self.score)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = super().load_weights(weights)

        # reranker checkpoint omits the inner LM seq-cls head
        # (`language_model.score.*`). It is unused by this outer model, but
        # the default loader expects all parameters to be initialized.
        for name, param in self.named_parameters():
            if not name.startswith("language_model.score.") or name in loaded_weights:
                continue

            if name.endswith(".weight"):
                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif name.endswith(".bias"):
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

            loaded_weights.add(name)

        return loaded_weights
