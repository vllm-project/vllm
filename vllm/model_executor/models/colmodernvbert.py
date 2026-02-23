# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ColModernVBERT: multimodal late-interaction retrieval model.

Combines SigLIP vision encoder + ModernBERT text encoder with a pixel
shuffle connector and ColBERT-style 128-dim per-token embeddings.

Reference: https://huggingface.co/ModernVBERT/colmodernvbert-merged
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, Literal

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.colmodernvbert import ColModernVBertConfig

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .interfaces_base import default_pooling_type
from .modernbert import ModernBertEmbeddings, ModernBertLayer
from .siglip import SiglipVisionModel
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

# ---------------------------------------------------------------------------
# Connector: pixel shuffle + simple linear projection
# ---------------------------------------------------------------------------


class ColModernVBertConnector(nn.Module):
    """Pixel shuffle spatial reduction followed by a linear projection.

    Reduces the vision encoder's token count by ``factor^2`` via pixel-shuffle
    spatial rearrangement, then projects the concatenated channels to the text
    encoder's hidden size with a single bias-free linear layer.
    """

    def __init__(self, config: ColModernVBertConfig):
        super().__init__()
        self.pixel_shuffle_factor = config.pixel_shuffle_factor
        vision_hidden_size = config.vision_config.hidden_size
        input_size = vision_hidden_size * (self.pixel_shuffle_factor**2)
        output_size = config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def pixel_shuffle(self, features: torch.Tensor) -> torch.Tensor:
        """Spatial rearrangement that reduces seq length by factor^2."""
        batch_size, seq_length, hidden_size = features.shape
        height = width = int(seq_length**0.5)
        factor = self.pixel_shuffle_factor

        # Reshape to (B, H, W, C)
        features = features.view(batch_size, height, width, hidden_size)

        # Reshape to (B, H/f, f, W/f, f, C)
        features = features.view(
            batch_size, height // factor, factor, width // factor, factor, hidden_size
        )

        # Permute to (B, H/f, W/f, f, f, C)
        features = features.permute(0, 1, 3, 2, 4, 5)

        # Reshape to (B, H/f, W/f, C * f^2)
        new_hidden_size = hidden_size * (factor**2)
        features = features.reshape(
            batch_size, height // factor, width // factor, new_hidden_size
        )

        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pixel_shuffle(features)
        batch_size = features.shape[0]
        features = features.reshape(batch_size, -1, features.shape[-1])
        return self.proj(features)


# ---------------------------------------------------------------------------
# Multimodal processing
# ---------------------------------------------------------------------------


class ColModernVBertProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ColModernVBertConfig:
        return self.ctx.get_hf_config(ColModernVBertConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        config = self.get_hf_config()
        size = config.vision_config.image_size
        return ImageSize(width=size, height=size)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return self.get_hf_config().image_seq_len


class ColModernVBertDummyInputsBuilder(
    BaseDummyInputsBuilder[ColModernVBertProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image")

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class ColModernVBertMultiModalProcessor(
    BaseMultiModalProcessor[ColModernVBertProcessingInfo],
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        text_encoding = tokenizer(
            prompt,
            return_tensors="pt",
            **tok_kwargs,
        )
        result = BatchFeature(data=dict(text_encoding))

        images = mm_data.get("images")
        if images:
            from transformers import Idefics3ImageProcessor

            image_processor = Idefics3ImageProcessor.from_pretrained(
                self.info.ctx.model_config.model,
                revision=self.info.ctx.model_config.revision,
            )
            image_outputs = image_processor(
                images=images,
                do_image_splitting=False,
                return_tensors="pt",
            )
            result.update(image_outputs)

        return result

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        image_token_id = config.image_token_id
        num_tokens = config.image_seq_len

        def get_replacement(item_idx: int):
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=PromptIndexTargets.start(),
                replacement=get_replacement,
            ),
        ]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    ColModernVBertMultiModalProcessor,
    info=ColModernVBertProcessingInfo,
    dummy_inputs=ColModernVBertDummyInputsBuilder,
)
@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColModernVBertForRetrieval(nn.Module, SupportsMultiModal):
    """ColModernVBERT multimodal late-interaction retrieval model.

    Architecture:
        Image -> SiglipVisionModel -> ColModernVBertConnector
                                                   ↓
        Text  -> ModernBertEmbeddings → [merge] → ModernBertLayers → norm
                                                                      ↓
                                              custom_text_proj → L2 norm
                                                   ↓
                                          per-token 128-d embeddings
    """

    is_pooling_model = True
    supports_late_interaction: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: ColModernVBertConfig = vllm_config.model_config.hf_config
        self.config = config
        text_config = config.text_config
        quant_config = vllm_config.quant_config

        # --- Vision encoder (reuses SiglipVisionModel from siglip.py) ---
        self.vision_model = SiglipVisionModel(
            config.vision_config,
            quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        # --- Connector (pixel shuffle + linear projection) ---
        self.connector = ColModernVBertConnector(config)

        # --- Text encoder (built from ModernBERT components directly) ---
        # We build the components individually rather than wrapping
        # ``ModernBertModel`` because ``ModernBertEncoderLayer`` reads
        # ``vllm_config.model_config.hf_config`` which would be
        # ``ColModernVBertConfig``, not ``ModernBertConfig``.
        self.text_embeddings = ModernBertEmbeddings(text_config)
        self.text_layers = nn.ModuleList(
            [
                ModernBertLayer(
                    config=text_config,
                    layer_id=i,
                    prefix=f"{prefix}.text_layers.{i}",
                )
                for i in range(text_config.num_hidden_layers)
            ]
        )
        self.text_final_norm = nn.LayerNorm(
            text_config.hidden_size,
            eps=text_config.norm_eps,
            bias=text_config.norm_bias,
        )

        # --- ColBERT projection (768 -> 128, with bias) ---
        self.custom_text_proj = nn.Linear(
            text_config.hidden_size,
            config.embedding_dim,
            bias=True,
            dtype=vllm_config.model_config.head_dtype,
        )

        # --- Pooler (applies projection + L2 normalize) ---
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_token_embed(
            pooler_config,
            projector=self.custom_text_proj,
        )

    # ---- multimodal ---------------------------------------------------------

    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        # Idefics3ImageProcessor may return (batch, tiles, C, H, W);
        # flatten to (batch*tiles, C, H, W) for SiglipVisionModel.
        if pixel_values.dim() == 5:
            b, t, c, h, w = pixel_values.shape
            pixel_values = pixel_values.reshape(b * t, c, h, w)
        vision_outputs = self.vision_model(
            pixel_values.to(dtype=self.vision_model.dtype),
        )
        return self.connector(vision_outputs)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return []
        assert isinstance(pixel_values, torch.Tensor)
        image_features = self._get_image_features(pixel_values)
        return list(image_features)

    # ---- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.text_embeddings(input_ids, inputs_embeds=inputs_embeds)

        for layer in self.text_layers:
            hidden_states = layer(hidden_states, positions)

        return self.text_final_norm(hidden_states)

    # ---- weight loading -----------------------------------------------------

    # Checkpoint prefix → vLLM param prefix.
    # More-specific prefixes must appear before shorter ones.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.text_model.layers.": "text_layers.",
            "model.text_model.embeddings.": "text_embeddings.",
            "model.text_model.final_norm.": "text_final_norm.",
            "model.connector.modality_projection.": "connector.",
            "model.custom_text_proj.": "custom_text_proj.",
            "model.vision_model.": "vision_model.vision_model.",
            "model.": "",
        },
    )

    # Checkpoint names for DecoupledEmbedding parts
    _BASE_EMB = "model.text_model.embeddings.tok_embeddings.weight"
    _EXTRA_EMB = (
        "model.text_model.embeddings.tok_embeddings.additional_embedding.weight"
    )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # DecoupledEmbedding requires concatenating base + additional
        # embedding tensors before loading, so we extract them first.
        base_embedding_weight: torch.Tensor | None = None
        additional_embedding_weight: torch.Tensor | None = None
        remaining: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            if name == self._BASE_EMB:
                base_embedding_weight = tensor
            elif name == self._EXTRA_EMB:
                additional_embedding_weight = tensor
            else:
                remaining.append((name, tensor))

        # Load all non-embedding weights via AutoWeightsLoader
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(
            remaining,
            mapper=self.hf_to_vllm_mapper,
        )

        # Concatenate and load DecoupledEmbedding weights
        if base_embedding_weight is not None:
            combined = base_embedding_weight
            if additional_embedding_weight is not None:
                combined = torch.cat(
                    [base_embedding_weight, additional_embedding_weight],
                    dim=0,
                )
            param_name = "text_embeddings.tok_embeddings.weight"
            params_dict = dict(self.named_parameters())
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, combined)
                loaded_params.add(param_name)
        elif additional_embedding_weight is not None:
            raise ValueError(
                "Found 'text_model.embeddings.tok_embeddings"
                ".additional_embedding.weight' but not "
                "'text_model.embeddings.tok_embeddings.weight'"
            )

        # The pooler wraps ``custom_text_proj`` as its head projector.
        # Mark those params as loaded under the pooler path too.
        if hasattr(self, "pooler") and hasattr(self.pooler, "head"):
            head = self.pooler.head
            projector = getattr(head, "projector", None)
            if projector is not None and isinstance(projector, nn.Module):
                for pname, _ in projector.named_parameters():
                    loaded_params.add(f"pooler.head.projector.{pname}")

        return loaded_params
