# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi-K2.5 Model Implementation for vLLM.

Kimi-K2.5 extends Kimi-K2 with vision support.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.models.interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.kimi_k25_processing import (
    KimiK25DummyInputsBuilder,
    KimiK25MediaPixelInputs,
    KimiK25MultiModalProcessor,
    KimiK25ProcessingInfo,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k25 import KimiK25Config

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class KimiK25ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsEagle,
    SupportsEagle3,
):
    """Kimi-K2.5 model for conditional generation.

    Supports both image and video-chunk modalities.
    Video-chunks are temporal segments (typically 4 frames) that are
    processed with temporal pooling.
    """

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # For legacy NVFP4 checkpoint compatibility:
            # see https://github.com/vllm-project/vllm/pull/33346#issuecomment-3851475033
            "language_model.layers.": "language_model.model.layers.",
            # mm projector
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Kimi-K2.5 uses video_chunk for all media types
        if modality == "image":
            return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        elif modality == "video":
            # return a placeholder, to be replaced in the future.
            return "<|kimi_k25_video_placeholder|>"

        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK25Config = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        # Check for MoonViT config compatibility
        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()
        # Build vision tower directly with KimiK25VisionConfig
        with self._mark_tower_model(vllm_config, "vision_chunk"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if self._maybe_ignore_quant_config(quant_config) is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype
                )

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["DeepseekV2ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        if isinstance(quant_config, compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        # The batch dimension of pixel_values has been flattened into shape[0]
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, got {type(grid_thws)}"
        )
        # In some cases (e.g. with merger), grid_thws has an extra middle dimension
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        # NOTE(moyan): This forward will automatically batch the forward pass internally
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        # Validate the multimodal input keyword arguments
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_media_input(media_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.language_model.compute_logits(hidden_states)
        return logits

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
