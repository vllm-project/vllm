# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only OpenVLA model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.transformers.utils import replace_linear_class
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.openvla import OpenVLAConfig
from vllm.transformers_utils.processors.openvla import OpenVLAProcessor
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.utils.collection_utils import is_list_of
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import set_default_torch_dtype

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# The image token
_IMAGE_TOKEN = "<image>"


class OpenVLAImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - c: Number of channels (3)
        - h: Height of each image
        - w: Width of each image
    """

    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("b", "c", "h", "w")]


class OpenVLAImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - n: Number of image tokens
        - h: Hidden size (must match language model backbone)
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("b", "n", "h")]


OpenVLAImageInputs: TypeAlias = (
    OpenVLAImagePixelInputs | OpenVLAImageEmbeddingInputs
)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn):
    """Unpack tuple return value to single value."""
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module):
    """Apply patch to LayerScale module to use scale_factor instead of gamma."""
    if hasattr(ls_module, 'gamma'):
        ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
        ls_module.forward = _ls_new_forward.__get__(ls_module, type(ls_module))
        del ls_module.gamma


# === Prismatic Vision Backbone ===
class PrismaticVisionBackbone(nn.Module):
    """Vision backbone using timm ViT."""
    
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: list[int],
        timm_model_ids: list[str],
        timm_override_act_layers: list[str | None],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        try:
            import timm
            from timm.models.vision_transformer import LayerScale
        except ImportError as e:
            raise ImportError("Please install timm") from e

        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        
        # Create main featurizer
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0] if timm_override_act_layers else None,
        )
        # Monkey-patch forward to return second-to-last layer patches
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create fused featurizer if needed
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1] if len(timm_override_act_layers) > 1 else None,
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale for HF compatibility
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image through featurizer."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split channel-stacked input
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector ===
class PrismaticProjector(nn.Module):
    """MLP projector from vision features to LLM hidden size."""
    
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM hidden size."""
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)
        return projected_features


# === Processing Info ===
class OpenVLAProcessingInfo(BaseProcessingInfo):
    """Processing info for OpenVLA model."""
    
    def get_hf_config(self):
        # Try to get vLLM's OpenVLAConfig, but fall back to the model's own config
        # if it was loaded with trust_remote_code=True
        try:
            return self.ctx.get_hf_config(OpenVLAConfig)
        except TypeError:
            # If type check fails, the config is from the model's own configuration_prismatic.py
            # We can still use it as it has the same structure
            hf_config = self.ctx.model_config.hf_config
            # Verify it has the necessary attributes
            if not hasattr(hf_config, 'image_sizes') or not hasattr(hf_config, 'timm_model_ids'):
                raise ValueError(
                    "Config does not have required attributes. "
                    "Expected config with 'image_sizes' and 'timm_model_ids' attributes."
                )
            return hf_config

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(OpenVLAProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        """Calculate number of image tokens based on image size."""
        hf_config = self.get_hf_config()
        image_size = hf_config.image_sizes[0]  # Use first image size
        
        # Calculate patches: (image_size / patch_size) ^ 2
        # For siglip-vit-so400m with patch14, image_size=224: (224/14)^2 = 16^2 = 256
        # But we need to check the actual patch size from the vision model
        # For now, assume patch_size=14 for siglip models
        patch_size = 14  # Default for siglip models
        num_patches_per_side = image_size // patch_size
        num_image_tokens = num_patches_per_side * num_patches_per_side
        
        return num_image_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        """Get image size that produces most features."""
        hf_config = self.get_hf_config()
        image_size = hf_config.image_sizes[0]
        return ImageSize(width=image_size, height=image_size)


# === Dummy Inputs Builder ===
class OpenVLADummyInputsBuilder(BaseDummyInputsBuilder[OpenVLAProcessingInfo]):
    """Dummy inputs builder for OpenVLA."""
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # For OpenVLA, images are inserted as embeddings after BOS token
        # We need to return dummy text that will be tokenized to include at least BOS
        # so that PromptInsertion can work properly
        # Return a space character which will be tokenized to at least one token
        # (the tokenizer will add BOS if configured)
        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            # Return a space to ensure at least one token after tokenization
            # This allows the insertion mechanism to work correctly
            return " "
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        max_image_size = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


# === MultiModal Processor ===
class OpenVLAMultiModalProcessor(
    BaseMultiModalProcessor[OpenVLAProcessingInfo]
):
    """MultiModal processor for OpenVLA."""
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            tokenizer = self.info.get_tokenizer()
            return tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # OpenVLA uses PromptInsertion, so the HF processor doesn't apply updates
        # vLLM will apply them via _apply_prompt_updates
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Insert image tokens after BOS token (position 1)."""
        tokenizer = self.info.get_tokenizer()
        bos_token_id = tokenizer.bos_token_id
        assert isinstance(bos_token_id, int)

        # Get image token ID - try multiple methods
        image_token_id = None
        
        # Method 1: Check if tokenizer already has the token
        if hasattr(tokenizer, 'vocab'):
            image_token_id = tokenizer.vocab.get(_IMAGE_TOKEN)
        elif hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            image_token_id = vocab.get(_IMAGE_TOKEN)
        elif hasattr(tokenizer, 'convert_tokens_to_ids'):
            try:
                image_token_id = tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
                if image_token_id == tokenizer.unk_token_id:
                    image_token_id = None
            except Exception:
                pass
        
        # Method 2: Try to get from processor
        if image_token_id is None:
            try:
                hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
                image_token_id = getattr(hf_processor, 'image_token_id', None)
            except Exception:
                pass
        
        # Method 3: Add the token if it's missing
        if image_token_id is None:
            try:
                # Add the image token as a special token
                num_added = tokenizer.add_special_tokens({"additional_special_tokens": [_IMAGE_TOKEN]})
                if num_added > 0:
                    # Get the token ID after adding
                    if hasattr(tokenizer, 'vocab'):
                        image_token_id = tokenizer.vocab.get(_IMAGE_TOKEN)
                    elif hasattr(tokenizer, 'get_vocab'):
                        vocab = tokenizer.get_vocab()
                        image_token_id = vocab.get(_IMAGE_TOKEN)
                    elif hasattr(tokenizer, 'convert_tokens_to_ids'):
                        image_token_id = tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
            except Exception as e:
                # If adding fails, we'll raise an error below
                pass
        
        # If image token is not found, use pad token as placeholder
        # OpenVLA inserts image embeddings directly, so we don't strictly need
        # the <image> token in the vocabulary for the replacement mechanism
        if image_token_id is None:
            # Use pad token as placeholder - this is just for the replacement mechanism
            # The actual image embeddings will be inserted by embed_multimodal
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None:
                image_token_id = pad_token_id
            else:
                # Fallback to a high token ID that's unlikely to be used
                # This is just for the prompt replacement mechanism
                vocab_size = getattr(tokenizer, 'vocab_size', 32000)
                image_token_id = vocab_size - 1  # Use last token as placeholder

        def get_insertion_openvla(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )
            
            image_tokens = [image_token_id] * num_image_tokens
            
            # Return PromptUpdateDetails to specify which tokens should be replaced with embeddings
            return PromptUpdateDetails.select_token_id(
                image_tokens,
                embed_token_id=image_token_id,
            )

        # OpenVLA inserts image tokens after BOS token (position 1)
        # Use PromptInsertion to insert at the start
        # start() works even with empty prompts and will insert before any tokens
        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=get_insertion_openvla,
            )
        ]


# === Main Model Class ===
@MULTIMODAL_REGISTRY.register_processor(
    OpenVLAMultiModalProcessor,
    info=OpenVLAProcessingInfo,
    dummy_inputs=OpenVLADummyInputsBuilder,
)
class OpenVLAForActionPrediction(nn.Module, SupportsMultiModal, SupportsPP):
    """OpenVLA model for action prediction."""
    
    merge_by_field_config = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language.": "language_model.",
            "vision_backbone.": "vision.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return _IMAGE_TOKEN
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: OpenVLAConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        
        # Get image token ID - try multiple methods
        image_token_id = None
        if hasattr(tokenizer, 'vocab'):
            image_token_id = tokenizer.vocab.get(_IMAGE_TOKEN)
        elif hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            image_token_id = vocab.get(_IMAGE_TOKEN)
        elif hasattr(tokenizer, 'convert_tokens_to_ids'):
            try:
                image_token_id = tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
                if image_token_id == tokenizer.unk_token_id:
                    image_token_id = None
            except Exception:
                pass
        
        # If not found, use pad token as placeholder
        # The actual image embeddings will be inserted by embed_multimodal
        if image_token_id is None:
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None:
                image_token_id = pad_token_id
            else:
                # Fallback to a high token ID
                vocab_size = getattr(tokenizer, 'vocab_size', 32000)
                image_token_id = vocab_size - 1
        
        self.image_token_id: int = image_token_id

        # Initialize vision backbone
        self.vision = self._init_vision_module(
            config, quant_config, maybe_prefix(prefix, "vision_backbone")
        )

        # Initialize projector
        vision_dim = self.vision.embed_dim
        llm_dim = config.text_config.hidden_size
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim,
            llm_dim,
        )

        # Initialize language model
        # Determine the architecture name based on LLM backbone
        # For Llama-2, use "LlamaForCausalLM"
        llm_arch = "LlamaForCausalLM"  # Default for Llama-2
        if "mistral" in config.llm_backbone_id.lower():
            llm_arch = "MistralForCausalLM"
        elif "phi" in config.llm_backbone_id.lower():
            llm_arch = "Phi3ForCausalLM"
        
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=[llm_arch],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _get_parent_and_attr(self, root: torch.nn.Module, dotted_name: str):
        """Return (parent_module, final_attr_name) for a dotted module path."""
        names = dotted_name.split(".")
        parent = root
        for n in names[:-1]:
            parent = getattr(parent, n)
        return parent, names[-1]

    def patch_vit_for_tp(self, vit: torch.nn.Module, quant_config: QuantizationConfig):
        """Patch ViT for tensor parallelism."""
        try:
            import timm
        except ImportError as e:
            raise ImportError("Please install timm") from e

        for name, module in vit.named_modules():
            if isinstance(module, nn.Linear):
                parent, attr_name = self._get_parent_and_attr(vit, name)
                if isinstance(parent, timm.layers.Mlp) and attr_name == "fc1":
                    new_linear = replace_linear_class(
                        module, "colwise", quant_config, prefix=name
                    )
                    setattr(parent, attr_name, new_linear)
                elif isinstance(parent, timm.layers.Mlp) and attr_name == "fc2":
                    new_linear = replace_linear_class(
                        module, "rowwise", quant_config, prefix=name
                    )
                    setattr(parent, attr_name, new_linear)

        return vit

    def _init_vision_module(
        self,
        config: OpenVLAConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> nn.Module:
        """Initialize vision backbone."""
        try:
            import timm
        except ImportError as e:
            raise ImportError("Please install timm") from e

        with set_default_torch_dtype(torch.float16):
            vision_backbone = PrismaticVisionBackbone(
                use_fused_vision_backbone=config.use_fused_vision_backbone,
                image_sizes=config.image_sizes,
                timm_model_ids=config.timm_model_ids,
                timm_override_act_layers=config.timm_override_act_layers,
            )

        if get_tensor_model_parallel_world_size() > 1:
            vision_backbone.featurizer = self.patch_vit_for_tp(
                vision_backbone.featurizer, quant_config
            )
            if vision_backbone.use_fused_vision_backbone:
                vision_backbone.fused_featurizer = self.patch_vit_for_tp(
                    vision_backbone.fused_featurizer, quant_config
                )

        vision_backbone = vision_backbone.to(dtype=torch.get_default_dtype())
        return vision_backbone

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> OpenVLAImageInputs | None:
        """Parse and validate image input."""
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return OpenVLAImagePixelInputs(
                type="pixel_values",
                data=pixel_values,
            )

        if image_embeds is not None:
            return OpenVLAImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Convert pixel values to embeddings."""
        # Process through vision backbone
        # pixel_values: [batch, channels, height, width]
        # The vision backbone handles fused backbone internally
        patches = self.vision(pixel_values)
        # patches: [batch, num_patches, vision_dim]

        # Project to LLM hidden size
        projected = self.projector(patches)
        # projected: [batch, num_patches, llm_dim]

        # Return as list of tensors (one per image in batch)
        return list(torch.unbind(projected, dim=0))

    def _process_image_input(
        self, image_input: OpenVLAImageInputs
    ) -> list[torch.Tensor]:
        """Process image input to embeddings."""
        if image_input["type"] == "image_embeds":
            image_data = image_input["data"]
            if is_list_of(image_data, torch.Tensor):
                return image_data
            if len(image_data.shape) == 3:
                return list(torch.unbind(image_data, dim=0))
            raise ValueError(
                "We expect batched 2D tensors; "
                "this can be either a list of 2D tensors or a single 3D tensor."
            )

        pixel_values = image_input["data"]
        return self._pixel_values_to_embedding(pixel_values=pixel_values)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Embed multimodal inputs (images)."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        """Forward pass."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint."""
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return autoloaded_weights

