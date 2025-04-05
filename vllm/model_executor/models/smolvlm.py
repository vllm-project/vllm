# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from typing import Dict, Literal, Optional, Set, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import (BatchFeature, SmolVLMConfig, SmolVLMImageProcessor,
                          SmolVLMProcessor)

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.parse import ImageProcessorItems
# yapf conflicts with isort for this block
# yapf: disable
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems,
                                        MultiModalFieldConfig,
                                        PromptReplacement, PromptUpdate)
# yapf: enable
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

# yapf: disable
from .idefics2_vision_model import (
    Idefics2VisionTransformer as SmolVLMVisionTransformer)
from .idefics3 import Idefics3ProcessingInfo
# yapf: enable
from .interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal
from .llama import LlamaModel
from .utils import (AutoWeightsLoader, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import scatter_patch_features, select_patch_features


class SmolVLMImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_patches, 
             num_channels, height, width)`
    """
    pixel_attention_mask: torch.Tensor

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""

    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond
    to patch tokens.

    Shape: `(batch_size * num_images, num_embeds)`
    """


class SmolVLMImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`
    `hidden_size` must match the hidden size of language model backbone.
    """

    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond
    to patch tokens.

    Shape: `(batch_size * num_images, num_embeds)`
    """


ImageInputs = Union[SmolVLMImagePixelInputs, SmolVLMImageEmbeddingInputs]


class SmolVLMProcessingInfo(Idefics3ProcessingInfo):

    def get_hf_processor(
        self,
        *,
        size: Optional[Dict[str, int]] = None,
        **kwargs: object,
    ) -> SmolVLMProcessor:
        if size is not None:
            kwargs["size"] = size

        return self.ctx.get_hf_processor(SmolVLMProcessor, **kwargs)

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[SmolVLMProcessor],
    ) -> str:
        if processor is None:
            processor = self.get_hf_processor()
        image_token = processor.image_token
        fake_image_token = processor.fake_image_token
        global_img_token = processor.global_image_token
        image_seq_len = processor.image_seq_len
        grid_placeholder = "<row_{n_h}_col_{n_w}>"

        p_img = image_token * image_seq_len
        global_img_placeholder = fake_image_token + global_img_token + p_img
        tile_img_placeholder = fake_image_token + grid_placeholder + p_img

        grid_w, grid_h = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )
        if grid_w == 0 and grid_h == 0:
            return global_img_placeholder + fake_image_token

        tiles_placeholder = list[str]()
        for i in range(grid_h):
            for j in range(grid_w):
                placeholder_per_tile = tile_img_placeholder.format(n_h=i + 1,
                                                                   n_w=j + 1)
                tiles_placeholder.append(placeholder_per_tile)
                # Add line break if it is the last tile in the row
                if j == grid_w - 1:
                    tiles_placeholder.append("\n")
        return "".join([
            *tiles_placeholder,
            "\n",
            global_img_placeholder,
            fake_image_token,
        ])


class SmolVLMDummyInputsBuilder(BaseDummyInputsBuilder[SmolVLMProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor()
        image_processor: SmolVLMImageProcessor = hf_processor.image_processor
        longest_edge = image_processor.max_image_size['longest_edge']
        image_token = hf_processor.image_token

        mm_data = {
            "image":
            self._get_dummy_images(width=longest_edge,
                                   height=longest_edge,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )


class SmolVLMMultiModalProcessor(BaseMultiModalProcessor[SmolVLMProcessingInfo]
                                 ):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not (images := mm_data.get("images", [])):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
        )

        parsed_images = (self._get_data_parser().parse_mm_data({
            "image": images
        }).get_items("image", ImageProcessorItems))
        image_sizes = [
            parsed_images.get_image_size(i) for i in range(len(parsed_images))
        ]
        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        image_repl_features = [
            self.info.get_image_repl(image_width=size.width,
                                     image_height=size.height,
                                     processor=hf_processor)
            for size in image_sizes
        ]

        tokenizer = self.info.get_tokenizer()
        image_repls_feature_tokens = [
            tokenizer.encode(image_repl, add_special_tokens=False)
            for image_repl in image_repl_features
        ]

        vocab = tokenizer.get_vocab()
        image_token_id = vocab[hf_processor.image_token]

        embed_is_patch = [
            torch.tensor(image_repl_tokens) == image_token_id
            for image_repl_tokens in image_repls_feature_tokens
        ]
        processed_outputs["embed_is_patch"] = embed_is_patch

        num_patches = [
            self.info.get_num_patches(
                image_width=size.width,
                image_height=size.height,
                processor=hf_processor,
            ) for size in image_sizes
        ]
        processed_outputs["num_patches"] = torch.tensor(num_patches)

        # Remove the extra batch dimension
        processed_outputs["pixel_values"].squeeze_(0)
        processed_outputs["pixel_attention_mask"].squeeze_(0)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            pixel_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            image_embeds=MultiModalFieldConfig.batched("image"),
            num_patches=MultiModalFieldConfig.batched("image"),
            embed_is_patch=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token

        def get_replacement_smolvlm(item_idx: int) -> str:
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)

            return self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_smolvlm,
            )
        ]


class SmolVLMSimpleMLP(nn.Module):

    def __init__(
        self,
        config: SmolVLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**
                                                         2)
        output_size = config.text_config.hidden_size
        self.proj = ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj(x)
        return out


class SmolVLMConnector(nn.Module):

    def __init__(
        self,
        config: SmolVLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = SmolVLMSimpleMLP(
            config,
            quant_config,
            prefix=maybe_prefix(prefix, "modality_projection"),
        )

    def pixel_shuffle(self,
                      x: torch.Tensor,
                      scale_factor: int = 2) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor),
                   embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)),
                      embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.pixel_shuffle(image_hidden_states,
                                                 self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class SmolVLMModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: SmolVLMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = SmolVLMVisionTransformer(config.vision_config,
                                                     quant_config=quant_config,
                                                     prefix=maybe_prefix(
                                                         prefix,
                                                         "vision_model"))
        self.connector = SmolVLMConnector(
            config,
            quant_config,
            prefix=maybe_prefix(prefix, "connector"),
        )
        self.text_model = LlamaModel(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "text_model"),
        )

        self.image_seq_len = int(
            ((config.vision_config.image_size //
              config.vision_config.patch_size)**2) / (config.scale_factor**2))
        self.image_token_id = self.config.image_token_id

    def image_pixels_to_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        pixel_values = pixel_values.to(
            dtype=self.vision_model.embeddings.patch_embedding.weight.dtype
        )  # fp16 compatibility

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(
            dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        # Remove padding images from the mask
        pixel_attention_mask = pixel_attention_mask[
            real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1,
                                                      size=patch_size,
                                                      step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2,
                                                 size=patch_size,
                                                 step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )

        return image_hidden_states

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        hidden_states = self.text_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(SmolVLMMultiModalProcessor,
                                        info=SmolVLMProcessingInfo,
                                        dummy_inputs=SmolVLMDummyInputsBuilder)
class SmolVLMForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.model = SmolVLMModel(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))
        self.image_token_id = self.config.image_token_id

        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
        )
        if self.config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.text_model.wte.weight
        self.logits_processor = LogitsProcessor(config.text_config.vocab_size)
        self.sampler = get_sampler()

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        embed_is_patch = kwargs.pop("embed_is_patch")
        if not isinstance(embed_is_patch, (torch.Tensor, list)):
            raise ValueError("Incorrect type of embed_is_patch. "
                             f"Got type: {type(embed_is_patch)}")

        embed_is_patch = flatten_bn(embed_is_patch)

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return SmolVLMImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
                embed_is_patch=embed_is_patch,
            )

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            pixel_attention_mask = kwargs.pop("pixel_attention_mask")
            if not isinstance(pixel_attention_mask, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel_attention_mask. "
                                 f"Got type: {type(pixel_attention_mask)}")

            num_patches = kwargs.pop("num_patches")
            if not isinstance(num_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of num_patches. "
                                 f"Got type: {type(num_patches)}")

            pixel_values = flatten_bn(pixel_values, concat=True)
            pixel_attention_mask = flatten_bn(pixel_attention_mask,
                                              concat=True)
            num_patches = flatten_bn(num_patches, concat=True)

            return SmolVLMImagePixelInputs(
                type="pixel_values",
                pixel_values=self._validate_pixel_values(pixel_values),
                pixel_attention_mask=pixel_attention_mask,
                num_patches=num_patches,
                embed_is_patch=embed_is_patch,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_pixels(self,
                              inputs: SmolVLMImagePixelInputs) -> torch.Tensor:
        pixel_values = inputs["pixel_values"]
        pixel_attention_mask = inputs["pixel_attention_mask"]

        return self.model.image_pixels_to_features(
            pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

    def _process_image_input(
        self,
        image_input: ImageInputs,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        image_features = self._process_image_pixels(image_input)
        image_features = self.model.connector(image_features)

        num_patches = image_input["num_patches"]
        return [
            e.flatten(0, 1) for e in image_features.split(num_patches.tolist())
        ]

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        image_features = self._process_image_input(image_input)

        return scatter_patch_features(
            image_features,
            image_input["embed_is_patch"],
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                select_patch_features(multimodal_embeddings),
                self.config.image_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.model.text_model(input_ids,
                                              positions,
                                              intermediate_tensors,
                                              inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.text_model",
            connector="model.connector",
            tower_model="model.vision_model")
