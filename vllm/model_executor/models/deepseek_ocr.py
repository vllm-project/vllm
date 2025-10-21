
"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""
import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import List, Literal, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import BatchFeature, CLIPVisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config
# from process.image_process import (
#     DeepseekOCRProcessor, count_tiles)
from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor, count_tiles
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper, init_vllm_registered_model, maybe_prefix)

from .deepencoder import build_sam_vit_b
from .deepseek_vl2 import MlpProjector
from .clip import CLIPVisionModel
# from deepencoder.clip_sdpa import build_clip_l
# from deepencoder.build_linear import MlpProjector
# from addict import Dict
# import time

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# The image token id may be various
_IMAGE_TOKEN = "<image>"


class DeepseekOCRProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(DeepseekOCRProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(self,
                             *,
                             image_width: int,
                             image_height: int,
                             cropping: bool = True) -> int:
        hf_processor = self.get_hf_processor()


        # image_size = hf_processor.image_size
        # patch_size = hf_processor.patch_size
        # downsample_ratio = hf_processor.downsample_ratio

        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if CROP_MODE:
            if image_width <= 640 and image_height <= 640:
                crop_ratio = [1, 1]
            else:
                # images_crop_raw, crop_ratio = hf_processor.dynamic_preprocess(image)

                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(image_width, image_height, image_size=IMAGE_SIZE)

                # print('===========')
                # print('crop_ratio ', crop_ratio)
                # print('============')
                
            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles >1 or num_height_tiles>1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0


        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:

        if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
            return ImageSize(width=1024*2, height=1024*2)
        return ImageSize(width=640*2, height=640*2)


class DeepseekOCRDummyInputsBuilder(
        BaseDummyInputsBuilder[DeepseekOCRProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width = max_image_size.width,
                height = max_image_size.height,
                num_images=num_images,
            )
        }


class DeepseekOCRMultiModalProcessor(
        BaseMultiModalProcessor[DeepseekOCRProcessingInfo]):
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:

        if mm_data:
            processed_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(prompt=prompt, **mm_data),
                mm_kwargs,
            )

        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=True,
                                          return_tensors="pt")

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            images_spatial_crop=MultiModalFieldConfig.batched("image"),
            images_crop=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_deepseek_vl2(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))



            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:

                size = images.get_image_size(item_idx)

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=size.width,
                    image_height=size.height,
                    cropping=CROP_MODE,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_deepseek_vl2,
            )
        ]

    # TODO(Isotr0py): Check if we still need this workaround for
    # deepseek-ocr processor.
    # def _cached_apply_hf_processor(
    #     self,
    #     prompt: str | list[int],
    #     mm_data_items: MultiModalDataItems,
    #     hf_processor_mm_kwargs: Mapping[str, object],
    #     tokenization_kwargs: Mapping[str, object],
    #     mm_uuids: MultiModalUUIDDict | None = None,
    # ) -> tuple[list[int], MultiModalKwargs, bool]:
    #     # The processor logic is different for len(images) <= 2 vs > 2
    #     # Since the processing cache assumes that the processor output is
    #     # invariant of how many images are passed per prompt, we only
    #     # perform caching for the most common case
    #     if mm_data_items.get_count("image", strict=False) > 2:
    #         # This code path corresponds to the cache being disabled
    #         return self._apply_hf_processor_main(
    #             prompt=prompt,
    #             mm_items=mm_data_items,
    #             hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #             enable_hf_prompt_update=True,
    #         )

    #     return super()._cached_apply_hf_processor(
    #         prompt=prompt,
    #         mm_data_items=mm_data_items,
    #         hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #     )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=DeepseekOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder)
class DeepseekOCRForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.layers.": "language_model.model.layers",
        "model.": "",
    },
    orig_to_new_substr={
        ".transformer.": ".encoder.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: DeepseekVLV2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # config.model_type ='deepseek_vl_v2'

        self.config = config
        self.multimodal_config = multimodal_config


        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        self.sam_model = build_sam_vit_b()
        # vit_model_cfg = dict(
        #     num_layers=24,
        #     hidden_size=1024,
        #     num_heads = 16,
        #     num_attention_heads=16,
        #     ffn_hidden_size=4096,
        #     seq_length=256,
        #     max_position_embeddings=256,
        #     use_flash_attn=False,
        #     understand_projector_stride=2,
        #     hidden_dropout = 0.0,
        #     attention_dropout = 0.0,
        #     no_persist_layer_norm = False,
        #     layernorm_epsilon = 1e-5,
        #     pre_layernorm_epsilon = 1e-5,
        #     image_size = 224,
        #     patch_size = 14,
        #     recompute_list = []
        # )
        # self.vision_model = build_clip_l()
        clip_vision_config = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            image_size=224,
            patch_size=14,
            projection_dim=512,
            layer_norm_eps=1e-5,
        )
        self.vision_model = CLIPVisionModel(config=clip_vision_config)

        n_embed = 1280
        self.projector = MlpProjector(self.projector_config)
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos
    
        # self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
        # self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
        # self.projector = torch.compile(self.projector, mode="max-autotune")

        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        if self.text_config.topk_method == "noaux_tc":
            architectures = ["DeepseekV3ForCausalLM"]
        elif not self.text_config.use_mla:
            architectures = ["DeepseekForCausalLM"]
        else:
            architectures = ["DeepseekV2ForCausalLM"]

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.text_config,
            prefix=maybe_prefix(prefix, "language"),
            architectures=architectures,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)



    def _parse_and_validate_image_input(
            self, **kwargs: object):
        
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)


        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(images_spatial_crop)}")

            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image crop. "
                                 f"Got type: {type(images_crop)}")

            return [pixel_values, images_crop, images_spatial_crop]

        raise AssertionError("This line should be unreachable.")


    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:

        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1

        images_in_this_batch = []


        # print(type(images_crop))

        # print(pixel_values.shape)


        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                # with torch.set_grad_enabled(False):
                patches = images_crop[jdx][0].to(torch.bfloat16) # batch_size = 1
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:  # if all values = 0, no crop
                    # P, C, H, W = patches.shape
                    # crop_flag = 1
                    local_features_1 = self.sam_model(patches)
                    #TODO del patches 
                    # torch.compiler.cudagraph_mark_step_begin()
                    local_features_2 = self.vision_model(patches, local_features_1)  


                    local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    local_features = self.projector(local_features)


                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2 ** 0.5)

                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )

                    global_features = global_features.view(-1, n_dim)


                    local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                    local_features = torch.cat(
                        [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                    )
                    local_features = local_features.view(-1, n_dim2)

                    global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
                
                else:
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )

                    global_features = global_features.view(-1, n_dim)

                    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def _process_image_input(
            self, image_input) -> torch.Tensor:
        

        # image_input: [pixel_values, images_crop, images_spatial_crop]
    
        pixel_values = image_input[0].to(torch.bfloat16)
        # print(image_input[1][0].shape)
        # print(type(image_input[1]))
        # exit()

        # images_crop = image_input[1].to(torch.bfloat16)
        images_crop = image_input[1]
        # images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values, images_crop = images_crop,  images_spatial_crop=images_spatial_crop)

        # local_total_time = time.time() - local_start

        # print('encoder_time: ', local_total_time)
        # exit()
        return vision_features

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
    


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        


        inputs_embeds = self.language_model.get_input_embeddings(input_ids)


        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.image_token_id)
            # print(len(multimodal_embeddings))
            # print(input_ids.shape)
            # print(type(inputs_embeds))
            # print(inputs_embeds.shape)
            
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object):

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            intermediate_tensors,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return autoloaded_weights
