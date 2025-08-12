# adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/faf18023f24b962b32d9f0a2d89e402a8d383a78/deepseek_vl2/models/modeling_deepseek_vl_v2.py
"""Inference-only Deepseek-VL2 model compatible with HuggingFace weights."""
import math
from functools import cached_property, partial
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoProcessor, BatchFeature, ProcessorMixin

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_vl2 import (DeepseekVLV2Config,
                                                          MlpProjectorConfig,
                                                          VisionEncoderConfig)
from vllm.utils import is_list_of

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)

# The image token id may be various
_IMAGE_TOKEN = "<image>"


class DeepseekVL2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`
    """
    images_spatial_crop: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`
    """


class DeepseekVL2VImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


DeepseekVL2ImageInputs = Union[DeepseekVL2ImagePixelInputs,
                               DeepseekVL2VImageEmbeddingInputs]


class MlpProjector(nn.Module):

    def __init__(self, cfg: MlpProjectorConfig):

        super().__init__()

        self.cfg = cfg
        assert not cfg.token_pooling, (
            "Token pooling is not supported currently.")

        if cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio *
                    cfg.downsample_ratio, cfg.n_embed * mlp_ratio)
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio,
                              cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise NotImplementedError(
                f"Unsupported projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(self, x):
        bs, hw, input_dim = x.shape
        h = w = int((hw)**0.5)
        """compute padding"""
        if h % self.cfg.downsample_ratio:
            pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
        else:
            pad = 0
        x = x.reshape(bs, h, w, input_dim)
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)
        """4 to 1 concat"""
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = F.unfold(x,
                     kernel_size=self.cfg.downsample_ratio,
                     stride=self.cfg.downsample_ratio,
                     padding=0)  # B, C*4, HW // 4
        x = x.permute(0, 2, 1)

        return self.layers(x)


class DeepseekVL2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self) -> ProcessorMixin:
        # TODO(Isotr0py): we should get rid of dependency on deepseek_vl2
        # in the future, because it's flasky and lack of maintenance.
        try:
            from deepseek_vl2.models.processing_deepseek_vl_v2 import (
                DeepseekVLV2Processor, select_best_resolution)
            AutoProcessor.register("DeepseekVLV2Processor",
                                   DeepseekVLV2Processor)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "You need to `pip install "
                "git+https://github.com/deepseek-ai/DeepSeek-VL2.git` "
                "to use this model") from exc

        processor = self.ctx.get_hf_processor(DeepseekVLV2Processor)
        processor.select_best_resolution = partial(
            select_best_resolution,
            candidate_resolutions=processor.candidate_resolutions)
        return processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(self, *, image_width: int,
                             image_height: int) -> int:
        hf_processor = self.get_hf_processor()
        image_size = hf_processor.image_size
        patch_size = hf_processor.patch_size
        downsample_ratio = hf_processor.downsample_ratio

        best_width, best_height = hf_processor.select_best_resolution(
            (image_width, image_height))

        num_width_tiles, num_height_tiles = (best_width // image_size,
                                             best_height // image_size)
        h = w = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        local_views_tokens = (num_height_tiles * h) * (num_width_tiles * w + 1)
        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()
        candidate_resolutions = hf_config.candidate_resolutions
        height, width = max(candidate_resolutions,
                            key=lambda x: self.get_num_image_tokens(
                                image_width=x[1], image_height=x[0]))
        return ImageSize(width=width, height=height)

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        max_image_size = self.get_image_size_with_most_features()
        max_image_tokens = self.get_num_image_tokens(
            image_height=max_image_size.height,
            image_width=max_image_size.width)

        return {"image": max_image_tokens}


class DeepseekVL2DummyInputsBuilder(
        BaseDummyInputsBuilder[DeepseekVL2ProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token

        max_image_size = self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=max_image_size.width,
                                   height=max_image_size.height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )


class DeepseekVL2MultiModalProcessor(
        BaseMultiModalProcessor[DeepseekVL2ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(prompt=prompt, **mm_data),
                mm_kwargs,
            )

            # Deepseek-vl2 processor don't return BatchFeature,
            # we need to manually create it
            processed_outputs = dict(input_ids=outputs["input_ids"])
            processed_outputs = BatchFeature(data=dict(processed_outputs),
                                             tensor_type="pt")

            # Remove batch dimension from processor outputs,
            # because we will try batch to create NestedTensors
            target_dtype = self.info.ctx.model_config.dtype
            pixel_values = outputs["images"].to(target_dtype).squeeze(0)
            images_spatial_crop = outputs["images_spatial_crop"].squeeze(0)
            patches_per_image = [
                x.prod().item() + 1 for x in images_spatial_crop
            ]

            # Rename `images` -> `pixel_values` to avoid confusion
            processed_outputs["pixel_values"] = list(
                pixel_values.split(patches_per_image))
            processed_outputs["images_spatial_crop"] = images_spatial_crop
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
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_processor = self.info.get_hf_processor()
        image_token_id: int = hf_processor.image_token_id

        def get_replacement_deepseek_vl2(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_deepseek_vl2,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekVL2MultiModalProcessor,
    info=DeepseekVL2ProcessingInfo,
    dummy_inputs=DeepseekVL2DummyInputsBuilder)
class DeepseekVLV2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "language.": "language_model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: DeepseekVLV2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        model_config = vllm_config.model_config
        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            tokenizer_revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
        self.image_token_id = tokenizer.vocab.get(_IMAGE_TOKEN)

        self.vision = self._init_vision_module(self.vision_config,
                                               quant_config,
                                               maybe_prefix(prefix, "vision"))

        self.projector = MlpProjector(self.projector_config)
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(
            torch.tensor(self.projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(
                torch.randn(self.projector_config.n_embed) * embed_std)
            # This is a typo in original implementation
            self.view_seperator = nn.Parameter(
                torch.randn(self.projector_config.n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.text_config,
            prefix=maybe_prefix(prefix, "language"),
            architectures=["DeepseekV3ForCausalLM"]
            if self.text_config.topk_method == "noaux_tc" else
            ["DeepseekV2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _init_vision_module(
        self,
        vision_config: VisionEncoderConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        # TODO: refactor vision model through timm wrapper from transformers
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm") from ImportError

        with set_default_torch_dtype(torch.float16):
            model = timm.create_model(
                "vit_so400m_patch14_siglip_384.webli",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )

        model = model.to(dtype=torch.get_default_dtype())
        return model

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_images_spatial_crop(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        expected_dims = 2

        def _validate_shape(d: torch.Tensor):
            actual_dims = d.size(-1)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[DeepseekVL2ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(images_spatial_crop)}")

            return DeepseekVL2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(flatten_bn(pixel_values)),
                images_spatial_crop=self._validate_images_spatial_crop(
                    flatten_bn(images_spatial_crop, concat=True)))

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return DeepseekVL2VImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _pixel_values_to_embedding(
        self,
        pixel_values: NestedTensors,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:
        # Pixel_values: n_image * batch_size * [patch_per_img, 3, height, width]
        total_tiles = [x for x in pixel_values]

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision.forward_features(total_tiles)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)

        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        vision_embeddings = []
        for jdx in range(images_spatial_crop.size(0)):
            # extra global & local features
            num_width_tiles, num_height_tiles = images_spatial_crop[jdx]
            if num_width_tiles == 0 or num_height_tiles == 0:
                break
            num_tiles_in_image = num_width_tiles * num_height_tiles

            # [hw, D]
            global_features = images_embeds[tile_index]

            # [num_height_tiles * num_width_tiles, hw, D]
            local_features = images_embeds[tile_index + 1:tile_index + 1 +
                                           num_tiles_in_image]
            tile_index += num_tiles_in_image + 1

            # format global and local features
            # ----------------- global view add newline -----------------
            # [hw, D] -> [h, w, D]
            global_features = global_features.view(h, w, n_dim)

            # [D]     -> [h, 1, D]
            new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)

            # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
            global_features = torch.cat([global_features, new_lines_in_global],
                                        dim=1)

            # [h, w + 1, D] -> [h * (w + 1), D]
            global_features = global_features.view(-1, n_dim)

            # ----------------- local view add newline -----------------
            # [num_height_tiles * num_width_tiles, h * w, D] ->
            # [num_height_tiles * h, num_width_tiles * w, D]
            local_features = rearrange(local_features,
                                       "(th tw) (h w) d -> (th h) (tw w) d",
                                       th=num_height_tiles,
                                       tw=num_width_tiles,
                                       h=h,
                                       w=w)

            # [D] -> [num_height_tiles * h, 1, D]
            new_lines_in_local = repeat(self.image_newline,
                                        "d -> (th h) 1 d",
                                        th=num_height_tiles,
                                        h=h)

            # [num_height_tiles * h, num_width_tiles * w + 1, D]
            local_features = torch.cat([local_features, new_lines_in_local],
                                       dim=1)

            # [num_height_tiles * h, num_width_tiles * w + 1, D]
            #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
            local_features = local_features.view(-1, n_dim)

            # merge global and local tiles
            if self.global_view_pos == "head":
                global_local_features = torch.cat([
                    global_features,
                    self.view_seperator[None, :],
                    local_features,
                ])
            else:
                global_local_features = torch.cat([
                    local_features,
                    self.view_seperator[None, :],
                    global_features,
                ])

            vision_embeddings.append(global_local_features)
        return vision_embeddings

    def _process_image_input(
            self, image_input: DeepseekVL2ImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            image_data = image_input["data"]
            if is_list_of(image_data, torch.Tensor):
                # it's already a list of tensors
                return image_data
            if len(image_data.shape) == 3:
                # 3D tensor
                return list(torch.unbind(image_data, dim=0))
            raise ValueError(
                "We expect batched 2D tensors;"
                "this can be either a list of 2D tensors or a single 3D tensor."
            )

        pixel_values = image_input["data"]
        images_spatial_crop = image_input["images_spatial_crop"]

        return self._pixel_values_to_embedding(
            pixel_values=pixel_values, images_spatial_crop=images_spatial_crop)

    def get_multimodal_embeddings(self, **kwargs: object) -> torch.Tensor:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.image_token_id)
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object):

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            attn_metadata,
                                            intermediate_tensors,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights,
                                                 mapper=self.hf_to_vllm_mapper)
        return autoloaded_weights
