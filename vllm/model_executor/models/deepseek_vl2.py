"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
from functools import cached_property, partial
from typing import (Any, Iterable, List, Literal, Mapping, Optional, Set,
                    Tuple, Type, TypedDict, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from transformers import BatchFeature, PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.inputs import InputContext
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, get_vit_attn_backend,
                    init_vllm_registered_model, maybe_prefix, flatten_bn)

logger = init_logger(__name__)


class DeepseekVL2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`
    """


class DeepseekVL2VImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


DeepseekVL2ImageInputs = Union[DeepseekVL2ImagePixelInputs, DeepseekVL2VImageEmbeddingInputs]


class MlpProjector(nn.Module):

    def __init__(self, cfg: PretrainedConfig):

        super().__init__()

        self.cfg = cfg
        assert not cfg.token_pooling, "Token pooling is not supported currently."

        if cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise NotImplementedError(f"Unsupported projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh ** 0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

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
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio,
                         padding=0)  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class DeepseekVLV2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language.": "language_model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.language_config = config.language_config

        self.vision = self._init_vision_module(
            self.vision_config, quant_config, maybe_prefix(prefix, "vision")
        )
        self.projector = MlpProjector(self.projector_config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language"),
            architectures=["DeepseekV2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors)
    
    def _init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        # TODO: refactor this vision model
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

        h = w = 384
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

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[DeepseekVL2ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return DeepseekVL2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(flatten_bn(pixel_values)))

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return DeepseekVL2VImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")
    
    def _process_image_input(self, image_input: DeepseekVL2ImageInputs) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented.")
    
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
        raise NotImplementedError("This method is not implemented.")

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