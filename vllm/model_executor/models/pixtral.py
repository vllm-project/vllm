# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cached_property
from typing import List, Literal, Optional, Set, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mistral_common.protocol.instruct.messages import ImageChunk
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from transformers import BatchFeature, PixtralVisionConfig, TensorType
from transformers.image_utils import ImageInput
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens)
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding, apply_rotary_pos_emb, position_ids_in_meshgrid)
from transformers.tokenization_utils_base import TextInput

from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import (MistralTokenizer,
                                               cached_tokenizer_from_config)

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (flatten_bn, init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import VisionEncoderInfo, resolve_visual_encoder_outputs

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False


class PixtralImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]

    images: list[torch.Tensor]
    """
    Shape: `(batch_size, num_channels, image_width, image_height)`

    The result of stacking :attr:`ImageEncoding.tokens` from each prompt.
    """

    num_image_tokens: list[int]
    """
    The number of image tokens matching :attr:`VisionEncoderArgs.image_token_id`
    in :attr:`ImageEncoding.tokens` for each image in the batch.

    This is used to split the embeddings which has the first two dimensions
    flattened.
    """


class PixtralProcessorAdapter:
    """
    Provide a HF-compatible interface for
    :class:`mistral_common.tokens.tokenizers.multimodal.ImageEncoder`.
    """

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer

    @property
    def image_processor(self) -> ImageEncoder:
        image_encoder = self.tokenizer.instruct.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    @cached_property
    def image_break_id(self) -> int:
        return self.image_processor.special_ids.img_break

    @cached_property
    def image_token_id(self) -> int:
        return self.image_processor.special_ids.img

    @cached_property
    def image_end_id(self) -> int:
        return self.image_processor.special_ids.img_end

    @cached_property
    def image_size(self) -> int:
        return self.image_processor.mm_config.max_image_size

    @cached_property
    def patch_size(self) -> int:
        return self.image_processor.mm_config.image_patch_size

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if not images:
            input_ids = self.tokenizer(text).input_ids

            return BatchFeature(
                {"input_ids": input_ids},
                tensor_type=return_tensors,
            )

        # Allow dummy text, which is used for profiling as well as token inputs
        if any(len(t) > 0 for t in text):
            raise ValueError(
                "You've passed text inputs instead of token inputs. "
                "Make sure to process your input via `mistral_common`'s "
                "tokenizer or pass a chat completion request. "
                "For more info, see: "
                "https://github.com/vllm-project/vllm/issues/8411.")

        images_flat = []
        num_crops_hw = []
        image_tokens_flat = []
        num_images_tokens = []

        patch_size = self.patch_size
        image_token_id = self.image_token_id

        for image in images:
            image_inputs = self.image_processor(ImageChunk(image=image))

            C, H, W = image_inputs.image.shape
            grid_h, grid_w = H // patch_size, W // patch_size
            image_flat = (image_inputs.image.reshape(
                C, grid_h, patch_size, grid_w,
                patch_size).transpose(1, 3, 0, 2,
                                      4).reshape(grid_h * grid_w, C,
                                                 patch_size, patch_size))

            num_image_tokens = image_inputs.tokens.count(image_token_id)
            assert num_image_tokens == grid_h * grid_w

            images_flat.append(image_flat)
            num_crops_hw.append([grid_h, grid_w])
            image_tokens_flat.extend(image_inputs.tokens)
            num_images_tokens.append(num_image_tokens)

        return BatchFeature(
            {
                "input_ids": [image_tokens_flat] * len(text),
                "images_flat": np.concatenate(images_flat),
                "num_crops_hw": np.array(num_crops_hw),
                "num_image_tokens": num_images_tokens,
            },
            tensor_type=return_tensors,
        )


class PixtralProcessingInfo(BaseProcessingInfo):

    def get_tokenizer(self) -> MistralTokenizer:
        tokenizer = cached_tokenizer_from_config(self.ctx.model_config)
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("This model requires `--tokenizer-mode mistral`")

        return tokenizer

    def get_hf_processor(self) -> PixtralProcessorAdapter:
        return PixtralProcessorAdapter(self.get_tokenizer())

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_vision_config(
        self,
        processor: Optional[PixtralProcessorAdapter] = None,
    ):
        if processor is None:
            processor = self.get_hf_processor()

        return PixtralVisionConfig(
            image_size=processor.image_size,
            patch_size=processor.patch_size,
        )

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[PixtralProcessorAdapter] = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        hf_vision_config = self.get_vision_config(processor)
        hf_encoder_info = PixtralEncoderInfo(hf_vision_config)

        return hf_encoder_info.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_hf_processor().image_processor
        max_image_size = image_processor.mm_config.max_image_size

        return ImageSize(width=max_image_size, height=max_image_size)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )


class PixtralDummyInputsBuilder(BaseDummyInputsBuilder[PixtralProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="",
            mm_data=mm_data,
        )


class PixtralMultiModalProcessor(BaseMultiModalProcessor[PixtralProcessingInfo]
                                 ):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_crops_hw = hf_inputs.get("num_crops_hw", torch.empty((0, 2)))

        return dict(
            images_flat=MultiModalFieldConfig.flat_from_sizes(
                "image", num_crops_hw.prod(-1)),
            num_crops_hw=MultiModalFieldConfig.batched("image"),
            num_image_tokens=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_break_id = processor.image_break_id
        image_token_id = processor.image_token_id
        image_end_id = processor.image_end_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = get_pixtral_image_feature_grid_size(
                self.info.get_vision_config(processor),
                image_width=image_size.width,
                image_height=image_size.height,
            )

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            return tokens

        return [
            PromptReplacement(
                modality="image",
                target="",  # Never match the prompt (see below note)
                replacement=get_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        prompt_ids, mm_kwargs, _ = super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        # NOTE: The tokens are already inserted by the chat template
        return prompt_ids, mm_kwargs, True


@MULTIMODAL_REGISTRY.register_processor(PixtralMultiModalProcessor,
                                        info=PixtralProcessingInfo,
                                        dummy_inputs=PixtralDummyInputsBuilder)
class PixtralForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)}
        vision_args = {
            key: value
            for key, value in self.config.vision_config.to_dict().items()
            if key in dataclass_fields
        }

        self.vision_args = VisionEncoderArgs(**vision_args)

        # init MistralForCausalLM
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.vision_encoder = VisionTransformer(self.vision_args)
        self.vision_language_adapter = VisionLanguageAdapter(
            self.vision_args, dim=config.text_config.hidden_size)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def get_multimodal_embeddings(
        self, **kwargs
    ) -> Union[list[torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...]]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.vision_args.image_token_id,
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
        """Run forward pass for pixtral."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[PixtralImagePixelInputs]:
        images_flat = kwargs.pop("images_flat", None)
        if images_flat is None:
            return None

        num_crops_hw = kwargs.pop("num_crops_hw")
        num_image_tokens = kwargs.pop("num_image_tokens")

        if not isinstance(images_flat, (torch.Tensor, list)):
            raise ValueError("Incorrect type of images. "
                             f"Got type: {type(images_flat)}")
        if not isinstance(num_crops_hw, torch.Tensor):
            raise ValueError("Incorrect type of num_crops_hw. "
                             f"Got type: {type(num_crops_hw)}")
        if not isinstance(num_image_tokens, torch.Tensor):
            raise ValueError("Incorrect type of num_image_tokens. "
                             f"Got type: {type(num_image_tokens)}")

        images_flat = flatten_bn(images_flat, concat=True)
        num_crops_hw = flatten_bn(num_crops_hw)
        num_image_tokens = flatten_bn(num_image_tokens)

        assert images_flat.dim() == 4, images_flat.shape
        assert num_crops_hw.dim() == 2, num_crops_hw.shape
        assert num_image_tokens.dim() == 1, num_image_tokens.shape

        images = list[torch.Tensor]()
        for image, (crop_h, crop_w) in zip(
                torch.split(images_flat,
                            num_crops_hw.prod(-1).tolist()),
                num_crops_hw.tolist(),
        ):
            _, C, h, w = image.shape

            images.append(
                image.reshape(crop_h, crop_w, C, h,
                              w).permute(2, 0, 3, 1,
                                         4).reshape(C, crop_h * h, crop_w * w))

        return PixtralImagePixelInputs(
            type="pixel_values",
            images=images,
            num_image_tokens=num_image_tokens.tolist(),
        )

    def _process_image_input(
            self, image_input: PixtralImagePixelInputs) -> torch.Tensor:
        images = image_input["images"]
        num_image_tokens = image_input["num_image_tokens"]

        image_embeds_flat = self.vision_language_adapter(
            self.vision_encoder(images))

        return image_embeds_flat.split(num_image_tokens, dim=0)

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        def is_vision_encoder_weights(weight: Tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_encoder")

        def is_vision_lang_adapter_weights(weight: Tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_language_adapter")

        # Get references to parameters for direct loading
        vision_encoder_dict = dict(self.vision_encoder.named_parameters())
        vision_lang_adapter_dict = dict(
            self.vision_language_adapter.named_parameters())

        def llm_weights_generator():
            # Single pass over weights
            for name, w in weights:
                if is_vision_encoder_weights((name, w)):
                    # Load vision encoder weights directly
                    trimmed_name = '.'.join(name.split(".")[1:])
                    param = vision_encoder_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_vision_lang_adapter_weights((name, w)):
                    # Load vision-language adapter weights directly
                    trimmed_name = '.'.join(name.split(".")[1:])
                    param = vision_lang_adapter_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                else:
                    # LLM weights: yield them to be loaded
                    # by language_model.load_weights
                    yield (name, w)

        # Now we call the language model load with the generator
        self.language_model.load_weights(llm_weights_generator())


# Vision encoder
@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float  # for rope-2D
    image_token_id: int
    adapter_bias: bool = True


def _reshape_for_broadcast(freqs_cis: torch.Tensor,
                           x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    freqs_cis: 2D complex tensor of shape (height, width, dim // 2)
        to be indexed by (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)


def apply_rotary_emb_vit(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    assert freqs_cis.dtype == torch.complex64
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        assert args.intermediate_size is not None
        self.w1 = nn.Linear(args.hidden_size,
                            args.intermediate_size,
                            bias=False)
        self.w2 = nn.Linear(args.intermediate_size,
                            args.hidden_size,
                            bias=False)
        self.w3 = nn.Linear(args.hidden_size,
                            args.intermediate_size,
                            bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        assert not args.hidden_size % args.num_attention_heads
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wv = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wo = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        return self.wo(out)


class TransformerBlock(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x),
                                   mask=mask,
                                   freqs_cis=freqs_cis)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x


def position_meshgrid(patch_embeds_list: List[torch.Tensor], ) -> torch.Tensor:
    positions = torch.cat([
        torch.stack(
            torch.meshgrid(
                torch.arange(p.shape[-2]),
                torch.arange(p.shape[-1]),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2) for p in patch_embeds_list
    ])
    return positions


class VisionTransformer(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        self.patch_conv = nn.Conv2d(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = Transformer(args)

        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @property
    def max_patches_per_side(self) -> int:
        return self.args.image_size // self.args.patch_size

    @property
    def device(self) -> torch.types.Device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes, 
                each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for 
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in images
        ]

        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        # pass through Transformer with a block diagonal mask delimiting images
        if USE_XFORMERS_OPS:
            mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], )
        else:
            raise ImportError("Xformers is required for Pixtral inference "
                              "with the Mistral format")
        out = self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)

        # remove batch dimension of the single sequence
        return out.squeeze(0)


class VisionLanguageAdapter(nn.Module):

    def __init__(self, args: VisionEncoderArgs, dim: int):
        super().__init__()
        assert isinstance(args, VisionEncoderArgs)
        self.w_in = nn.Linear(
            args.hidden_size,
            dim,
            bias=args.adapter_bias,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=args.adapter_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


#### HF Transformers version of Pixtral ####
# Based off https://github.com/huggingface/transformers/blob/d7950bff82b18c823193d17d72188c5e46d06c83/src/transformers/models/pixtral/modeling_pixtral.py
# This model follows the Llava family, meaning image embeddings are placed
# instead of the `[IMG]` token placeholders.
# The model uses [`PixtralVisionModel`] for its vision encoder,
# and [`MistralForCausalLM`] for its language decoder.


# Adapted from transformers.models.pixtral.image_processing_pixtral.get_resize_output_image_size # noqa: E501
# https://github.com/huggingface/transformers/blob/2bd4d5897dc73e8b172832070a6f9e567a0df017/src/transformers/models/pixtral/image_processing_pixtral.py#L180
def get_pixtral_image_feature_grid_size(
    hf_config: PixtralVisionConfig,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    max_width = max_height = hf_config.image_size
    patch_width = patch_height = hf_config.patch_size

    ratio = max(image_width / max_width, image_height / max_height)

    if ratio > 1:
        image_width = int(math.ceil(image_width / ratio))
        image_height = int(math.ceil(image_height / ratio))

    nrows, ncols = _get_pixtral_hf_num_image_tokens(
        (image_height, image_width),
        (patch_height, patch_width),
    )  # type: ignore

    return ncols, nrows


class PixtralEncoderInfo(VisionEncoderInfo[PixtralVisionConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        grid_length = self.get_patch_grid_length()

        # Consider the image_break_token
        return (grid_length + 1) * grid_length

    def get_max_image_tokens(self) -> int:
        grid_length = self.get_patch_grid_length()

        # Consider the image_break_token
        return (grid_length + 1) * grid_length

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()

        # Since interpolation is applied, the image size need not be divisible
        # assert image_size % patch_size == 0
        return image_size // patch_size


class PixtralHFMLP(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert config.intermediate_size is not None
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(input_size=config.intermediate_size,
                                           output_size=config.hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.act_and_mul = get_act_and_mul_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_and_mul(gate_up)
        x, _ = self.down_proj(x)
        return x


class PixtralHFAttention(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        assert not config.hidden_size % config.num_attention_heads
        self.total_num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.n_heads = divide(config.num_attention_heads, tp_size)
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        assert self.total_num_heads * self.head_dim == config.hidden_size
        self.o_proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, patches, _ = hidden_states.size()

        qkv_states, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv_states.chunk(3, dim=-1)

        # Transpose q and k to apply HF's Rotary Position Embedding
        q = q.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, patches, self.n_heads, self.head_dim)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)

        if USE_XFORMERS_OPS:
            # Transpose q and k back for attention
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()

            out = xops.memory_efficient_attention(q,
                                                  k,
                                                  v,
                                                  attn_bias=attention_mask)
        else:
            v = v.transpose(1, 2)
            out = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask)
            out = out.transpose(1, 2)

        out = out.view(batch, patches, self.n_heads * self.head_dim)
        attn_output, _ = self.o_proj(out)

        return attn_output, None


class PixtralHFTransformerBlock(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.attention = PixtralHFAttention(config,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.attention")
        self.feed_forward = PixtralHFMLP(config,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.feed_forward")
        self.ffn_norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        r, _ = self.attention.forward(self.attention_norm(hidden_states),
                                      attention_mask=attention_mask,
                                      position_embeddings=position_embeddings)
        h = hidden_states + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class PixtralHFTransformer(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            PixtralHFTransformerBlock(config=config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> torch.Tensor:
        hidden_states_pool = [x]

        for layer in self.layers:
            x = layer(x, attention_mask, position_embeddings)
            if return_all_hidden_states:
                hidden_states_pool.append(x)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return x


class PixtralHFVisionModel(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralHFTransformer(
            config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.transformer",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.transformer.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.transformer.layers)} "
                "layers.")

        if require_post_norm is True:
            msg = "PixtralHFVisionModel does not have post-layernorm"
            raise ValueError(msg)

        self.dtype = next(self.parameters()).dtype
        self.device = next(self.parameters()).device
        self.patch_positional_embedding = PixtralRotaryEmbedding(
            config, self.device)

    def forward(
        self,
        pixel_values: List[torch.Tensor],
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Each image to be processed will be a separate tensor
                in pixel_values. This means it will be a list of tensors
                because multiple requests batched can have multiple images,
                each with their own shape potentially
            feature_sample_layers: Layer indices whose features should be
                concatenated and used as the visual encoder output. If none
                are provided, the last layer is used.

        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype))
            for img in pixel_values
        ]

        patch_embeds = [
            p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list
        ]
        embed_sizes = [p.shape[1] for p in patch_embeds]

        # flatten to a single sequence
        patch_embeds = torch.cat(patch_embeds, dim=1)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size).to(
                self.device)
        position_embedding = self.patch_positional_embedding(
            patch_embeds, position_ids)

        if USE_XFORMERS_OPS:
            attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], )
        else:
            from transformers.models.pixtral.modeling_pixtral import (
                generate_block_attention_mask)
            attention_mask = generate_block_attention_mask(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
                patch_embeds)

        return_all_hidden_states = feature_sample_layers is not None
        out = self.transformer(
            patch_embeds,
            attention_mask,
            position_embedding,
            return_all_hidden_states=return_all_hidden_states)

        out = resolve_visual_encoder_outputs(out, feature_sample_layers, None,
                                             self.config.num_hidden_layers)

        # squeeze dim 0 and split into separate tensors for each image
        out = torch.split(torch.squeeze(out), embed_sizes)
        return out

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        layer_count = len(self.transformer.layers)

        for name, loaded_weight in weights:
            # omit layers when num_hidden_layers_override is set
            if name.startswith("transformer.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
