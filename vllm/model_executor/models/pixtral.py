# SPDX-License-Identifier: Apache-2.0

import math
import re
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mistral_common.protocol.instruct.messages import ImageChunk
from PIL import Image
from transformers import BatchFeature, PixtralVisionConfig
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens)
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding, apply_rotary_pos_emb, position_ids_in_meshgrid)

from vllm import envs
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
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (ImageItem, MultiModalDataDict,
                                    MultiModalFieldConfig, MultiModalInputs,
                                    MultiModalKwargsItem, NestedTensors,
                                    PlaceholderRange)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               cached_tokenizer_from_config)

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import VisionEncoderInfo, resolve_visual_encoder_outputs

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False

PIXTRAL_IMAGE_TOKEN = "[IMG]"


class PixtralProcessingInfo(BaseProcessingInfo):

    def get_tokenizer(self) -> AnyTokenizer:
        return cached_tokenizer_from_config(self.ctx.model_config)

    def get_hf_processor(self):
        raise ValueError(
            "Pixtral model in Mistral format does not support HF processor.")

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        """
        Return max number of image tokens + image break tokens + image end token
        """
        tokenizer = self.get_tokenizer()
        image_encoder = tokenizer.instruct.mm_encoder
        max_image_size = image_encoder.image_config.max_image_size
        max_image = Image.new("RGB", (max_image_size, max_image_size), color=0)
        max_num_tokens = len(image_encoder(ImageChunk(image=max_image)).tokens)
        return {"image": max_num_tokens}

    def get_image_size_with_most_features(self) -> ImageSize:
        tokenizer = self.get_tokenizer()
        image_encoder = tokenizer.instruct.mm_encoder
        max_image_size = image_encoder.image_config.max_image_size

        return ImageSize(width=max_image_size, height=max_image_size)


class PixtralDummyInputsBuilder(BaseDummyInputsBuilder[PixtralProcessingInfo]):

    # This is supposed to only build image related inputs.
    # It builds a prompt text reprensenting the image placeholder text,
    # and builds images as multi-modal data.
    # The tokenization and mm processing is assumed to be done later.
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 1)
        w, h = self.info.get_image_size_with_most_features()
        images = self._get_dummy_images(width=w,
                                        height=h,
                                        num_images=num_images)

        return ProcessorInputs(
            prompt_text=PIXTRAL_IMAGE_TOKEN * num_images,
            mm_data={"image": images},
        )


class PixtralMultiModalProcessor(BaseMultiModalProcessor[PixtralProcessingInfo]
                                 ):

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputs:
        tokenizer = self.info.get_tokenizer()
        image_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
        image_token_id = image_encoder.special_ids.img
        image_break_id = image_encoder.special_ids.img_break
        image_end_id = image_encoder.special_ids.img_end

        mm_items = self._to_mm_items(mm_data)

        if isinstance(prompt, str):
            prompt, num_matched_images = re.subn(
                PIXTRAL_IMAGE_TOKEN,
                "",
                prompt,
            )
            assert num_matched_images == mm_items.get_all_counts().get("image")
        else:
            if image_token_id not in prompt:
                raise ValueError(
                    f"You've passed {prompt=} without {image_token_id=}"
                    " Make sure to process your input via mistral_common's"
                    " tokenizer or pass a chat completion request."
                    " For more info, see: "
                    "https://github.com/vllm-project/vllm/issues/8411.")

        if envs.VLLM_USE_V1:
            model_id = self.info.model_id
            mm_hashes = {
                modality: [
                    MultiModalHasher.hash_kwargs(model_id=model_id,
                                                 **{modality: item},
                                                 **hf_processor_mm_kwargs)
                    for item in items
                ]
                for modality, items in mm_items.items()
            }
        else:
            mm_hashes = None

        mm_kwargs = self._cached_get_mm_kwargs(mm_items,
                                               hf_processor_mm_kwargs)

        if isinstance(prompt, str):
            tokenized_prompt = tokenizer.encode(prompt)
            # Add all image tokens to the beginning of prompt
            prompt_token_ids = []
            for image_elem in mm_kwargs.get_items("image"):
                prompt_token_ids.extend(
                    image_elem["image_tokens"].data.tolist())
            prompt_token_ids.extend(tokenized_prompt)
        else:
            prompt_token_ids = prompt

        # Get precise tracking of placeholder positions
        placeholder_ranges: List[PlaceholderRange] = []
        curr_offset = -1
        curr_length = 0
        for i in range(len(prompt_token_ids)):
            if prompt_token_ids[i] in (image_token_id, image_break_id):
                if curr_offset < 0:
                    curr_offset = i
                curr_length += 1
            elif prompt_token_ids[i] == image_end_id:
                curr_length += 1
                placeholder_ranges.append(
                    PlaceholderRange(offset=curr_offset, length=curr_length))
                curr_offset = -1
                curr_length = 0
            else:
                pass

        assert len(placeholder_ranges) == mm_items.get_all_counts().get(
            "image")

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt if isinstance(prompt, str) else "",
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders={"image": placeholder_ranges},
        )

    def _get_mm_kwargs_from_mm_data(
        self,
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalKwargs:
        tokenizer = self.info.get_tokenizer()
        images = []
        image_tokens_list = []
        for image_data in mm_data_items.get_items("image",
                                                  ImageProcessorItems):
            image = ImageChunk(image=image_data)
            encoding = tokenizer.instruct.mm_encoder(image)
            image = torch.from_numpy(encoding.image).to(dtype=torch.float16)
            images.append(image)
            image_tokens_list.append(encoding.tokens)

        image_tokens = [torch.tensor(tokens) for tokens in image_tokens_list]

        mm_kwargs = self._get_mm_kwargs_from_images(images, image_tokens,
                                                    hf_processor_mm_kwargs)

        mm_item_counts = mm_data_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        return mm_kwargs

    def _cached_get_mm_kwargs(
        self,
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalKwargs:
        cache = self.cache
        model_id = self.info.model_id

        _, passthrough_data = self._get_hf_mm_data(mm_data_items)
        if cache is None or passthrough_data:
            return self._get_mm_kwargs_from_mm_data(mm_data_items,
                                                    hf_processor_mm_kwargs)

        mm_maybe_cached_kw_items = {
            modality: [
                cache.get(model_id, modality, item, hf_processor_mm_kwargs)
                for item in items
            ]
            for modality, items in mm_data_items.items()
        }

        mm_missing_idxs = {
            modality:
            [idx for idx, item in enumerate(kw_items) if item is None]
            for modality, kw_items in mm_maybe_cached_kw_items.items()
        }
        mm_missing_data = {
            modality: [mm_data_items[modality][idx] for idx in idxs]
            for modality, idxs in mm_missing_idxs.items()
        }
        mm_missing_data_items = self._to_mm_items(mm_missing_data)

        if not all([
                len(mm_missing_idxs[modality]) == 0
                for modality in mm_missing_idxs
        ]):
            mm_missing_kwargs = self._get_mm_kwargs_from_mm_data(
                mm_missing_data_items, hf_processor_mm_kwargs)
        else:
            mm_missing_kwargs = {}

        mm_missing_next_idx = {
            modality: 0
            for modality in mm_missing_data_items
        }
        merged_kw_items = list[MultiModalKwargsItem]()
        for modality, kw_items in mm_maybe_cached_kw_items.items():
            for idx, kw_item in enumerate(kw_items):
                if kw_item is None:
                    kw_item = mm_missing_kwargs.get_item(
                        modality,
                        mm_missing_next_idx[modality],
                    )

                    cache.put(
                        model_id,
                        modality,
                        mm_data_items[modality][idx],
                        hf_processor_mm_kwargs,
                        kw_item,
                    )

                    mm_missing_next_idx[modality] += 1

                merged_kw_items.append(kw_item)

        if self.enable_sanity_checks:
            mm_missing_counts = mm_missing_data_items.get_all_counts()
            assert all(
                item_count == mm_missing_counts[modality]
                for modality, item_count in mm_missing_next_idx.items()), dict(
                    mm_missing_next_idx=mm_missing_next_idx,
                    mm_missing_counts=mm_missing_counts)

        mm_kwargs = MultiModalKwargs.from_items(merged_kw_items)
        return mm_kwargs

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        raise NotImplementedError

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Output the metadata of each field."""
        return dict(
            images=MultiModalFieldConfig.batched("image"),
            image_tokens=MultiModalFieldConfig.batched("image"),
        )

    def _get_mm_kwargs_from_images(
        self,
        images: list[ImageItem],
        image_tokens: List[torch.Tensor],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalKwargs:
        # Convert images and image tokens into compatible HF type BatchFeature
        # such that we can reuse from_hf_inputs() which builds the mm kwargs.
        inputs = BatchFeature(dict(
            images=images,
            image_tokens=image_tokens,
        ))
        return MultiModalKwargs.from_hf_inputs(
            inputs,
            self._get_mm_fields_config(inputs, hf_processor_mm_kwargs),
        )


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

        if not ("image_break_token_id" in vision_args
                and "image_end_token_id" in vision_args):
            raise ValueError(
                "'image_break_token_id' and 'image_end_token_id' not found "
                "in the vision_encoder arguments. Please download the latest "
                "version of 'params.json' from the model repository.")

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
        image_input, image_tokens = self._parse_and_validate_image_input(
            **kwargs)
        if image_input is None:
            return None

        vision_embeddings = self._process_image_input(image_input)

        # NOTE: We patch the outputs of the vision encoder with embeddings
        # from `[IMG_BREAK]` and `[IMG_END]` tokens.
        image_embeds = self.language_model.get_input_embeddings(image_tokens)
        image_token_mask = image_tokens == self.vision_args.image_token_id
        image_embeds[image_token_mask] = vision_embeddings

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the indices of `[IMG_END]` token.
        image_end_mask = image_tokens == self.vision_args.image_end_token_id
        split_indices = torch.where(image_end_mask)[0] + 1
        if len(split_indices) <= 1:
            # Do not split, return as tensor of shape [1, fs, hs]
            return image_embeds.unsqueeze(0)

        # If the last split index is the last index in image_tokens, we
        # ignore it to avoid empty split tensor
        if split_indices[-1] == len(image_tokens):
            split_indices = split_indices[:-1]

        image_embeds = image_embeds.tensor_split(split_indices.cpu())
        return image_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, [
                    self.vision_args.image_token_id,
                    self.vision_args.image_break_token_id,
                    self.vision_args.image_end_token_id,
                ])
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for pixtral.
        """
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
        self,
        images: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor],
                               torch.Tensor]] = None,
        image_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        if images is None:
            return None, None

        if isinstance(images, torch.Tensor):
            # if passed as batch take all images
            N, B, C, W, H = images.shape
            images = images.reshape(N * B, C, W, H)
            images = [images[i] for i in range(images.size(0))]
        elif isinstance(images, list):
            # if passed as list flatten lists of tensors
            flatten_images = []
            for imgs_per_req in images:
                imgs_per_req = [
                    imgs_per_req[i] for i in range(imgs_per_req.size(0))
                ] if isinstance(imgs_per_req, torch.Tensor) else imgs_per_req

                flatten_images.extend(imgs_per_req)

            images = flatten_images

        if isinstance(image_tokens, torch.Tensor):
            # image_tokens are batched
            image_tokens = image_tokens.flatten()
        elif isinstance(image_tokens, list):
            # image_tokens are of different lengths thus passed as a list
            image_tokens = torch.cat(image_tokens)

        assert image_tokens.dim() == 1

        return images, image_tokens

    def _process_image_input(self,
                             image_input: List[torch.Tensor]) -> torch.Tensor:
        return self.vision_language_adapter(self.vision_encoder(image_input))

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
    image_break_token_id: int
    image_end_token_id: int
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


def get_pixtral_hf_patch_grid_length(*, image_size: int,
                                     patch_size: int) -> int:
    # Since interpolation is applied, the image size need not be divisible
    # assert image_size % patch_size == 0
    return image_size // patch_size


def get_pixtral_hf_image_feature_size(
    *,
    image_size: int,
    patch_size: int,
) -> int:
    grid_length = get_pixtral_hf_patch_grid_length(
        image_size=image_size,
        patch_size=patch_size,
    )

    # Consider the image_break_token
    return (grid_length + 1) * grid_length


def get_max_pixtral_hf_image_tokens(hf_config: PixtralVisionConfig) -> int:
    grid_length = get_pixtral_hf_patch_grid_length(
        image_size=hf_config.image_size,
        patch_size=hf_config.patch_size,
    )

    # Consider the image_break_token
    return (grid_length + 1) * grid_length


def dummy_image_for_pixtral_hf(
    hf_config: PixtralVisionConfig,
    num_images: int,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}


# Adapted from transformers.models.pixtral.image_processing_pixtral.get_resize_output_image_size # noqa: E501
# https://github.com/huggingface/transformers/blob/2bd4d5897dc73e8b172832070a6f9e567a0df017/src/transformers/models/pixtral/image_processing_pixtral.py#L180
def get_pixtral_hf_image_feature_grid_size(
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


class PixtralHFEncoderInfo(VisionEncoderInfo[PixtralVisionConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return get_pixtral_hf_image_feature_size(
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
        )

    def get_max_image_tokens(self) -> int:
        return get_max_pixtral_hf_image_tokens(self.vision_config)

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        return get_pixtral_hf_patch_grid_length(
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
        )


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
