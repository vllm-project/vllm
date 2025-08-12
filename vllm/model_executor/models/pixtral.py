from dataclasses import dataclass, fields
from functools import cached_property
from itertools import tee
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mistral_common.protocol.instruct.messages import ImageChunk
from PIL import Image
from transformers import PixtralVisionConfig, PretrainedConfig
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens)
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding, apply_rotary_pos_emb, position_ids_in_meshgrid)

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, MultiModalConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import merge_multimodal_embeddings
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   consecutive_placeholder_ranges)
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils import is_list_of

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import init_vllm_registered_model

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False


def get_max_pixtral_image_tokens(ctx: InputContext):
    tokenizer = cached_get_tokenizer(
        ctx.model_config.tokenizer,
        tokenizer_mode=ctx.model_config.tokenizer_mode)
    mm_encoder = tokenizer.instruct.mm_encoder

    max_image_size = mm_encoder.mm_config.max_image_size
    image_patch_size = mm_encoder.mm_config.image_patch_size

    return ((max_image_size // image_patch_size)**2)


def dummy_data_for_pixtral(ctx: InputContext, seq_len: int,
                           mm_counts: Mapping[str, int]):
    tokenizer = cached_get_tokenizer(
        ctx.model_config.tokenizer,
        tokenizer_mode=ctx.model_config.tokenizer_mode)

    mm_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
    patch_size = mm_encoder.mm_config.image_patch_size
    image_token_id = mm_encoder.special_ids.img

    mm_config = ctx.model_config.multimodal_config
    num_images = mm_config.limit_per_prompt.get("image", 1)

    # dummy size
    size = 256
    image = Image.new("RGB", (size, size), color=0)

    image_feature_size = (size**2) // (patch_size**2)

    num_image_tokens = image_feature_size * num_images
    seq_data = SequenceData.from_prompt_token_counts(
        (image_token_id, num_image_tokens),
        (0, seq_len - num_image_tokens),
    )

    mm_data = {"image": num_images * [image]}
    mm_placeholders = {
        "image":
        consecutive_placeholder_ranges(num_items=num_images,
                                       item_size=image_feature_size)
    }
    return DummyData(seq_data, mm_data, mm_placeholders)


def input_mapper_for_pixtral(ctx: InputContext,
                             data: object) -> MultiModalInputs:
    """Maps the input data to its MultiModalInputs (if any).

    Args:
        ctx: Context of the loaded model.
        data: data potentially containing image/image embeddings to be mapped
            to pixel_values in .forward() for a visual QWenLMHeadModel model.

    Returns:
        MultiModalInputs containing the stacked normalized images tensor or
        image embeddings.
    """
    # Early exit if we have provided an image to a language only Qwen model
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer, tokenizer_mode=model_config.tokenizer_mode)

    data_list = data if isinstance(data, list) else [data]

    images = []
    for image_data in data_list:
        image = ImageChunk(image=image_data)
        encoding = tokenizer.instruct.mm_encoder(image)
        image = torch.from_numpy(encoding.image).to(device="cuda",
                                                    dtype=torch.float16)
        images.append(image)

    return MultiModalInputs({"images": images})


def input_processor_for_pixtral(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is not None and "image" in multi_modal_data:
        tokenizer = cached_get_tokenizer(
            ctx.model_config.tokenizer,
            tokenizer_mode=ctx.model_config.tokenizer_mode)

        mm_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
        image_token_id = mm_encoder.special_ids.img

        if image_token_id not in inputs['prompt_token_ids']:
            raise ValueError(
                (f"You've passed {inputs=} without {image_token_id=}"
                 " Make sure to process your input via mistral_common's"
                 " tokenizer or pass a chat completion request. For more"
                 " For more info, see: "
                 "https://github.com/vllm-project/vllm/issues/8411."))

    return inputs


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_pixtral)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_pixtral_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_pixtral)
@INPUT_REGISTRY.register_input_processor(input_processor_for_pixtral)
class PixtralForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsPP):

    def __init__(self,
                 config: PretrainedConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

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
            config.text_config,
            cache_config,
            quant_config,
            prefix="language_model")

        self.vision_encoder = VisionTransformer(self.vision_args)
        self.vision_language_adapter = VisionLanguageAdapter(
            self.vision_args, dim=config.text_config.hidden_size)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for pixtral.

        TODO

        """
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            image_input = self._parse_and_validate_image_input(**kwargs)

            if image_input is not None:
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = self.language_model.model.get_input_embeddings(
                    input_ids)

                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.vision_args.image_token_id)

                input_ids = None
            else:
                inputs_embeds = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def _parse_and_validate_image_input(
        self,
        images: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor],
                               torch.Tensor]] = None
    ) -> Optional[List[torch.Tensor]]:
        if images is None:
            return None

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

        return images

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

        def is_vision_weights(weight: Tuple[str, torch.Tensor]):
            return is_vision_encoder_weights(
                weight) or is_vision_lang_adapter_weights(weight)

        llm_weights, vision_encoder_weights, vision_lang_adapter_weights = tee(
            weights, 3)

        # llm
        llm_weights = filter(lambda x: not is_vision_weights(x), llm_weights)
        self.language_model.load_weights(llm_weights)

        # vision encoder
        vision_encoder_weights = filter(is_vision_encoder_weights,
                                        vision_encoder_weights)
        vision_encoder_dict = dict(self.vision_encoder.named_parameters())
        for name, loaded_weight in vision_encoder_weights:
            # cut 'vision_encoder.'
            name = '.'.join(name.split(".")[1:])
            param = vision_encoder_dict[name]

            default_weight_loader(param, loaded_weight)

        # adapter
        vision_lang_adapter_weights = filter(is_vision_lang_adapter_weights,
                                             vision_lang_adapter_weights)
        vision_lang_adpter_dict = dict(
            self.vision_language_adapter.named_parameters())
        for name, loaded_weight in vision_lang_adapter_weights:
            # cut 'vision_language_adapter.'
            name = '.'.join(name.split(".")[1:])
            param = vision_lang_adpter_dict[name]
            default_weight_loader(param, loaded_weight)


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
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
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
            bias=True,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=True)

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


def get_pixtral_hf_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_pixtral_hf_patch_grid_length(image_size=image_size,
                                                   patch_size=patch_size)
    return grid_length * grid_length


def get_max_pixtral_hf_image_feature_size(
        hf_config: PixtralVisionConfig) -> int:
    return get_pixtral_hf_num_patches(image_size=hf_config.image_size,
                                      patch_size=hf_config.patch_size)


def get_max_pixtral_hf_image_tokens(hf_config: PixtralVisionConfig) -> int:
    return get_max_pixtral_hf_image_feature_size(hf_config)


def dummy_seq_data_for_pixtral_hf(
        hf_config: PixtralVisionConfig,
        seq_len: int,
        num_images: int,
        *,
        image_token_id: int,
        image_feature_size_override: Optional[int] = None,
        mm_key: str = "image"):
    if image_feature_size_override is None:
        image_feature_size = get_max_pixtral_hf_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    return SequenceData.from_prompt_token_counts(
        (image_token_id, image_feature_size * num_images),
        (0, seq_len - image_feature_size * num_images),
    ), {
        mm_key:
        consecutive_placeholder_ranges(num_items=num_images,
                                       item_size=image_feature_size)
    }


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


def get_pixtral_hf_image_feature_size(hf_config: PixtralVisionConfig,
                                      image_width: int,
                                      image_height: int) -> Tuple[int, int]:
    # Adapted from transformers.models.pixtral.image_processing_pixtral.get_resize_output_image_size # noqa: E501
    # https://github.com/huggingface/transformers/blob/2bd4d5897dc73e8b172832070a6f9e567a0df017/src/transformers/models/pixtral/image_processing_pixtral.py#L180 # noqa: E501
    max_width, max_height = hf_config.image_size, hf_config.image_size
    patch_width, patch_height = hf_config.patch_size, hf_config.patch_size

    ratio = max(image_width / max_width, image_height / max_height)

    if ratio > 1:
        image_width = int(numpy.ceil(image_width / ratio))
        image_height = int(numpy.ceil(image_height / ratio))

    num_height_tokens, num_width_tokens = _num_image_tokens(
        (image_height, image_width), (patch_height, patch_width))

    return num_width_tokens, num_height_tokens


def input_processor_for_pixtral_hf(
    model_config: ModelConfig,
    hf_config: PixtralVisionConfig,
    inputs: DecoderOnlyInputs,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[Union[int, List[int]]] = None,
) -> DecoderOnlyInputs:
    assert image_feature_size_override is None, (
        "image_feature_size_override is not supported for Pixtral")

    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    processor = cached_get_processor(model_config.model)

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        image_data = [image_data]
    elif not is_list_of(image_data, Image.Image):
        raise TypeError(f"Invalid image type: {type(image_data)}")

    new_prompt = inputs.get("prompt")
    new_token_ids = inputs["prompt_token_ids"]

    image_token = processor.image_token
    image_break_token = processor.image_break_token
    image_end_token = processor.image_end_token

    # Update new_prompt if present
    if new_prompt:
        parts = new_prompt.split(image_token)
        assert len(parts) - 1 == len(image_data)
        new_parts = [parts[0]]  # Start with the part before any image tokens

        for image, next_part in zip(image_data, parts[1:]):
            w, h = image.size
            (num_width_tokens,
             num_height_tokens) = get_pixtral_hf_image_feature_size(
                 hf_config, image_width=w, image_height=h)

            replace_tokens = [image_token] * num_width_tokens + [
                image_break_token
            ]
            replace_tokens = replace_tokens * num_height_tokens
            replace_tokens[-1] = image_end_token

            new_parts.append("".join(replace_tokens))
            new_parts.append(next_part)

        new_prompt = "".join(new_parts)

    # Update new_token_ids
    convert_tokens_to_ids = processor.tokenizer.convert_tokens_to_ids
    image_token_id = convert_tokens_to_ids(image_token)
    image_break_id = convert_tokens_to_ids(image_break_token)
    image_end_id = convert_tokens_to_ids(image_end_token)
    placeholder_token_id = -999
    # Find all image token indices at once
    placeholder_indices = [
        idx for idx, token_id in enumerate(new_token_ids)
        if token_id == image_token_id
    ]
    assert len(placeholder_indices) == len(image_data)
    replace_tokens_list = []
    for placeholder_idx, image in zip(placeholder_indices, image_data):
        new_token_ids[placeholder_idx] = placeholder_token_id

        w, h = image.size
        (num_width_tokens,
         num_height_tokens) = get_pixtral_hf_image_feature_size(hf_config,
                                                                image_width=w,
                                                                image_height=h)

        replace_tokens = [image_token_id] * num_width_tokens + [image_break_id]
        replace_tokens = replace_tokens * num_height_tokens
        replace_tokens[-1] = image_end_id
        replace_tokens_list.append(replace_tokens)

    # Backward iteration for replacement without affecting known indices
    for placeholder_idx, replace_tokens in zip(reversed(placeholder_indices),
                                               reversed(replace_tokens_list)):
        new_token_ids[placeholder_idx:placeholder_idx + 1] = replace_tokens

    # NOTE: Create a defensive copy of the original inputs
    return token_inputs(prompt_token_ids=new_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data)


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
        # TODO: Use quant_config and prefix after optimizing this
        self.gate_proj = nn.Linear(config.hidden_size,
                                   config.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size,
                                 config.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,
                                   config.hidden_size,
                                   bias=False)
        self.act = get_act_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


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
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.scale = self.head_dim**-0.5

        # TODO: Use quant_config and prefix after optimizing this
        self.q_proj = nn.Linear(config.hidden_size,
                                config.hidden_size,
                                bias=False)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.hidden_size,
                                bias=False)
        self.v_proj = nn.Linear(config.hidden_size,
                                config.hidden_size,
                                bias=False)
        self.o_proj = nn.Linear(config.hidden_size,
                                config.hidden_size,
                                bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, patches, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Transpose q and k to apply HF's Rotary Position Embedding
        q = q.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)

        if USE_XFORMERS_OPS:
            # Transpose q and k back for attention
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.reshape(batch, patches, self.n_heads, self.head_dim)

            out = xops.memory_efficient_attention(q,
                                                  k,
                                                  v,
                                                  attn_bias=attention_mask)
        else:
            v = v.reshape(batch, patches, self.n_heads,
                          self.head_dim).transpose(1, 2)
            out = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask)
            out = out.transpose(1, 2)

        out = out.reshape(batch, patches, self.n_heads * self.head_dim)

        return self.o_proj(out)


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
        r = self.attention.forward(self.attention_norm(hidden_states),
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
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask, position_embeddings)
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
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Each image to be processed will be a separate tensor
                in pixel_values. This means it will be a list of tensors
                because multiple requests batched can have multiple images,
                each with their own shape potentially

        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype))
            for img in pixel_values
        ]

        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1)
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

        out = self.transformer(patch_embeds, attention_mask,
                               position_embedding)

        return out

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = []
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
