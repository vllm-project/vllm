# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from https://github.com/amalad/vllm/blob/nemotron_parse/vllm/model_executor/models/nemotron_parse.py
# that's based on https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/blob/main/hf_nemotron_parse_modeling.py
#
# Bart classes based on old vLLM codebase:
# https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/bart.py

import math
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from transformers import (
    BartConfig,
    BatchFeature,
    PreTrainedConfig,
)

from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
)
from vllm.model_executor.models.radio import RadioModel
from vllm.model_executor.models.whisper import WhisperCrossAttention
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)

_PARSE_MTP_WEIGHT_NAMES = frozenset(
    {
        "decoder.extra_heads.0.weight",
        "decoder.extra_heads.0.bias",
        "decoder.extra_proj.0.weight",
        "decoder.extra_proj.0.bias",
    }
)


def _content_normalized_positions(
    patches: torch.Tensor,
    patch_generator: nn.Module,
    image_size: tuple[int, int],
    content_size: tuple[int, int],
) -> torch.Tensor:
    """Apply Parse 3.5's content-normalized positional field to one page."""
    height, width = image_size
    content_height, content_width = content_size
    patch_size = int(patch_generator.patch_size)
    rows, cols = height // patch_size, width // patch_size
    if height % patch_size or width % patch_size:
        raise ValueError(f"image size {image_size} is not patch aligned")
    if not (0 < content_height <= height and 0 < content_width <= width):
        raise ValueError(f"content size {content_size} must be within {image_size}")

    positional_grid = patch_generator.pos_embed.reshape(
        1, patch_generator.num_rows, patch_generator.num_cols, -1
    ).permute(0, 3, 1, 2)
    content_rows = (content_height + patch_size - 1) // patch_size
    content_cols = (content_width + patch_size - 1) // patch_size
    if min(content_rows, content_cols) == 1:
        field = F.adaptive_avg_pool2d(
            positional_grid.float(), (content_rows, content_cols)
        )
    else:
        field = F.interpolate(
            positional_grid.float(),
            size=(content_rows, content_cols),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    row_starts = torch.arange(rows, device=patches.device) * patch_size
    col_starts = torch.arange(cols, device=patches.device) * patch_size
    row_extents = (content_height - row_starts).clamp(0, patch_size)
    col_extents = (content_width - col_starts).clamp(0, patch_size)
    normalized_y = (row_starts + 0.5 * row_extents).clamp(
        max=content_height
    ) / content_height
    normalized_x = (col_starts + 0.5 * col_extents).clamp(
        max=content_width
    ) / content_width
    grid_y, grid_x = torch.meshgrid(normalized_y, normalized_x, indexing="ij")
    sampling_grid = torch.stack((2 * grid_x - 1, 2 * grid_y - 1), dim=-1)
    positions = F.grid_sample(
        field,
        sampling_grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).to(dtype=patches.dtype)
    return patches + positions.flatten(2).permute(0, 2, 1)


class BartScaledWordEmbedding(VocabParallelEmbedding):
    """This module overrides VocabParallelEmbedding's
    forward by multiplying with embeddings scale.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class BartParallelLMHead(ParallelLMHead):
    """This module overrides ParallelLMHead's
    forward by dividing by embeddings scale,
    yielding effectively the inverse of
    BartScaledWordEmbedding
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) / self.embed_scale


class NemotronParseSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        head_dim = embed_dim // num_heads
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.q_size = self.qkv_proj.num_heads * head_dim
        self.kv_size = self.qkv_proj.num_kv_heads * head_dim
        self.attn = Attention(
            self.qkv_proj.num_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=self.qkv_proj.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class BartDecoderLayer(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = NemotronParseSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            num_kv_heads=getattr(config, "self_attention_kv_heads", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = get_act_fn(config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.add_cross_attention = getattr(config, "add_cross_attention", True)
        if self.add_cross_attention:
            """
            afeldman-nm: personally I would call this "cross-attention",
            however I left the name as "encoder_attn" to maintain consistency
            with the name of the pretrained weights.
            """
            self.encoder_attn = WhisperCrossAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.encoder_attn",
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            ffn_hidden_size,
            ffn_intermediate_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            ffn_intermediate_size,
            ffn_hidden_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Args:
            decoder_hidden_states: torch.Tensor of *decoder* input embeddings.
            encoder_hidden_states: torch.Tensor of *encoder* input embeddings.

        Returns:
            Decoder layer output torch.Tensor

        """
        residual = decoder_hidden_states

        # Self Attention
        hidden_states = self.self_attn(hidden_states=decoder_hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block

        residual = hidden_states

        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class MBartDecoderLayer(BartDecoderLayer):
    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states

        # Cross-Attention Block

        if self.add_cross_attention:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class MBartDecoderNoPos(nn.Module, EagleModelMixin):
    """Transformer decoder consisting of *config.decoder_layers* layers.
    Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding

    """

    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.layers = nn.ModuleList(
            [
                MBartDecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.decoder_layers)
            ]
        )

        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @property
    def model(self) -> "MBartDecoderNoPos":
        return self

    @property
    def lm_head(self) -> ParallelLMHead:
        return self._lm_head_ref()

    def forward(
        self,
        decoder_input_ids: torch.Tensor | None,
        *,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        r"""Args:
            decoder_input_ids: Indices of *decoder* input sequence tokens in the
                vocabulary. Padding will be ignored by default should you provide it.
            encoder_hidden_states: Tensor of encoder output embeddings
        Returns:
            Decoder output torch.Tensor

        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(decoder_input_ids)

        hidden_states = self.layernorm_embedding(inputs_embeds)

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, None)
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, None
            )

        hidden_states = self.layer_norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            # MergedColumnParallelLinear uses integer indices (0, 1)
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", 0),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith("embed_positions"):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class NemotronParsePixelInputs(TensorSchema):
    """Dimensions:
    - b: Batch size
    - c: Number of channels (3)
    - h: Height
    - w: Width
    """

    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("b", 3, "h", "w")]


class NemotronParsePackedPixelInputs(TensorSchema):
    """Packed 16x16 RADIO patches and their per-image pixel geometry."""

    type: Literal["packed_pixel_values"]
    data: Annotated[torch.Tensor, TensorShape(1, "n", "p")]
    image_sizes: torch.Tensor
    content_sizes: torch.Tensor


class NemotronParseProcessingInfo(BaseProcessingInfo):
    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # Because the encoder prompt is padded

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_hf_processor(self, **kwargs: object):
        processor = super().get_hf_processor(**kwargs)
        config = self.get_hf_config()
        if getattr(config.decoder, "causal_vision_prefix", False):
            expected_packing = bool(getattr(config, "navit_packing", False))
            actual_packing = bool(
                getattr(processor.image_processor, "navit_packing", False)
            )
            if actual_packing != expected_packing:
                raise ValueError(
                    "Checkpoint image processor navit_packing does not match the "
                    "native model configuration. Some checkpoint processors reset "
                    "packing when image processor overrides are supplied; remove "
                    "incompatible mm_processor_kwargs or fix the checkpoint processor."
                )
        return processor

    def is_dynamic_resolution(self) -> bool:
        return bool(getattr(self.get_hf_config(), "navit_packing", False))

    def get_num_image_tokens(self) -> int:
        config = self.get_hf_config()
        configured_tokens = getattr(config, "num_image_tokens", None)
        if configured_tokens is not None:
            return configured_tokens

        final_size = config.image_size
        patch_size = config.encoder.patch_size

        return (final_size[0] // patch_size) * ((final_size[1] // patch_size) // 4) + 1

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        image_tokens = self.get_num_image_tokens()
        return {"image": image_tokens}


class NemotronParseDummyInputsBuilder(
    BaseDummyInputsBuilder[NemotronParseProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_height, target_width = self.info.get_hf_config().image_size

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            )
        }


def _to_pil_image(image: object) -> Image.Image:
    """Convert a `uint8` array/tensor image item to PIL, since the checkpoint's
    image processor calls `.convert("RGB")` on every input."""
    if isinstance(image, Image.Image):
        return image

    array = image.numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3:
        raise ValueError(
            "Nemotron-Parse expects image arrays/tensors to be 3-D uint8, "
            f"got {array.dtype} {array.shape}"
        )
    # Same layout heuristic as `ImageProcessorItems.get_image_size`.
    if array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    if array.shape[-1] == 1:
        array = array[..., 0]

    return Image.fromarray(np.ascontiguousarray(array))


def _build_uint8_canvas(
    image_processor: Any, images: Sequence[Image.Image]
) -> torch.Tensor:
    """Reproduce the image processor's pipeline up to, but excluding, its
    `ToTensor` rescale, yielding the padded `uint8` canvas as `(N, 3, H, W)`.

    Raises `AttributeError`/`TypeError` if the checkpoint's remote code does
    not expose that pipeline.
    """
    resize = image_processor._resize_with_aspect_ratio
    pad = image_processor._pad_to_size
    # v1.x pads with albumentations, 2.0 sets `transform` to None and pads
    # with `_pad_to_size`; mirror `preprocess` and pick the same branch.
    transform = image_processor.transform

    canvases = []
    for image in images:
        array = np.asarray(image.convert("RGB"))
        array = resize(array)
        array = transform(image=array)["image"] if transform is not None else pad(array)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise TypeError(
                f"expected a uint8 HWC canvas, got {array.dtype} {array.shape}"
            )
        canvases.append(torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1))

    return torch.stack(canvases)


class RadioMBartPrefixDummyInputsBuilder(NemotronParseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<pad>" * mm_counts.get("image", 0)


class RadioMBartPrefixMultiModalProcessor(
    BaseMultiModalProcessor[NemotronParseProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        if "image_sizes" not in hf_inputs:
            return dict(pixel_values=MultiModalFieldConfig.batched("image"))
        patch_size = self.info.get_hf_config().encoder.patch_size
        patch_grid = hf_inputs["image_sizes"] // patch_size
        patch_counts = patch_grid[:, 0] * patch_grid[:, 1]
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", patch_counts),
            image_sizes=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            content_sizes=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        replacement: list[int] | Callable[[int], list[int]]
        if self.info.is_dynamic_resolution():

            def get_replacement(item_idx: int) -> list[int]:
                item = out_mm_kwargs["image"][item_idx]
                image_sizes = item["image_sizes"].data
                assert isinstance(image_sizes, torch.Tensor)
                height, width = image_sizes.tolist()
                patch_size = self.info.get_hf_config().encoder.patch_size
                num_tokens = (height // patch_size) * (width // (4 * patch_size)) + 1
                return [1] * num_tokens

            replacement = get_replacement
        else:
            replacement = [1] * self.info.get_num_image_tokens()

        return [
            PromptReplacement(
                modality="image",
                target=[1],
                replacement=replacement,
            )
        ]


class NemotronParseMultiModalProcessor(
    EncDecMultiModalProcessor[NemotronParseProcessingInfo]
):
    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        return [0]

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        images = [_to_pil_image(i) for i in cast(list, hf_data.get("images") or [])]
        if images:
            hf_data = {**hf_data, "images": images}

        # The checkpoint ships its image processor as remote code, and its
        # `preprocess` ignores `do_rescale`/`do_normalize`: it always runs
        # `ToTensor` and normalises from `self.do_normalize`. Build the uint8
        # canvas here so the device-side norm is not applied on top.
        mm_config = self.info.ctx.model_config.get_multimodal_config()
        if images and set(hf_data) == {"images"} and mm_config.mm_device_do_normalize:
            image_processor = self.info.get_hf_processor().image_processor
            try:
                pixel_values = _build_uint8_canvas(image_processor, images)
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    f"{type(image_processor).__name__} does not expose the "
                    f"expected pre-rescale pipeline ({exc}), so the uint8 canvas "
                    "that mm_device_do_normalize requires cannot be built. Pass "
                    "--no-mm-device-do-normalize to normalise on the host instead."
                ) from exc
            return BatchFeature({"pixel_values": pixel_values})

        return super()._call_hf_processor(hf_data, hf_kwargs)

    _get_mm_fields_config = RadioMBartPrefixMultiModalProcessor._get_mm_fields_config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        replacement: list[int] | Callable[[int], list[int]]
        if self.info.is_dynamic_resolution():

            def get_replacement(item_idx: int) -> list[int]:
                item = out_mm_kwargs["image"][item_idx]
                image_sizes = item["image_sizes"].data
                assert isinstance(image_sizes, torch.Tensor)
                height, width = image_sizes.tolist()
                patch_size = self.info.get_hf_config().encoder.patch_size
                num_tokens = (height // patch_size) * (width // (4 * patch_size)) + 1
                return [0] * num_tokens

            replacement = get_replacement
        else:
            replacement = [0] * self.info.get_num_image_tokens()
        return [
            PromptReplacement(modality="image", target=[0], replacement=replacement)
        ]


class RadioWithNeck(nn.Module):
    """Vision encoder using RADIO model with custom neck."""

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config.encoder
        self.image_size = tuple(config.image_size)

        self.model_encoder = self.get_vit_model_from_radio_config(
            config, quant_config=quant_config
        )

        # Neck components
        last_hidden_state = int(config.decoder.d_model)
        spatial_dim = int(getattr(config, "vision_spatial_dim", 1280))
        summary_dim = int(getattr(config, "vision_summary_dim", 3840))
        self.conv1 = nn.Conv1d(spatial_dim, last_hidden_state, 1)
        self.layer_norm1 = nn.LayerNorm(
            last_hidden_state, eps=1e-06, elementwise_affine=True
        )
        self.conv2 = nn.Conv2d(
            last_hidden_state,
            last_hidden_state,
            kernel_size=(1, 4),
            stride=(1, 4),
            padding=0,
            bias=False,
        )
        self.layer_norm2 = nn.LayerNorm(
            last_hidden_state, eps=1e-06, elementwise_affine=True
        )
        self.sum_proj = ColumnParallelLinear(
            summary_dim,
            last_hidden_state,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.sum_proj",
        )
        self.layer_norm3 = nn.LayerNorm(
            last_hidden_state, eps=1e-06, elementwise_affine=True
        )

    def get_vit_model_from_radio_config(
        self,
        hf_config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> RadioModel:
        hf_config_vision = hf_config.encoder
        model_name = hf_config_vision.args.get("model")
        if model_name is None:
            raise ValueError(f"Unsupported vit model type: {model_name}")

        radio_config = RadioConfig(
            model_name=model_name,
            image_size=hf_config.image_size,
            **hf_config_vision.args,
        )

        return RadioModel(config=radio_config, quant_config=quant_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: list[tuple[int, int]] | None = None,
        content_sizes: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if content_sizes is not None:
            return self._encode_parse35(pixel_values, image_sizes, content_sizes)
        if image_sizes is None:
            summary, feature = self.model_encoder(pixel_values)
        else:
            summary, feature = self._encode_dynamic(pixel_values, image_sizes)

        output = self.conv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.layer_norm1(output)

        patch_size = self.config.patch_size
        if image_sizes is not None:
            features: list[torch.Tensor] = []
            summaries = summary.reshape(len(image_sizes), -1)
            offset = 0
            for index, (height, width) in enumerate(image_sizes):
                patch_count = (height // patch_size) * (width // patch_size)
                image_output = output[:, offset : offset + patch_count]
                image_output = rearrange(
                    image_output,
                    "b (h w) d -> b d h w",
                    h=height // patch_size,
                    w=width // patch_size,
                )
                image_output = self.conv2(image_output)
                image_output = rearrange(image_output, "b d h w -> b (h w) d")
                image_output = self.layer_norm2(image_output)
                image_summary = self.layer_norm3(
                    self.sum_proj(summaries[index : index + 1])[0]
                )
                features.append(
                    torch.cat((image_output, image_summary.unsqueeze(1)), dim=1)[0]
                )
                offset += patch_count
            return tuple(features)

        output = rearrange(
            output,
            "b (h w) d -> b d h w",
            h=pixel_values.shape[-2] // patch_size,
            w=pixel_values.shape[-1] // patch_size,
        )

        output = self.conv2(output)
        output = rearrange(output, "b d h w -> b (h w) d")
        output = self.layer_norm2(output)
        summary = self.layer_norm3(self.sum_proj(summary)[0])
        output = torch.cat((output, summary.unsqueeze(1)), dim=1)

        return output

    def _encode_parse35(
        self,
        pixel_values: torch.Tensor,
        image_sizes: list[tuple[int, int]] | None,
        content_sizes: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, ...]:
        if image_sizes is None or len(image_sizes) != len(content_sizes):
            raise ValueError("image_sizes and content_sizes must have equal length")
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)
        patch_generator = self.model_encoder.model.patch_generator
        backbone = self.model_encoder.model
        patch_size = int(patch_generator.patch_size)
        num_skip = int(patch_generator.num_skip)
        num_cls = int(patch_generator.num_cls_tokens)
        patches = patch_generator.embedder(pixel_values)
        cursor = 0
        positioned_pages = []
        for image_size, content_size in zip(image_sizes, content_sizes):
            height, width = image_size
            patch_count = (height // patch_size) * (width // patch_size)
            page = patches[:, cursor : cursor + patch_count]
            if page.shape[1] != patch_count:
                raise ValueError("packed patch sequence does not match image_sizes")
            cursor += patch_count
            positioned_pages.append(
                _content_normalized_positions(
                    page, patch_generator, image_size, content_size
                )
            )
        if not positioned_pages or cursor != patches.shape[1]:
            raise ValueError("packed patch sequence does not match image_sizes")
        hidden = patch_generator.cls_token_dynamic(
            torch.cat(positioned_pages, dim=1), imgs_sizes=image_sizes
        )
        hidden = patch_generator.patch_normalizer(hidden)
        mask_meta = None
        if len(image_sizes) > 1:
            mask_meta = backbone.inter_image_mask_metadata(
                image_sizes, device=hidden.device
            )
        hidden = backbone.encoder(inputs_embeds=hidden, mask_meta=mask_meta)

        cursor = 0
        outputs = []
        for height, width in image_sizes:
            patch_count = (height // patch_size) * (width // patch_size)
            summary = hidden[:, cursor : cursor + num_cls]
            if self.model_encoder.summary_idxs is not None:
                summary = summary[:, self.model_encoder.summary_idxs]
            summary = summary.flatten(1)
            feature = hidden[:, cursor + num_skip : cursor + num_skip + patch_count]
            cursor += num_skip + patch_count
            feature = feature.to(self.conv1.weight.dtype)
            output = self.conv1(feature.transpose(1, 2)).transpose(1, 2)
            output = self.layer_norm1(output)
            output = output.transpose(1, 2).reshape(
                1, output.shape[-1], height // patch_size, width // patch_size
            )
            output = self.conv2(output).flatten(2).transpose(1, 2)
            output = self.layer_norm2(output)
            summary = summary.to(dtype=self.layer_norm3.weight.dtype)
            summary = self.layer_norm3(self.sum_proj(summary)[0]).unsqueeze(1)
            outputs.append(torch.cat((output, summary), dim=1).squeeze(0))
        if cursor != hidden.shape[1]:
            raise ValueError("packed encoder sequence does not match image_sizes")
        return tuple(outputs)

    def _encode_dynamic(
        self,
        pixel_values: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract every image's summaries from packed RADIO output."""
        encoded = self.model_encoder.model(pixel_values, imgs_sizes=image_sizes)
        patch_generator = self.model_encoder.model.patch_generator
        num_skip = patch_generator.num_skip
        num_cls_tokens = patch_generator.num_cls_tokens
        patch_size = patch_generator.patch_size
        summary_indices = self.model_encoder.summary_idxs
        summaries: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        offset = 0
        for height, width in image_sizes:
            patch_count = (height // patch_size) * (width // patch_size)
            image_summary = encoded[:, offset : offset + num_cls_tokens]
            if summary_indices is not None:
                image_summary = image_summary[:, summary_indices]
            summaries.append(image_summary.flatten(1))
            features.append(
                encoded[:, offset + num_skip : offset + num_skip + patch_count]
            )
            offset += num_skip + patch_count
        return torch.cat(summaries), torch.cat(features, dim=1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_encoder_weights = []
        adaptor_dict = {
            name: param
            for name, param in dict(self.named_parameters()).items()
            if not name.startswith("model_encoder")
        }
        for name, w in weights:
            if name.startswith("model_encoder"):
                model_encoder_weights.append((".".join(name.split(".")[1:]), w))
            else:
                param = adaptor_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, w)

        self.model_encoder.load_weights(model_encoder_weights)


def resize_vision_embeddings(
    embeddings: torch.Tensor,
    num_tokens: int | None,
) -> torch.Tensor:
    """Tile or crop vision embeddings to an experiment-specific length."""
    if num_tokens is None or embeddings.shape[1] == num_tokens:
        return embeddings

    repeats = (num_tokens + embeddings.shape[1] - 1) // embeddings.shape[1]
    return embeddings.repeat(1, repeats, 1)[:, :num_tokens]


@MULTIMODAL_REGISTRY.register_processor(
    NemotronParseMultiModalProcessor,
    info=NemotronParseProcessingInfo,
    dummy_inputs=NemotronParseDummyInputsBuilder,
)
class NemotronParseForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsEagle3
):
    supports_mm_device_do_normalize = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vision_config = config.encoder
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.input_norm = build_mm_input_norm(vllm_config.model_config)
        self.causal_vision_prefix = bool(
            getattr(config.decoder, "causal_vision_prefix", False)
        )
        if self.causal_vision_prefix:
            self.configure_mm_token_handling(
                vocab_size=config.decoder.vocab_size, mm_token_ids=[1]
            )

        with self._mark_tower_model(vllm_config, "image"):
            self.encoder = RadioWithNeck(
                config=config, quant_config=quant_config, prefix=f"{prefix}.encoder"
            )

        with self._mark_language_model(vllm_config):
            self.decoder = MBartDecoderNoPos(
                config.decoder,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.decoder",
            )

        self.vocab_size = config.decoder.vocab_size
        self.lm_head = ParallelLMHead(
            config.decoder.vocab_size, config.decoder.d_model, quant_config=quant_config
        )
        # Some checkpoints (e.g. compact exports) tie the output head to the
        # decoder's input embeddings instead of materializing a separate
        # lm_head.weight tensor.
        tie_word_embeddings = bool(
            getattr(config, "tie_word_embeddings", False)
            or getattr(config.decoder, "tie_word_embeddings", False)
        )
        if tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.decoder.embed_tokens)
        self.decoder._lm_head_ref = weakref.ref(self.lm_head)
        self.logits_processor = LogitsProcessor(
            self.vocab_size, config.decoder.vocab_size
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> NemotronParsePixelInputs | NemotronParsePackedPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        content_sizes = kwargs.pop("content_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            assert isinstance(pixel_values, torch.Tensor)
            if getattr(self.config, "navit_packing", False):
                if image_sizes is None:
                    raise ValueError("Dynamic Parse inputs require image_sizes")
                if content_sizes is None:
                    raise ValueError("Dynamic Parse inputs require content_sizes")
                patch_size = getattr(self.config.encoder, "patch_size", 16)
                return NemotronParsePackedPixelInputs(
                    type="packed_pixel_values",
                    data=pixel_values.unsqueeze(0),
                    image_sizes=image_sizes,
                    content_sizes=content_sizes,
                    resolve_bindings={
                        "n": pixel_values.shape[0],
                        "p": 3 * patch_size**2,
                    },
                )
            h, w = self.config.image_size
            return NemotronParsePixelInputs(
                type="pixel_values",
                data=pixel_values,
                resolve_bindings={
                    "h": h,
                    "w": w,
                },
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: NemotronParsePixelInputs | NemotronParsePackedPixelInputs,
    ) -> MultiModalEmbeddings:
        pixel_values = image_input["data"]
        dtype = next(self.encoder.parameters()).dtype
        flat = self.input_norm(pixel_values.flatten(1), dtype)
        pixel_values = flat.view(pixel_values.shape)
        with torch.cuda.nvtx.range("nemotron_parse.vision_encoder"):
            if image_input["type"] == "packed_pixel_values":
                image_sizes = [
                    (int(height), int(width))
                    for height, width in image_input["image_sizes"].tolist()
                ]
                content_sizes = [
                    (int(height), int(width))
                    for height, width in image_input["content_sizes"].tolist()
                ]
                return self.encoder(
                    pixel_values,
                    image_sizes=image_sizes,
                    content_sizes=content_sizes,
                )
            embeddings = self.encoder(pixel_values)
        return resize_vision_embeddings(
            embeddings,
            getattr(self.config, "num_image_tokens", None),
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_language_model(self) -> nn.Module:
        return self.decoder

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.decoder._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.decoder.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        r"""Args:
            input_ids: torch.Tensor of *decoder* input token ids.
            positions: torch.Tensor of *decoder* position indices.
            encoder_outputs: List of encoder output tensors (vision embeddings).
                During profiling, this may be None or empty.

        Returns:
            Output torch.Tensor

        """
        del positions, kwargs
        if self.causal_vision_prefix and inputs_embeds is None:
            inputs_embeds = self.decoder.embed_input_ids(input_ids)
            if encoder_outputs:
                vision = torch.cat(encoder_outputs, dim=0)
                inputs_embeds[: vision.shape[0]] = vision.to(inputs_embeds.dtype)
        if not self.causal_vision_prefix:
            inputs_embeds = (
                torch.cat(encoder_outputs, dim=0) if encoder_outputs else None
            )
            return self.decoder(
                decoder_input_ids=input_ids,
                encoder_hidden_states=inputs_embeds,
            )
        hidden_states = self.decoder(
            decoder_input_ids=None,
            encoder_hidden_states=None,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        lm_head_dict = dict(self.lm_head.named_parameters())

        def is_encoder(name: str) -> bool:
            return name.startswith("encoder")

        def is_decoder(name: str) -> bool:
            return name.startswith("decoder")

        def is_lm_head(name: str):
            return name.startswith("lm_head")

        # Separate weights by component
        encoder_weights = []
        decoder_weights = []

        for name, w in weights:
            if is_encoder(name):
                encoder_weights.append((".".join(name.split(".")[1:]), w))
            elif is_decoder(name):
                decoder_weights.append((".".join(name.split(".")[1:]), w))
            elif is_lm_head(name):
                trimmed_name = ".".join(name.split(".")[1:])
                param = lm_head_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
            else:
                logger.info("Found unexpected weight: %s", name)

        # Load encoder weights
        self.encoder.load_weights(encoder_weights)
        # Load decoder weights
        self.decoder.load_weights(decoder_weights)


@MULTIMODAL_REGISTRY.register_processor(
    RadioMBartPrefixMultiModalProcessor,
    info=NemotronParseProcessingInfo,
    dummy_inputs=RadioMBartPrefixDummyInputsBuilder,
)
class RadioMBartPrefixForCausalLM(NemotronParseForConditionalGeneration):
    """Native Parse 3.5 causal visual-prefix architecture."""

    # The checkpoint normalizes packed patches on the host.
    supports_mm_device_do_normalize = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        spec = vllm_config.speculative_config
        if (
            spec is not None
            and spec.method == "mtp"
            and not vllm_config.use_v2_model_runner
        ):
            raise ValueError(
                "Nemotron Parse MTP requires Model Runner V2. "
                "Set VLLM_USE_V2_MODEL_RUNNER=1."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<pad>"
        raise ValueError("Only image modality is supported")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Parse 3.5 weights with complete parameter accounting.

        The shared Parse/RADIO loaders predate strict weight tracking and do
        not propagate their delegated load sets.  Returning the fully-prefixed
        set here lets ``DefaultModelLoader`` reject an incomplete checkpoint.
        """
        encoder_weights: list[tuple[str, torch.Tensor]] = []
        decoder_weights: list[tuple[str, torch.Tensor]] = []
        lm_head_params = dict(self.lm_head.named_parameters())
        loaded: set[str] = set()

        for name, weight in weights:
            if name in _PARSE_MTP_WEIGHT_NAMES:
                continue
            if name.startswith("encoder."):
                encoder_weights.append((name.removeprefix("encoder."), weight))
            elif name.startswith("decoder."):
                decoder_weights.append((name.removeprefix("decoder."), weight))
            elif name.startswith("lm_head."):
                short_name = name.removeprefix("lm_head.")
                if short_name not in lm_head_params:
                    raise KeyError(f"Unknown Parse 3.5 LM-head parameter: {short_name}")
                weight_loader = getattr(
                    lm_head_params[short_name], "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_params[short_name], weight)
                loaded.add(name)
            else:
                raise KeyError(f"Unexpected Parse 3.5 checkpoint tensor: {name}")

        encoder_adaptor = {
            name: param
            for name, param in self.encoder.named_parameters()
            if not name.startswith("model_encoder")
        }
        radio_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in encoder_weights:
            if name.startswith("model_encoder."):
                radio_weights.append((name.removeprefix("model_encoder."), weight))
                continue
            if name not in encoder_adaptor:
                raise KeyError(f"Unknown Parse 3.5 encoder parameter: {name}")
            param = encoder_adaptor[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded.add(f"encoder.{name}")

        radio_loaded = self.encoder.model_encoder.load_weights(radio_weights)

        # SO400M is constructed without LayerScale in the reference model, but
        # the shared InternViT implementation materializes identity parameters.
        # Account for them only after verifying both the architecture and value.
        encoder_args = getattr(self.config.encoder, "args", {})
        radio_model_name = (
            encoder_args.get("model")
            if isinstance(encoder_args, Mapping)
            else getattr(encoder_args, "model", None)
        )
        identity_layerscale_models = {
            "vit_so400m_patch16_224",
        }
        radio_source_names = {name for name, _ in radio_weights}
        synthetic_layerscales = {
            name: param
            for name, param in self.encoder.model_encoder.named_parameters()
            if name.endswith((".ls1", ".ls2")) and name not in radio_loaded
        }
        if synthetic_layerscales:
            source_layerscales = {
                name for name in radio_source_names if ".ls1" in name or ".ls2" in name
            }
            if radio_model_name not in identity_layerscale_models:
                raise RuntimeError(
                    "RADIO LayerScale parameters were not loaded for unsupported "
                    f"architecture {radio_model_name!r}: "
                    f"{sorted(synthetic_layerscales)}"
                )
            if source_layerscales:
                raise RuntimeError(
                    "SO400M checkpoint unexpectedly contains LayerScale tensors "
                    f"ignored by the RADIO loader: {sorted(source_layerscales)}"
                )
            non_identity = [
                name
                for name, param in synthetic_layerscales.items()
                if not torch.equal(param.detach(), torch.ones_like(param.detach()))
            ]
            if non_identity:
                raise RuntimeError(
                    "Synthetic SO400M LayerScale parameters are not identity: "
                    f"{non_identity}"
                )
            radio_loaded.update(synthetic_layerscales)

        loaded.update(f"encoder.model_encoder.{name}" for name in radio_loaded)
        decoder_loaded = self.decoder.load_weights(decoder_weights)
        loaded.update(f"decoder.{name}" for name in decoder_loaded)

        if (
            self.config.tie_word_embeddings
            and self.lm_head.weight is not self.decoder.embed_tokens.weight
        ):
            raise RuntimeError("Parse 3.5 tied LM-head weight lost its alias")
        return loaded
