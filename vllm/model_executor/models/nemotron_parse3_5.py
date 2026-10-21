# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Standalone causal visual-prefix implementation for Nemotron Parse 3.5.
# Reference checkpoint: https://huggingface.co/nvidia/Nemotron-3.5-Parse/tree/main
#
# Bart classes based on old vLLM codebase:
# https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/bart.py

import math
import weakref
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BartConfig,
    BatchFeature,
    PreTrainedConfig,
)

from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
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
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backend import AttentionType

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


class MBartDecoderLayer(nn.Module):
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
        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.decoder_ffn_dim
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
    ) -> torch.Tensor:
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states)

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
    Each layer is a causal pre-normalized MBart decoder layer.

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
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        r"""Args:
            decoder_input_ids: Indices of *decoder* input sequence tokens in the
                vocabulary. Padding will be ignored by default should you provide it.

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
            # Track destination parameters, not individual QKV source shards.
            # Complete fused weights and biases rely on checkpoint integrity.
            loaded_params.add(name)
        return loaded_params


class NemotronParsePackedPixelInputs(TensorSchema):
    """Packed 16x16 RADIO patches and their per-image pixel geometry."""

    type: Literal["packed_pixel_values"]
    data: Annotated[torch.Tensor, TensorShape(1, "n", "p")]
    image_sizes: torch.Tensor
    content_sizes: torch.Tensor


class NemotronParseProcessingInfo(BaseProcessingInfo):
    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_hf_processor(self, **kwargs: object):
        processor = super().get_hf_processor(**kwargs)
        config = self.get_hf_config()
        image_processor = processor.image_processor
        required_settings = {
            "navit_packing": True,
            "navit_coordinate_frame": "content_normalized",
            "navit_padding_mode": "top_left_white",
            "do_normalize": True,
        }
        for setting, expected in required_settings.items():
            actual = getattr(image_processor, setting, None)
            if actual != expected:
                raise ValueError(
                    f"Parse 3.5 image processor requires {setting}={expected!r}; "
                    f"got {actual!r}. Check the checkpoint processor and "
                    "mm_processor_kwargs."
                )
        if not getattr(config, "navit_packing", False):
            raise ValueError("Parse 3.5 model configuration requires navit_packing")
        return processor

    def get_num_image_tokens(self) -> int:
        config = self.get_hf_config()
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


class RadioMBartPrefixDummyInputsBuilder(
    BaseDummyInputsBuilder[NemotronParseProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<pad>" * mm_counts.get("image", 0)

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


class RadioMBartPrefixMultiModalProcessor(
    BaseMultiModalProcessor[NemotronParseProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        patch_size = self.info.get_hf_config().encoder.patch_size
        image_sizes = hf_inputs.get(
            "image_sizes", torch.empty((0, 2), dtype=torch.long)
        )
        patch_grid = image_sizes // patch_size
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
        def get_replacement(item_idx: int) -> list[int]:
            item = out_mm_kwargs["image"][item_idx]
            image_sizes = item["image_sizes"].data
            assert isinstance(image_sizes, torch.Tensor)
            height, width = image_sizes.tolist()
            patch_size = self.info.get_hf_config().encoder.patch_size
            num_tokens = (height // patch_size) * (width // (4 * patch_size)) + 1
            return [1] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[1],
                replacement=get_replacement,
            )
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
            config, quant_config=quant_config, prefix=f"{prefix}.model_encoder"
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
        prefix: str = "",
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

        return RadioModel(config=radio_config, quant_config=quant_config, prefix=prefix)

    def forward(
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


@MULTIMODAL_REGISTRY.register_processor(
    RadioMBartPrefixMultiModalProcessor,
    info=NemotronParseProcessingInfo,
    dummy_inputs=RadioMBartPrefixDummyInputsBuilder,
)
class RadioMBartPrefixForCausalLM(nn.Module, SupportsMultiModal, SupportsEagle3):
    """Native Parse 3.5 causal visual-prefix architecture."""

    supports_mm_device_do_normalize = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vision_config = config.encoder
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        if not getattr(config.decoder, "causal_vision_prefix", False):
            raise ValueError("Parse 3.5 requires a causal visual prefix")
        if getattr(config.decoder, "add_cross_attention", True):
            raise ValueError("Parse 3.5 does not support decoder cross-attention")
        if not getattr(config, "navit_packing", False):
            raise ValueError("Parse 3.5 requires native packed image processing")
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
            return "<pad>"

        raise ValueError("Only image modality is supported")

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> NemotronParsePackedPixelInputs | None:
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

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: NemotronParsePackedPixelInputs,
    ) -> MultiModalEmbeddings:
        pixel_values = image_input["data"]
        dtype = next(self.encoder.parameters()).dtype
        if not pixel_values.is_floating_point():
            raise ValueError("Parse 3.5 requires host-normalized floating patches")
        image_sizes = [
            (int(height), int(width))
            for height, width in image_input["image_sizes"].tolist()
        ]
        content_sizes = [
            (int(height), int(width))
            for height, width in image_input["content_sizes"].tolist()
        ]
        return self.encoder(
            pixel_values.to(dtype=dtype),
            image_sizes=image_sizes,
            content_sizes=content_sizes,
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
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        del positions, kwargs
        return self.decoder(
            decoder_input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Parse 3.5 weights and report loaded destination parameters.

        Returning fully-prefixed destinations lets ``DefaultModelLoader`` detect
        missing parameters. Individual source shards of fused parameters are
        not independently tracked, matching the common vLLM loader convention.
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
