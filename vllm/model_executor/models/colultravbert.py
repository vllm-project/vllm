# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature, Idefics3ImageProcessor
from transformers.models.idefics3.image_processing_idefics3 import (
    MAX_IMAGE_SIZE,
    _resize_output_size_rescale_to_max_len,
    _resize_output_size_scale_below_upper_bound,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.colultravbert import (
    ColUltraVBertConfig,
    UltraBertConfig,
)
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLateInteraction,
    SupportsMultiModal,
)
from .interfaces_base import default_pooling_type
from .siglip import SiglipVisionModel
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

IMAGE_TOKEN = "<image>"
FAKE_IMAGE_TOKEN = "<fake_token_around_image>"
GLOBAL_IMAGE_TOKEN = "<global-img>"


class UltraBertAttention(nn.Module):
    def __init__(
        self,
        config: UltraBertConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        layer_type = config.layer_types[layer_id]
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters[layer_type],
        )
        # `sliding_window` is an inclusive radius; vLLM's window counts the
        # query token, so radius r becomes r + 1.
        sliding_window = (
            config.sliding_window + 1 if layer_type == "sliding_attention" else None
        )
        self.attn = EncoderOnlyAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class UltraBertMLP(nn.Module):
    def __init__(
        self,
        config: UltraBertConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = get_act_and_mul_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        output, _ = self.down_proj(self.act_fn(gate_up))
        return output


class UltraBertLayer(nn.Module):
    def __init__(
        self,
        config: UltraBertConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = UltraBertAttention(
            config, layer_id, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = UltraBertMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            positions, self.input_layernorm(hidden_states)
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


@support_torch_compile
class UltraBertModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: UltraBertConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                UltraBertLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_id}",
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # UltraBERT feeds token embeddings to its first layer unnormalized.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class ColUltraVBertConnector(nn.Module):
    def __init__(
        self,
        config: ColUltraVBertConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.factor = config.pixel_shuffle_factor
        hidden_size = config.text_config.hidden_size
        self.modality_projection = ReplicatedLinear(
            config.vision_config.hidden_size * self.factor**2,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "modality_projection"),
        )
        self.norm = (
            RMSNorm(hidden_size, eps=config.text_config.rms_norm_eps)
            if config.connector_norm
            else nn.Identity()
        )

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        side, f = math.isqrt(length), self.factor
        x = x.reshape(batch, side // f, f, side // f, f, dim)
        x = x.permute(0, 1, 3, 2, 4, 5)
        return x.reshape(batch, length // (f * f), dim * f * f)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        projected, _ = self.modality_projection(self.pixel_shuffle(image_hidden_states))
        return self.norm(projected)


def get_capped_longest_edge(
    height: int, width: int, longest_edge: int, tile_size: int, max_image_tiles: int
) -> int:
    # Counts tiles with the tiling's own resize arithmetic, as transformers'
    # UltraVBertImageProcessor does; a closed-form count disagrees at tile
    # boundaries.
    for edge in range(longest_edge, tile_size - 1, -tile_size):
        h, w = _resize_output_size_rescale_to_max_len(height, width, max_len=edge)
        h, w = _resize_output_size_scale_below_upper_bound(h, w, max_len=MAX_IMAGE_SIZE)
        aspect_ratio = w / h
        if w >= h:
            cols = math.ceil(w / tile_size)
            rows = math.ceil(int(cols * tile_size / aspect_ratio) / tile_size)
        else:
            rows = math.ceil(h / tile_size)
            cols = math.ceil(int(rows * tile_size * aspect_ratio) / tile_size)
        if rows * cols <= max_image_tiles:
            return edge
    return tile_size


def image_prompt(rows: int, cols: int, image_seq_len: int) -> str:
    tile = IMAGE_TOKEN * image_seq_len
    global_view = f"{FAKE_IMAGE_TOKEN}{GLOBAL_IMAGE_TOKEN}{tile}{FAKE_IMAGE_TOKEN}"
    if rows == 0 and cols == 0:
        return global_view
    grid = "".join(
        "".join(
            f"{FAKE_IMAGE_TOKEN}<row_{r}_col_{c}>{tile}" for c in range(1, cols + 1)
        )
        + "\n"
        for r in range(1, rows + 1)
    )
    return f"{grid}\n{global_view}"


class ColUltraVBertImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class ColUltraVBertProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ColUltraVBertConfig:
        return self.ctx.get_hf_config(ColUltraVBertConfig)

    def get_image_processor(self) -> Idefics3ImageProcessor:
        model_config = self.ctx.model_config
        return cached_get_image_processor(
            model_config.model,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            processor_cls_overrides=Idefics3ImageProcessor,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_longest_edge(self, *, image_width: int, image_height: int) -> int:
        image_processor = self.get_image_processor()
        edge = image_processor.size["longest_edge"]
        max_image_tiles = getattr(image_processor, "max_image_tiles", None)
        if not max_image_tiles:
            return edge
        return get_capped_longest_edge(
            image_height,
            image_width,
            edge,
            image_processor.max_image_size["longest_edge"],
            max_image_tiles,
        )

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        edge = self.get_longest_edge(image_width=image_width, image_height=image_height)
        _, rows, cols = self.get_image_processor().get_number_of_image_patches(
            image_height, image_width, {"size": {"longest_edge": edge}}
        )
        return (rows * cols + 1) * self.get_hf_config().image_seq_len

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        edge = image_processor.size["longest_edge"]
        tile = image_processor.max_image_size["longest_edge"]
        sizes = [ImageSize(width=edge, height=h) for h in range(tile, edge + 1, tile)]
        return max(
            sizes,
            key=lambda s: self.get_num_image_tokens(
                image_width=s.width, image_height=s.height
            ),
        )


class ColUltraVBertDummyInputsBuilder(
    BaseDummyInputsBuilder[ColUltraVBertProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class ColUltraVBertMultiModalProcessor(
    BaseMultiModalProcessor[ColUltraVBertProcessingInfo],
):
    def _process_images(self, images: list, **kwargs: object) -> BatchFeature:
        items = ImageProcessorItems(images)
        image_processor = self.info.get_image_processor()
        pixel_values, grids = [], []
        for i in range(len(items)):
            size = items.get_image_size(i)
            edge = self.info.get_longest_edge(
                image_width=size.width, image_height=size.height
            )
            outputs = image_processor(
                [[items.get(i)]],
                size={"longest_edge": edge},
                return_row_col_info=True,
                return_tensors="pt",
            )
            pixel_values.append(outputs["pixel_values"][0])
            grids.append([outputs["rows"][0][0], outputs["cols"][0][0]])
        return BatchFeature(
            {
                "pixel_values": torch.cat(pixel_values),
                "num_patches": torch.tensor([len(p) for p in pixel_values]),
                "image_grid": torch.tensor(grids),
            }
        )

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return self.info.ctx.call_hf_processor(self._process_images, hf_data)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            image_grid=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            grid = out_mm_kwargs["image"][item_idx]["image_grid"].data
            rows, cols = grid.tolist()
            text = image_prompt(rows, cols, config.image_seq_len)
            token_ids = cached_encode(tokenizer, text, add_special_tokens=False)
            return PromptUpdateDetails.select_token_id(token_ids, config.image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[config.image_token_id],
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    ColUltraVBertMultiModalProcessor,
    info=ColUltraVBertProcessingInfo,
    dummy_inputs=ColUltraVBertDummyInputsBuilder,
)
@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColUltraVBertForRetrieval(nn.Module, SupportsMultiModal, SupportsLateInteraction):
    is_pooling_model = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # transformers >= 5.10 saves SigLIP without its inner `vision_model.`
    # prefix; the longer prefix must stay first.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vision_model.vision_model.": "vision_model.vision_model.",
            "vision_model.": "vision_model.vision_model.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: ColUltraVBertConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = SiglipVisionModel(
                config.vision_config,
                quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.connector = ColUltraVBertConnector(
                config, quant_config, prefix=maybe_prefix(prefix, "connector")
            )

        with self._mark_language_model(vllm_config):
            self.text_model = UltraBertModel(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "text_model"),
            )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_token_embed(pooler_config)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> ColUltraVBertImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        size = self.config.vision_config.image_size
        return ColUltraVBertImagePixelInputs(
            pixel_values=pixel_values,
            num_patches=kwargs.pop("num_patches"),
            resolve_bindings={"h": size, "w": size},
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        pixel_values = image_input["pixel_values"]
        features = self.connector(
            self.vision_model(pixel_values.to(dtype=self.vision_model.dtype))
        )
        num_patches = image_input["num_patches"].tolist()
        return [f.flatten(0, 1) for f in features.split(num_patches)]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.text_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        projector = getattr(self.pooler.head, "projector", None)
        if isinstance(projector, nn.Module):
            loaded |= {
                f"pooler.head.projector.{name}"
                for name, _ in projector.named_parameters()
            }
        return loaded
