# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GraniteDocling vision-language model.

GraniteDocling converts document page images to DocLang markup. It combines

* a tiled SigLIP-style vision encoder that also returns intermediate features
  (DeepStack taps),
* a pixel-shuffle connector with a coarse path and an optional fine path (four
  times as many image tokens per tile), and
* a dense Granite decoder that adds the projected DeepStack features to the
  hidden states of its first layers.

Parameter names follow the Hugging Face implementation
(``transformers.models.granite_docling``).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature, GraniteDoclingProcessor
from transformers.models.granite_docling.image_processing_granite_docling import (
    get_all_supported_aspect_ratios,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
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
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .granite import GraniteModel
from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)

logger = init_logger(__name__)


class GraniteDoclingImagePixelInputs(TensorSchema):
    """Dimensions:
    - bnp: Batch size * number of images * number of tiles
    - c: Number of channels (3)
    - h: Height
    - w: Width
    - bn: Batch size * number of images
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]
    # True for the tiles that take the fine connector path
    tile_fine_mask: Annotated[torch.Tensor, TensorShape("bnp")]


# -----------------------------------------------------------------------------
# Multimodal processing
# -----------------------------------------------------------------------------


class GraniteDoclingProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> GraniteDoclingProcessor:
        return self.ctx.get_hf_processor(GraniteDoclingProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_kwargs(self, mm_kwargs: Mapping[str, object]) -> dict[str, object]:
        """Per-request kwargs merged with the model's `mm_processor_kwargs`."""
        image_kwargs = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        use_fine_route = self.get_hf_config().vision_config.use_fine_route
        if image_kwargs.get("fine_route", False) and not use_fine_route:
            raise ValueError(
                "This model has no fine connector path "
                "(`vision_config.use_fine_route=False`); `fine_route=True` is "
                "not supported."
            )
        return image_kwargs

    def get_tile_info(
        self,
        *,
        image_width: int,
        image_height: int,
        mm_kwargs: Mapping[str, object],
        processor: GraniteDoclingProcessor | None = None,
    ) -> tuple[int, int, int]:
        """`(num_tiles incl. thumbnail, num_rows, num_cols)` of a page."""
        if processor is None:
            processor = self.get_hf_processor(**mm_kwargs)
        num_rows, num_cols = processor.image_processor.get_tile_grid(
            image_height, image_width, self.get_image_kwargs(mm_kwargs)
        )
        num_tiles = num_rows * num_cols
        # A page that is split into several tiles also gets a thumbnail tile
        return (num_tiles + 1 if num_tiles > 1 else 1), num_rows, num_cols

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: GraniteDoclingProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> str:
        _, num_rows, num_cols = self.get_tile_info(
            image_width=image_width,
            image_height=image_height,
            mm_kwargs=mm_kwargs,
            processor=processor,
        )
        image_inputs: dict[str, object] = {"rows": [[num_rows]], "cols": [[num_cols]]}
        if self.get_image_kwargs(mm_kwargs).get(
            "fine_route", processor.image_processor.fine_route
        ):
            # The processor builds the fine-route prompt when the key is present
            image_inputs["tile_fine_mask"] = None
        return processor.replace_image_token(image_inputs, 0)


class GraniteDoclingDummyInputsBuilder(
    BaseDummyInputsBuilder[GraniteDoclingProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.get_hf_processor().image_token * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_processor = self.info.get_hf_processor().image_processor
        image_kwargs = self.info.get_image_kwargs({})
        # The grid with the most tiles the image processor can select
        min_patches = image_kwargs.get("min_patches", image_processor.min_patches)
        max_patches = image_kwargs.get("max_patches", image_processor.max_patches)
        grids = get_all_supported_aspect_ratios(min_patches, max_patches)
        num_cols, num_rows = max(grids, key=lambda grid: grid[0] * grid[1])
        # A checkpoint with a spread rule (`dense_max_patches` in its image
        # processor config) tiles wide landscape pages with more than
        # `max_patches` tiles: the largest page the processor really selects
        # such a grid for is then the largest input.
        dense_max_patches = (
            image_kwargs.get(
                "dense_max_patches", getattr(image_processor, "dense_max_patches", 0)
            )
            or 0
        )
        if dense_max_patches > max_patches:
            tile = image_processor.size
            dense_grids = get_all_supported_aspect_ratios(
                min_patches, dense_max_patches
            )
            for cols, rows in sorted(dense_grids, key=lambda grid: -grid[0] * grid[1]):
                if cols * rows <= num_cols * num_rows:
                    break
                selected = image_processor.get_tile_grid(
                    tile["height"] * rows, tile["width"] * cols, image_kwargs
                )
                if selected == (rows, cols):
                    num_cols, num_rows = cols, rows
                    break
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_processor.size["width"] * num_cols,
                height=image_processor.size["height"] * num_rows,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class GraniteDoclingMultiModalProcessor(
    BaseMultiModalProcessor[GraniteDoclingProcessingInfo]
):
    def _postprocess_hf_mm_data(
        self,
        mm_data: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
        processed_data: BatchFeature,
    ) -> BatchFeature:
        if not (images := mm_data.get("images", [])):
            return processed_data

        # All images of a prompt are one HF sample: `(1, num_tiles, C, H, W)`
        # without padding tiles.
        pixel_values = processed_data["pixel_values"].flatten(0, 1)
        processed_data["pixel_values"] = pixel_values
        processed_data.pop("pixel_attention_mask", None)

        image_processor = self.info.get_hf_processor(
            **hf_processor_mm_kwargs
        ).image_processor
        fine_route = bool(
            self.info.get_image_kwargs(hf_processor_mm_kwargs).get(
                "fine_route", image_processor.fine_route
            )
        )
        # The HF mask has one value per prompt, the model needs one per tile
        processed_data["tile_fine_mask"] = torch.full(
            (pixel_values.shape[0],), fine_route, dtype=torch.bool
        )

        parsed_images = self.info.parse_mm_data(
            {"image": images}, validate=False
        ).get_items("image", ImageProcessorItems)
        processed_data["num_patches"] = torch.tensor(
            [
                self.info.get_tile_info(
                    image_width=size.width,
                    image_height=size.height,
                    mm_kwargs=hf_processor_mm_kwargs,
                )[0]
                for size in map(parsed_images.get_image_size, range(len(parsed_images)))
            ]
        )
        return processed_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            tile_fine_mask=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_token_id = hf_processor.image_token_id

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            image_repl = self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
                mm_kwargs=hf_processor_mm_kwargs,
            )
            image_repl_ids = cached_encode(
                tokenizer, image_repl, add_special_tokens=False
            )
            return PromptUpdateDetails.select_token_id(image_repl_ids, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


# -----------------------------------------------------------------------------
# Vision encoder that also returns the DeepStack taps
# -----------------------------------------------------------------------------


class GraniteDoclingVisionTransformer(Idefics2VisionTransformer):
    """SigLIP-style encoder returning the residual stream after each layer listed
    in `deepstack_visual_indexes` (before the post-layernorm) next to the final
    hidden states."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix=prefix)
        self.deepstack_visual_indexes = frozenset(config.deepstack_visual_indexes)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Tiles are always full and square: every patch is valid
        num_tiles, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_attention_mask = torch.ones(
            (num_tiles, height // patch_size, width // patch_size),
            dtype=torch.bool,
            device=pixel_values.device,
        )
        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )
        deepstack_features = []
        for layer_idx, layer in enumerate(self.encoder.layers):
            hidden_states = layer(hidden_states)
            if layer_idx in self.deepstack_visual_indexes:
                deepstack_features.append(hidden_states)
        return self.post_layernorm(hidden_states), deepstack_features


# -----------------------------------------------------------------------------
# Connector: vision features -> image tokens (+ DeepStack features)
# -----------------------------------------------------------------------------


def _build_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sine-cosine position embedding of a `grid_size x grid_size`
    token grid in row-major order, laid out `[cols | rows]` like the Hugging Face
    implementation (frequencies are computed in float64)."""
    pos_dim = embed_dim // 4
    omega = 1.0 / 10000 ** (torch.arange(pos_dim, dtype=torch.float64) / pos_dim)
    grid_h, grid_w = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float64),
        torch.arange(grid_size, dtype=torch.float64),
        indexing="ij",
    )
    emb_h = grid_h.flatten().outer(omega)
    emb_w = grid_w.flatten().outer(omega)
    pos_embed = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)
    return pos_embed.to(torch.float32).roll(embed_dim // 2, -1)


def _pixel_shuffle(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Merge each `scale_factor x scale_factor` block of tokens into one token."""
    bsz, seq, embed_dim = x.size()
    height = width = int(seq**0.5)
    x = x.view(bsz, height, width, embed_dim)
    x = x.view(bsz, height, width // scale_factor, embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz,
        width // scale_factor,
        height // scale_factor,
        embed_dim * (scale_factor**2),
    )
    x = x.permute(0, 2, 1, 3)
    return x.reshape(bsz, seq // (scale_factor**2), embed_dim * (scale_factor**2))


class GraniteDoclingDeepStackMerger(nn.Module):
    """Projects pixel-shuffled intermediate vision features to the LM width."""

    def __init__(
        self,
        config,
        scale_factor: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        merged_dim = config.hidden_size * scale_factor**2
        self.layer_norm = nn.LayerNorm(merged_dim)
        self.linear_1 = ReplicatedLinear(
            merged_dim,
            config.out_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "linear_1"),
        )
        self.act = nn.GELU()
        self.linear_2 = ReplicatedLinear(
            config.out_hidden_size,
            config.out_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "linear_2"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x, _ = self.linear_1(x)
        x, _ = self.linear_2(self.act(x))
        return x


class GraniteDoclingConnectorPath(nn.Module):
    """Projection, position embedding and DeepStack mergers for one
    pixel-shuffle factor."""

    def __init__(
        self,
        config,
        scale_factor: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.projection = ReplicatedLinear(
            config.hidden_size * scale_factor**2,
            config.out_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "projection"),
        )
        tokens_per_side = config.image_size // config.patch_size
        # Not part of the checkpoint
        self.register_buffer(
            "pos_embed",
            _build_pos_embed(config.out_hidden_size, tokens_per_side // scale_factor),
            persistent=False,
        )
        self.deepstack_mergers = nn.ModuleList(
            GraniteDoclingDeepStackMerger(
                config,
                scale_factor,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"deepstack_mergers.{i}"),
            )
            for i in range(len(config.deepstack_visual_indexes))
        )

    def get_pos_embed(self, like: torch.Tensor) -> torch.Tensor:
        # Rebuild the table if the module was created on the meta device
        if self.pos_embed.is_meta:
            grid_size = int(self.pos_embed.shape[0] ** 0.5)
            self.pos_embed = _build_pos_embed(self.pos_embed.shape[-1], grid_size)
        return self.pos_embed.to(device=like.device, dtype=like.dtype)


class GraniteDoclingConnector(nn.Module):
    """Turns the vision features of each tile into image tokens, on the coarse
    path (`scale_factor`) or the fine path (half the factor, four times as many
    tokens), and projects the DeepStack taps the same way. The layer norms and
    `mlp_fc2` are shared by both paths."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.out_hidden_size
        self.ln_in = nn.LayerNorm(config.hidden_size)
        self.ln_mid = nn.LayerNorm(hidden_size)
        self.mlp_fc2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "mlp_fc2"),
        )
        self.ln_out = nn.LayerNorm(hidden_size)
        self.coarse = GraniteDoclingConnectorPath(
            config,
            config.scale_factor,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "coarse"),
        )
        self.fine = (
            GraniteDoclingConnectorPath(
                config,
                config.scale_factor // 2,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "fine"),
            )
            if config.use_fine_route
            else None
        )

    def _project(
        self,
        image_hidden_states: torch.Tensor,
        deepstack_hidden_states: Sequence[torch.Tensor],
        fine: bool,
    ) -> torch.Tensor:
        path = self.fine if fine else self.coarse
        if path is None:
            raise ValueError(
                "This model has no fine connector path "
                "(`vision_config.use_fine_route=False`)."
            )
        x = _pixel_shuffle(self.ln_in(image_hidden_states), path.scale_factor)
        x, _ = path.projection(x)
        x = x + path.get_pos_embed(x)
        x = nn.functional.gelu(self.ln_mid(x))
        x, _ = self.mlp_fc2(x)
        x = self.ln_out(x)
        deepstack_features = [
            merger(_pixel_shuffle(hidden.contiguous(), path.scale_factor))
            for merger, hidden in zip(path.deepstack_mergers, deepstack_hidden_states)
        ]
        return torch.cat([x, *deepstack_features], dim=-1)

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        deepstack_hidden_states: Sequence[torch.Tensor],
        tile_fine_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Image tokens of all tiles in tile order, with the DeepStack features
        concatenated along the last dim:
        `[num_image_tokens, hidden * (1 + num_deepstack_levels)]`."""
        if not tile_fine_mask.any():
            return self._project(
                image_hidden_states, deepstack_hidden_states, fine=False
            ).flatten(0, 1)
        if tile_fine_mask.all():
            return self._project(
                image_hidden_states, deepstack_hidden_states, fine=True
            ).flatten(0, 1)

        # Mixed batch: run each path on its tiles, then restore the tile order
        fine_idx = tile_fine_mask.nonzero(as_tuple=True)[0]
        coarse_idx = (~tile_fine_mask).nonzero(as_tuple=True)[0]
        tiles: list[torch.Tensor | None] = [None] * tile_fine_mask.shape[0]
        for idx, fine in ((coarse_idx, False), (fine_idx, True)):
            features = self._project(
                image_hidden_states[idx],
                [hidden[idx] for hidden in deepstack_hidden_states],
                fine=fine,
            )
            for tile_idx, tile_features in zip(idx.tolist(), features):
                tiles[tile_idx] = tile_features
        return torch.cat(tiles, dim=0)


# -----------------------------------------------------------------------------
# Text model: Granite decoder with DeepStack injection
# -----------------------------------------------------------------------------


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class GraniteDoclingTextModel(GraniteModel):
    """`GraniteModel` that adds DeepStack feature `i` to the hidden states of the
    image tokens after decoder layer `i`."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds * self.config.embedding_multiplier

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states)
            if deepstack_input_embeds is not None:
                key = f"deepstack_input_embeds_{layer_idx}"
                if key in deepstack_input_embeds.tensors:
                    hidden_states = hidden_states + deepstack_input_embeds[key]
        return self.norm(hidden_states)


class GraniteDoclingModel(nn.Module):
    """Vision encoder, connector and text model, named like the HF checkpoint
    (`model.vision_model.connector.*` is loaded into `model.connector.*`)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.vision_model = GraniteDoclingVisionTransformer(
            config.vision_config,
            quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )
        self.connector = GraniteDoclingConnector(
            config.vision_config,
            quant_config,
            prefix=maybe_prefix(prefix, "connector"),
        )
        self.text_model = GraniteDoclingTextModel(
            vllm_config=vllm_config.with_hf_config(
                config.text_config, architectures=["GraniteForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "text_model"),
        )


# -----------------------------------------------------------------------------
# Top-level model
# -----------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    GraniteDoclingMultiModalProcessor,
    info=GraniteDoclingProcessingInfo,
    dummy_inputs=GraniteDoclingDummyInputsBuilder,
)
class GraniteDoclingForConditionalGeneration(nn.Module, SupportsMultiModal):
    hf_to_vllm_mapper = WeightsMapper(
        # In HF the connector is a child of the vision model
        orig_to_new_prefix={"model.vision_model.connector.": "model.connector."}
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "GraniteDocling does not support pipeline parallelism."
            )

        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        vision_config = config.vision_config
        self.deepstack_num_level = len(vision_config.deepstack_visual_indexes)
        self.visual_dim = config.text_config.hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        # Image tokens of a tile on the coarse path
        self.image_seq_len = (
            vision_config.image_size // vision_config.patch_size
        ) ** 2 // vision_config.scale_factor**2

        with self._mark_composite_model(
            vllm_config,
            language_targets=(GraniteDoclingTextModel,),
            tower_targets={
                "image": (GraniteDoclingVisionTransformer, GraniteDoclingConnector)
            },
        ):
            self.model = GraniteDoclingModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )
            # DeepStack scratch buffers, one per level, allocated up front so the
            # language model signature is stable from the first warmup run on.
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    self.visual_dim,
                )
                for _ in range(self.deepstack_num_level)
            ]
            # Number of tokens currently staged in the buffers (0: all zero)
            self.deepstack_input_embeds_num_tokens = 0

        text_config = config.text_config
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if text_config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.text_model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            text_config.vocab_size, scale=1.0 / text_config.logits_scaling
        )

    # ---- multimodal embeddings ----

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> GraniteDoclingImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        image_size = self.config.vision_config.image_size
        return GraniteDoclingImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            num_patches=kwargs.pop("num_patches"),
            tile_fine_mask=kwargs.pop("tile_fine_mask"),
            resolve_bindings={"h": image_size, "w": image_size},
        )

    def _process_image_input(
        self, image_input: GraniteDoclingImagePixelInputs
    ) -> tuple[torch.Tensor, ...]:
        pixel_values = image_input["pixel_values"]
        tile_fine_mask = image_input["tile_fine_mask"]
        vision_model = self.model.vision_model
        vision_dtype = vision_model.embeddings.patch_embedding.weight.dtype
        last_hidden_state, deepstack_features = vision_model(
            pixel_values.to(dtype=vision_dtype)
        )
        image_features = self.model.connector(
            last_hidden_state, deepstack_features, tile_fine_mask
        )
        # Split per image: a tile has `image_seq_len` tokens, or four times as
        # many on the fine path.
        num_patches = image_input["num_patches"].tolist()
        num_fine = [int(mask.sum()) for mask in tile_fine_mask.split(num_patches)]
        tokens_per_image = [
            self.image_seq_len * (n + 3 * f) for n, f in zip(num_patches, num_fine)
        ]
        return image_features.split(tokens_per_image)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    # ---- DeepStack scratch buffers (same scheme as Qwen3-VL) ----

    def _get_deepstack_input_embeds(self, num_tokens: int) -> IntermediateTensors:
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self._resize_deepstack_input_embeds(num_tokens)
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _resize_deepstack_input_embeds(self, num_tokens: int) -> None:
        self.deepstack_input_embeds = [
            torch.zeros(
                num_tokens,
                self.visual_dim,
                device=self.deepstack_input_embeds[0].device,
                dtype=self.deepstack_input_embeds[0].dtype,
            )
            for _ in range(self.deepstack_num_level)
        ]

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        """Copy `[num_levels, num_tokens, dim]` embeddings into the buffers."""
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self._resize_deepstack_input_embeds(num_tokens)
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )
        self.deepstack_input_embeds_num_tokens = num_tokens

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # Decode-only steps never stage anything, skip the zeroing kernels
        if self.deepstack_input_embeds_num_tokens == 0:
            return
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()
            self.deepstack_input_embeds_num_tokens = 0

    def _compute_deepstack_embeds(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
    ) -> tuple[torch.Tensor, MultiModalEmbeddings]:
        """Split the DeepStack channels off the multimodal embeddings and scatter
        them to the image-token positions: `[num_levels, seq_len, visual_dim]`."""
        visual_lens = [len(x) for x in multimodal_embeddings]
        main, multiscale = torch.split(
            torch.cat(list(multimodal_embeddings), dim=0),
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )
        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0), self.multiscale_dim
        )
        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multiscale.split(visual_lens, dim=0),
            is_multimodal=is_multimodal,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.size(0), self.deepstack_num_level, self.visual_dim
        ).permute(1, 0, 2)
        return deepstack_input_embeds, main.split(visual_lens, dim=0)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.text_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)
        deepstack_input_embeds, multimodal_embeddings = self._compute_deepstack_embeds(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        deepstack_input_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )

        hidden_states = self.model.text_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )
        # The merge relies on the buffers being zero at text positions
        if inputs_embeds is not None:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Tensors of modules that are not implemented here (the multi-token
        # prediction heads and the density router) are skipped.
        loader = AutoWeightsLoader(
            self, ignore_unexpected_prefixes=["mtp.", "model.density_router."]
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
