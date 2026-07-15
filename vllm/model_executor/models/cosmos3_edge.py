# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import ProcessorMixin

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from .lfm2_siglip2 import Siglip2VisionTransformer
from .nemotron_h import (
    NemotronHAttention,
    NemotronHMLPDecoderLayer,
    NemotronHModel,
)
from .qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from .vision import (
    is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model,
)


class Cosmos3EdgeVisionEncoder(Siglip2VisionTransformer):
    """Adapts Cosmos (T, H, W) metadata to vLLM packed SigLIP2."""

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # SigLIP2 attends to each frame independently, so expand every THW
        # entry into T separate HW attention sequences.
        grid_thw_cpu = grid_thw.to(device="cpu")
        spatial_shapes = torch.repeat_interleave(
            grid_thw_cpu[:, 1:],
            grid_thw_cpu[:, 0],
            dim=0,
        )
        lengths_cpu = spatial_shapes.prod(dim=-1).to(torch.int32)
        lengths = lengths_cpu.to(
            device=pixel_values.device,
            non_blocking=True,
        )

        cu_seqlens = torch.zeros(
            lengths.numel() + 1,
            dtype=torch.int32,
            device=pixel_values.device,
        )
        cu_seqlens[1:] = lengths.cumsum(dim=0)

        max_seqlen = lengths_cpu.max().reshape(1)

        return super().forward(
            pixel_values_packed=pixel_values,
            spatial_shapes=spatial_shapes,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )[0]


def patch_merging_by_param(
    image_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_size: int = 2,
) -> torch.Tensor:
    """Merge each spatial ``merge_size`` block into the channel dimension."""
    merged_embeds: list[torch.Tensor] = []
    current_idx = 0
    hidden_size = image_embeds.shape[-1]

    for grid in grid_thw:
        t, h, w = grid.tolist()
        if h % merge_size != 0 or w % merge_size != 0:
            raise ValueError(
                "Grid height and width must be divisible by merge_size: "
                f"got grid={(t, h, w)} and merge_size={merge_size}."
            )

        num_patches = t * h * w
        media_embeds = image_embeds[current_idx : current_idx + num_patches]
        if media_embeds.shape[0] != num_patches:
            raise ValueError(
                "image_embeds contains fewer patches than grid_thw describes."
            )
        current_idx += num_patches

        media_embeds = media_embeds.view(t, h, w, hidden_size)
        media_embeds = media_embeds.view(
            t,
            h // merge_size,
            merge_size,
            w // merge_size,
            merge_size,
            hidden_size,
        )
        media_embeds = media_embeds.permute(0, 1, 3, 2, 4, 5).contiguous()
        merged_embeds.append(
            media_embeds.view(-1, merge_size * merge_size * hidden_size)
        )

    if current_idx != image_embeds.shape[0]:
        raise ValueError(
            "image_embeds contains more patches than grid_thw describes: "
            f"got {image_embeds.shape[0]}, expected {current_idx}."
        )
    if not merged_embeds:
        return image_embeds.new_empty((0, merge_size * merge_size * hidden_size))
    return torch.cat(merged_embeds, dim=0)


class Cosmos3EdgePatchMerger(nn.Module):
    """
    Projector: LayerNorm -> Linear -> GELU -> Linear

    Reads config from projector_config (not vision_config).
    input_hidden_size * spatial_merge_size² -> merger_intermediate_size
    -> out_hidden_size
    """

    def __init__(
        self,
        input_hidden_size: int,
        out_hidden_size: int,
        merger_intermediate_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = input_hidden_size * (spatial_merge_size**2)
        self.input_hidden_size = input_hidden_size
        self.out_hidden_size = out_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.hidden_size if use_postshuffle_norm else input_hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)

        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            merger_intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            merger_intermediate_size,
            out_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x, _ = self.linear_fc1(x)
        x = self.act_fn(x)
        x, _ = self.linear_fc2(x)
        return x


class Cosmos3EdgeVisionModel(nn.Module):
    """Complete Cosmos vision tower returning language-model embeddings."""

    def __init__(
        self,
        vision_config,
        projector_config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = projector_config.spatial_merge_size
        self.out_hidden_size = projector_config.out_hidden_size

        self.encoder = Cosmos3EdgeVisionEncoder(
            vision_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "encoder"),
        )
        self.projector = Cosmos3EdgePatchMerger(
            input_hidden_size=projector_config.input_hidden_size,
            out_hidden_size=projector_config.out_hidden_size,
            merger_intermediate_size=projector_config.merger_intermediate_size,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=getattr(
                projector_config, "use_postshuffle_norm", False
            ),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "projector"),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        grid_thw = torch.as_tensor(grid_thw, dtype=torch.int64, device="cpu")
        image_embeds = self.encoder(pixel_values.type(self.dtype), grid_thw=grid_thw)
        image_embeds = patch_merging_by_param(
            image_embeds,
            grid_thw,
            merge_size=self.spatial_merge_size,
        )
        image_embeds = image_embeds.view(
            -1,
            self.spatial_merge_size**2,
            self.projector.input_hidden_size,
        )
        return self.projector(image_embeds)


class Cosmos3EdgeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        return self.ctx.get_hf_processor(**kwargs)


class Cosmos3EdgeMultiModalProcessor(Qwen3VLMultiModalProcessor):
    pass


class Cosmos3EdgeDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    pass


class Cosmos3EdgeAttention(NemotronHAttention):
    """Nemotron-H attention with interleaved multimodal RoPE."""

    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config=None,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            prefix,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Cosmos3EdgeAttentionDecoderLayer(nn.Module):
    """Pre-norm attention layer for the Cosmos3 Edge dense text model."""

    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config=None,
        cache_config=None,
        quant_config=None,
        parallel_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
        layer_config = get_layer_config(layer_idx) if get_layer_config else config
        self.mixer = Cosmos3EdgeAttention(
            layer_config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.mixer",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        hidden_states = self.mixer(positions=positions, hidden_states=hidden_states)
        return hidden_states, residual


class Cosmos3EdgeTextModel(NemotronHModel):
    """Nemotron-H backbone with Cosmos-owned checkpoint name mapping."""

    # The enclosing Cosmos model maps and packs checkpoint weights. Disabling
    # Nemotron-H's checkpoint-specific hook prevents those names from being
    # remapped a second time when AutoWeightsLoader descends into this module.
    load_weights = None  # type: ignore[assignment]


class Cosmos3EdgeForCausalLM(nn.Module):
    """Minimal CausalLM wrapper for the Cosmos3 Edge dense text model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        vllm_config = vllm_config.with_hf_config(text_config)

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = text_config
        pattern = text_config.hybrid_override_pattern
        unsupported = set(pattern) - {"*", "-"}
        if unsupported:
            raise ValueError(
                "Cosmos3 Edge only supports attention (`*`) and MLP (`-`) "
                f"layers, but found: {sorted(unsupported)}"
            )
        if len(pattern) != text_config.num_hidden_layers:
            raise ValueError(
                "hybrid_override_pattern must contain one entry per layer: "
                f"expected {text_config.num_hidden_layers}, "
                f"got {len(pattern)}"
            )

        self.model = Cosmos3EdgeTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            decoder_layer_types={
                "*": Cosmos3EdgeAttentionDecoderLayer,
                "-": NemotronHMLPDecoderLayer,
            },
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(text_config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)


def _cosmos3_edge_diffusers_prefix_map() -> dict[str, str]:
    """Map Diffusers blocks to interleaved Nemotron-H layers.

    The checkpoint stores attention and MLP in one ``layers.N`` block, while
    the Nemotron-H model represents them as separate decoder layers. Therefore,
    checkpoint block ``N`` maps its attention to layer ``2N`` and its MLP to
    layer ``2N + 1``.
    """
    mappings = {
        "embed_tokens.": "language_model.model.embed_tokens.",
        "norm.": "language_model.model.norm_f.",
    }
    for physical_idx in range(28):
        attention_idx = 2 * physical_idx
        mlp_idx = attention_idx + 1
        source_prefix = f"layers.{physical_idx}"
        attention_prefix = f"language_model.model.layers.{attention_idx}"
        mlp_prefix = f"language_model.model.layers.{mlp_idx}"
        mappings.update(
            {
                f"{source_prefix}.input_layernorm.": f"{attention_prefix}.norm.",
                f"{source_prefix}.self_attn.qkv_proj.": (
                    f"{attention_prefix}.mixer.qkv_proj."
                ),
                f"{source_prefix}.self_attn.o_proj.": (
                    f"{attention_prefix}.mixer.o_proj."
                ),
                f"{source_prefix}.post_attention_layernorm.": (f"{mlp_prefix}.norm."),
                f"{source_prefix}.mlp.": f"{mlp_prefix}.mixer.",
            }
        )
    return mappings


@MULTIMODAL_REGISTRY.register_processor(
    Cosmos3EdgeMultiModalProcessor,
    info=Cosmos3EdgeProcessingInfo,
    dummy_inputs=Cosmos3EdgeDummyInputsBuilder,
)
class Cosmos3EdgeForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
):
    """
    Cosmos3 Edge model with a SigLIP2 vision encoder.

    Architecture:
        - self.visual: SigLIP2 encoder + patch merger + projector
        - self.language_model: Cosmos3EdgeForCausalLM (pure attention + RoPE)
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
        },
        orig_to_new_prefix={
            **_cosmos3_edge_diffusers_prefix_map(),
            "proj_in.": None,
            "proj_out.": None,
            "time_embedder.": None,
            "action_proj_in.": None,
            "action_proj_out.": None,
            "audio_modality_embed": None,
            "action_modality_embed": None,
            "model.visual.": "visual.encoder.",
            "model.projector.": "visual.projector.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        },
        orig_to_new_substr={
            "_moe_gen": None,
            ".add_q_proj.": None,
            ".add_k_proj.": None,
            ".add_v_proj.": None,
            ".to_add_out.": None,
            ".norm_added_q.": None,
            ".norm_added_k.": None,
            ".self_attn.to_q.": ".self_attn.q_proj.",
            ".self_attn.to_k.": ".self_attn.k_proj.",
            ".self_attn.to_v.": ".self_attn.v_proj.",
            ".self_attn.to_out.": ".self_attn.o_proj.",
            "language_model.embeddings": "language_model.embed_tokens",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    # Cosmos3 Edge unified Diffusers checkpoints store reasoner weights across
    # transformer/ and vision_encoder/. Match both while excluding VAE weights.
    allow_patterns_overrides = ["[tv]*er/*.safetensors"]

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Cosmos3EdgeVisionModel(
                vision_config=config.vision_config,
                projector_config=config.projector_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Cosmos3EdgeForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the complete vision tower and split by media item."""
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return image_embeds.split(sizes)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
            merge_size = self.visual.spatial_merge_size
            sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
            return image_embeds.split(sizes)

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            )
        return self._get_image_features(pixel_values, grid_thw)

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
            merge_size = self.visual.spatial_merge_size
            sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
            return video_embeds.split(sizes)

        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values_videos, grid_thw.tolist(), rope_type="rope_3d"
            )
        return self._get_image_features(pixel_values_videos, grid_thw)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            modalities["image"] = image_input
        video_input = self._parse_and_validate_video_input(**kwargs)
        if video_input is not None:
            modalities["video"] = video_input
        return modalities

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.projector",
            tower_model="visual.",
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        return Qwen3VLForConditionalGeneration._get_mrope_input_positions(
            input_tokens=input_tokens,
            mm_features=mm_features,
            config=self.config,
        )
