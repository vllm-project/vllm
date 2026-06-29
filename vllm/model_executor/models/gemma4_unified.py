# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 Unified multimodal model (encoder-free image + audio + video).

The Unified Gemma4 variant has no SigLIP vision tower and no audio tower.
Raw pixel patches are projected directly to LM space via a Dense+LayerNorm
pipeline with factorized 2D positional embeddings (Gemma4UnifiedVisionEmbedder),
then routed through the same Gemma4MultimodalEmbedder used by the tower-based
variant.  Audio inputs are raw waveform frames projected directly through the
multimodal embedder.

This module subclasses Gemma4ForConditionalGeneration from gemma4_mm rather
than reimplementing it from scratch.  Only the multimodal pipeline differs;
the language model, MTP integration, bidirectional attention helpers,
embedding/forward path, and LoRA support are all inherited unchanged.
"""

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedConfig,
)
from transformers.models.gemma4_unified.processing_gemma4_unified import (
    Gemma4UnifiedProcessor,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import VideoDummyOptions
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import (
    _SUPPORTED_SOFT_TOKENS,
    _VIDEO_MAX_FRAMES,
    _VIDEO_MAX_SOFT_TOKENS,
    Gemma4AudioInputs,
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4ImageInputs,
    Gemma4ImagePixelInputs,
    Gemma4MultimodalEmbedder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
    _get_max_soft_tokens,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# Re-export so tests/code targeting the unified variant can import from here
# rather than reaching into gemma4_mm.
__all__ = [
    "Gemma4ImagePixelInputs",
    "Gemma4UnifiedVisionEmbedder",
    "Gemma4UnifiedProcessingInfo",
    "Gemma4UnifiedForConditionalGeneration",
]


# ---------------------------------------------------------------------------
# Encoder-free vision embedder
# ---------------------------------------------------------------------------


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free vision embedder for Gemma4 Unified variants.

    Projects raw pixel patches to LM space via dense projection and
    factorized 2D positional embeddings.  Replaces the SigLIP vision
    tower used by the tower-based Gemma4 variant.

    Pipeline: raw patches → LN₁ → Dense → LN₂ → +factorized_posemb → LN₃.
    """

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        patch_dim = config.model_patch_size**2 * 3
        mm_embed_dim = config.mm_embed_dim

        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = ColumnParallelLinear(
            patch_dim,
            mm_embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.patch_dense",
            gather_output=True,
        )
        self.patch_ln2 = nn.LayerNorm(mm_embed_dim)

        self.pos_embedding = nn.Parameter(
            torch.zeros(config.mm_posemb_size, 2, mm_embed_dim)
        )
        self.pos_norm = nn.LayerNorm(mm_embed_dim)

    def _factorized_posemb(self, positions_xy: torch.Tensor) -> torch.Tensor:
        clamped_pos = positions_xy.clamp(min=0).long()
        valid_mask = positions_xy != -1

        pos_embs = torch.zeros(
            *positions_xy.shape[:-1],
            self.pos_embedding.shape[-1],
            device=positions_xy.device,
            dtype=self.pos_embedding.dtype,
        )
        for i in range(2):
            axis_pe = self.pos_embedding[:, i, :][clamped_pos[..., i]]
            mask = valid_mask[..., i].unsqueeze(-1).to(axis_pe.dtype)
            pos_embs = pos_embs + (axis_pe * mask)
        return pos_embs

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_ln1(pixel_values.to(self.pos_embedding.dtype))
        hidden_states, _ = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        pos_embs = self._factorized_posemb(pixel_position_ids)
        hidden_states = hidden_states + pos_embs
        hidden_states = self.pos_norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Processing info
# ---------------------------------------------------------------------------


class Gemma4UnifiedProcessingInfo(Gemma4ProcessingInfo):
    """ProcessingInfo for the Gemma4 Unified variant.

    Two field-name differences from the tower-based parent:
      * config → ``Gemma4UnifiedConfig`` (not ``Gemma4Config``)
      * vision_config.``num_soft_tokens`` (not ``default_output_length``)

    Everything else (token sequencing, audio limits, video frame budget,
    parser construction) is inherited unchanged.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma4UnifiedConfig)

    def get_hf_processor(self, **kwargs: object) -> Gemma4UnifiedProcessor:
        return self.ctx.get_hf_processor(
            Gemma4UnifiedProcessor,
            **kwargs,
        )

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        config = self.get_hf_config()
        # Unified field is `num_soft_tokens`.  Tower-based parent uses
        # `default_output_length`, hence the override.
        tokens_per_image = config.vision_config.num_soft_tokens
        merged_kwargs = self.ctx.get_merged_mm_kwargs({})
        val, _ = _get_max_soft_tokens(merged_kwargs)
        if isinstance(val, int) and val in _SUPPORTED_SOFT_TOKENS:
            tokens_per_image = val
        tokens: dict[str, int] = {"image": tokens_per_image}
        if config.audio_config is not None:
            processor = self.get_hf_processor()
            tokens["audio"] = processor.audio_seq_length
        num_frames = _VIDEO_MAX_FRAMES
        mm_config = self.ctx.model_config.get_multimodal_config()
        video_opts = mm_config.limit_per_prompt.get("video")
        if (
            isinstance(video_opts, VideoDummyOptions)
            and video_opts.num_frames is not None
        ):
            num_frames = min(num_frames, video_opts.num_frames)
        tokens["video"] = num_frames * (_VIDEO_MAX_SOFT_TOKENS + 2 + 6)
        return tokens

    def _compute_num_soft_tokens(
        self,
        image_width: int,
        image_height: int,
        max_soft_tokens: int | None = None,
    ) -> int:
        vision_cfg = self.get_hf_config().vision_config
        patch_size = vision_cfg.patch_size
        pooling_kernel_size = vision_cfg.pooling_kernel_size

        if max_soft_tokens is None:
            max_soft_tokens = vision_cfg.num_soft_tokens

        unit = patch_size * pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2
        num_patches_orig = (image_height / patch_size) * (image_width / patch_size)
        scale = math.sqrt(max_patches / num_patches_orig)
        target_h = max(unit, int(math.floor(image_height * scale / unit)) * unit)
        target_w = max(unit, int(math.floor(image_width * scale / unit)) * unit)
        num_patches = (target_h // patch_size) * (target_w // patch_size)
        num_soft_tokens = num_patches // (pooling_kernel_size**2)
        return min(num_soft_tokens, max_soft_tokens)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4UnifiedProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4UnifiedForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Encoder-free Gemma4 (Unified) for conditional generation.

    Inherits multimodal embedding routing, PLE handling, bidirectional
    attention helpers, language-model forward, LoRA, and pipeline-parallel
    support from :class:`Gemma4ForConditionalGeneration`.  Overrides only:

      * ``__init__`` — builds the encoder-free vision embedder instead of
        SigLIP/audio towers (LightOnOCR-style: ``nn.Module.__init__`` +
        full rebuild, no ``super().__init__()``).
      * ``hf_to_vllm_mapper`` — adds the ``model.vision_embedder.`` prefix.
      * ``_process_image_input`` / ``_process_video_input`` /
        ``_process_audio_input`` — encoder-free projection paths.
      * ``load_weights`` — ignore-prefix list excludes the absent towers.
      * ``get_mm_mapping`` — no tower entries.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.embed_audio.": "embed_audio.",
            "model.embed_vision.": "embed_vision.",
            "model.language_model.": "language_model.model.",
            "model.vision_embedder.": "vision_embedder.",
            "lm_head.": "language_model.lm_head.",
            "model": "language_model.model",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # LightOnOCR-style rebuild: do NOT call super().__init__ — that
        # would build a SigLIP vision tower and an audio tower we don't
        # need.  Initialize nn.Module directly and assemble the
        # encoder-free pipeline below.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        # No towers — set to None so inherited load_weights / get_mm_mapping
        # and any tower-aware logic short-circuits.
        self.vision_tower = None
        self.audio_tower = None

        # ---- Encoder-free vision embedder ----
        self.vision_embedder = (
            Gemma4UnifiedVisionEmbedder(
                config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_embedder"),
            )
            if config.vision_config is not None
            else None
        )
        self.embed_vision = (
            Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
            )
            if config.vision_config is not None
            else None
        )

        # ---- Encoder-free audio embedder ----
        self.embed_audio = (
            Gemma4MultimodalEmbedder(
                config.audio_config,
                config.text_config,
            )
            if config.audio_config is not None
            else None
        )

        # ---- Language model (vLLM optimised) ----
        with self._mark_language_model(vllm_config):
            self.language_model: Gemma4ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma4ForCausalLM"],
            )

            # PLE is disabled for the unified variant (text config defaults
            # hidden_size_per_layer_input to 0).  Skip the buffer.
            ple_dim = getattr(
                config.text_config,
                "hidden_size_per_layer_input",
                None,
            )
            if ple_dim is not None and ple_dim > 0:
                embed = self.language_model.model.embed_tokens
                self.per_layer_embeddings = torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.num_hidden_layers,
                    ple_dim,
                    device=next(embed.parameters()).device,
                    dtype=vllm_config.model_config.dtype,
                )
            else:
                self.per_layer_embeddings = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # --- Precompute full-attention layer indices for bidi clearing ---
        self._full_attn_layer_idxs: frozenset[int] = frozenset()
        text_config = config.text_config
        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types:
                self._full_attn_layer_idxs = frozenset(
                    i for i, lt in enumerate(layer_types) if lt != "sliding_attention"
                )

        # --- MixtureOfExperts delegation to language_model ---
        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = self.language_model.num_moe_layers
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_expert_groups = self.language_model.num_expert_groups
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts
        self.set_eplb_state = self.language_model.set_eplb_state

        gen_cfg = vllm_config.model_config.try_get_generation_config()
        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None

    # ------------------------------------------------------------------ #
    # Multimodal processing (encoder-free overrides)
    # ------------------------------------------------------------------ #

    def _process_image_input(
        self,
        image_input: Gemma4ImageInputs,
    ) -> list[torch.Tensor]:
        """Project raw image patches directly to LM space.

        No vision tower: each image's pre-patchified pixel values are
        embedded via Gemma4UnifiedVisionEmbedder, projected through
        Gemma4MultimodalEmbedder, and padding patches (pp == -1) are
        stripped per image.
        """
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]
        target_dtype = self.embed_vision.embedding_projection.weight.dtype

        per_image_features: list[torch.Tensor] = []
        for pv, pp in zip(pixel_values, pixel_position_ids, strict=True):
            pv = pv.unsqueeze(0)
            pp = pp.unsqueeze(0)
            embedded = self.vision_embedder(pv, pp)
            projected = self.embed_vision(embedded.to(target_dtype))
            padding_mask = (pp.squeeze(0) == -1).all(dim=-1)
            valid_features = projected.squeeze(0)[~padding_mask]
            per_image_features.append(valid_features)
        return per_image_features

    def _process_video_input(
        self,
        video_input: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Project video frames to LM space, one frame at a time.

        Frames are split per video, each frame is embedded + projected,
        and per-frame valid embeddings are concatenated per video.
        """
        pixel_values = video_input["pixel_values_videos"]
        pixel_position_ids = video_input["pixel_position_ids_videos"]
        frame_counts = video_input["video_frame_counts"]
        target_dtype = self.embed_vision.embedding_projection.weight.dtype

        if isinstance(frame_counts, torch.Tensor):
            fc_list = frame_counts.tolist()
        else:
            fc_list = list(frame_counts)

        pv_per_video = torch.split(pixel_values, fc_list, dim=0)
        pp_per_video = torch.split(pixel_position_ids, fc_list, dim=0)

        per_video_embeddings: list[torch.Tensor] = []
        for pv_chunk, pp_chunk in zip(pv_per_video, pp_per_video):
            frame_embs: list[torch.Tensor] = []
            for i in range(pv_chunk.shape[0]):
                pv = pv_chunk[i].unsqueeze(0)
                pp = pp_chunk[i].unsqueeze(0)
                embedded = self.vision_embedder(pv, pp)
                projected = self.embed_vision(embedded.to(target_dtype))
                padding_mask = (pp.squeeze(0) == -1).all(dim=-1)
                frame_embs.append(projected.squeeze(0)[~padding_mask])
            per_video_embeddings.append(torch.cat(frame_embs, dim=0))
        return per_video_embeddings

    def _process_audio_input(
        self,
        audio_input: Gemma4AudioInputs,
    ) -> list[torch.Tensor]:
        """Project raw waveform-frame features directly to LM space.

        No audio tower: the per-frame raw features are passed straight
        through the multimodal embedder, then padding is stripped.
        """
        input_features = audio_input["input_features_padded"].squeeze(1)
        input_features_mask = audio_input["input_features_mask"].squeeze(1)

        target_dtype = self.embed_audio.embedding_projection.weight.dtype
        audio_features = self.embed_audio(input_features.to(target_dtype))
        per_audio: list[torch.Tensor] = []
        for enc, mask in zip(audio_features, input_features_mask, strict=True):
            per_audio.append(enc[mask])
        return per_audio

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ignore_prefixes = [
            # Vestigial Gemma3n-style embedding tables not used by
            # Gemma4MultimodalEmbedder (which has only projection + norm).
            "embed_vision.embedding.",
            "embed_audio.embedding.",
        ]
        if self.embed_audio is None:
            ignore_prefixes.append("embed_audio.")

        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=ignore_prefixes,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # ------------------------------------------------------------------ #
    # LoRA / multimodal mapping
    # ------------------------------------------------------------------ #

    def get_mm_mapping(self) -> MultiModelKeys:
        """Module prefix mapping for the encoder-free model (no towers)."""
        connectors = ["embed_vision"]
        if self.embed_audio is not None:
            connectors.append("embed_audio")
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=connectors,
            tower_model=[],
        )
