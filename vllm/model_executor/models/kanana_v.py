# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import numpy as np
import regex as re
import torch
from einops import rearrange
from PIL import Image
from timm.layers import LayerNorm, LayerNorm2d
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.regnet import RegStage
from torch import nn
from transformers import BatchFeature
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .llama import LlamaForCausalLM
from .qwen2_vl import Qwen2VisionTransformer
from .utils import AutoWeightsLoader, _merge_multimodal_embeddings

logger = init_logger(__name__)


def build_pos_embeds(
    config: Qwen2VLVisionConfig,
    num_input_tokens: int,
    vision_hidden_size: int,
) -> nn.Parameter | None:
    """Build positional embeddings for the visual encoder output."""
    if config.pos_emb:
        pos_emb = nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(
    config: Qwen2VLVisionConfig,
    output_hidden_size: int,
) -> nn.Parameter | None:
    """Build extra EOS / \"think\" tokens optionally appended after vision tokens."""
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: Qwen2VLVisionConfig) -> LayerNorm | None:
    """Optionally build a LayerNorm applied before the projector."""
    if getattr(config, "prenorm", False):
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(
    depth: int,
    hidden_size: int,
    output_hidden_size: int,
) -> nn.Sequential:
    """Simple SiLU-activated MLP used as a projector readout."""
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class PatchMerge(nn.Module):
    """Merge neighboring patches spatially to reduce resolution."""

    def __init__(self, merge_size: int) -> None:
        super().__init__()
        self.merge_size = merge_size

    def forward(
        self,
        x: torch.Tensor,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """Merge patches by `merge_size x merge_size`."""
        if channel_last:
            x = rearrange(x, "B H W D -> B D H W")
        _, _, H, W = x.shape
        merged_x = rearrange(
            x,
            "B D (H h2) (W w2) -> B (D h2 w2) H W",
            h2=self.merge_size,
            w2=self.merge_size,
        )
        return merged_x


class DynamicCAbstractor(nn.Module):
    """Dynamic C-Abstractor based on RegNet blocks."""

    def __init__(
        self,
        config: Qwen2VLVisionConfig,
        num_input_tokens: int,
    ) -> None:
        super().__init__()
        assert hasattr(config, "merge_size"), "merge_size must be provided."
        self.config = config
        self.merge_size = config.merge_size
        self.pos_emb_size = config.pos_emb_size
        if num_input_tokens == -1:
            num_input_tokens = config.pos_emb_size
        self.num_input_tokens = num_input_tokens
        self.eos_tokens = build_eos_tokens(config, config.output_hidden_size)
        self.pos_emb = build_pos_embeds(
            config, num_input_tokens, config.encoder_hidden_size
        )
        self.prenorm = build_prenorm(config)
        self.build_net()

    def _load_from_state_dict(self, state_dict, *args, **kwargs) -> None:
        if not state_dict:
            return

        if self.pos_emb is not None:
            key_re = re.compile(r"[\w,.]*abstractor[\w,.]*pos_emb")
            pos_emb_key = None
            for key in state_dict:
                if key_re.match(key):
                    pos_emb_key = key
                    break

            assert pos_emb_key is not None
            # update old ckpt compatible with current code
            pos_emb = state_dict[pos_emb_key]
            if pos_emb.size(1) == self.pos_emb.size(1) + 1:
                # remove obsolete first pos emb (for cls token originally)
                state_dict[pos_emb_key] = pos_emb[:, 1:]

        super()._load_from_state_dict(state_dict, *args, **kwargs)

    def build_net(self) -> None:
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = PatchMerge(merge_size=self.merge_size)
        s2 = RegBlock(
            depth,
            self.merge_size**2 * hidden_size,
            hidden_size,
        )

        if depth:
            self.net = nn.ModuleList([s1, sampler, s2])
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)

    def forward(
        self,
        flattened_visual_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
        **unused_kwargs: object,
    ) -> BaseModelOutput:
        """Apply the dynamic abstractor over flattened visual embeddings."""
        n_token_loc = torch.prod(grid_thw, dim=1)
        split_visual_embeds = torch.split(flattened_visual_embeds, n_token_loc.tolist())

        flattened_visual_embeds = []
        for _visual_embeds, _grid_thw in zip(split_visual_embeds, grid_thw):
            T, H, W = _grid_thw
            assert T == 1, "T must be 1. Video is not supported yet."
            reshaped_visual_embeds = rearrange(
                _visual_embeds, "(t h w) d -> 1 t h w d", t=T, h=H, w=W
            )
            # remove temporal dim
            reshaped_visual_embeds = reshaped_visual_embeds[:, 0]

            if self.prenorm is not None:
                reshaped_visual_embeds = self.prenorm(reshaped_visual_embeds)

            if self.pos_emb is not None:
                # interpolate pos emb and add to visual embeds
                _local_pos_emb = resample_abs_pos_embed(
                    posemb=self.pos_emb,
                    old_size=tuple([int(self.pos_emb_size**0.5)] * 2),
                    new_size=(H, W),
                    num_prefix_tokens=0,
                )
                _local_pos_emb = rearrange(
                    _local_pos_emb,
                    "1 (h w) d -> 1 h w d",
                    h=H,
                    w=W,
                )
                reshaped_visual_embeds = reshaped_visual_embeds + _local_pos_emb

            reshaped_visual_embeds = self._forward(
                reshaped_visual_embeds,
                input_size=(H, W),
            )
            flattened_visual_embeds.append(reshaped_visual_embeds)
        reshaped_visual_embeds = torch.cat(flattened_visual_embeds, dim=0)
        return BaseModelOutput(last_hidden_state=reshaped_visual_embeds)

    def _forward(
        self,
        x: torch.Tensor,
        input_size: tuple[int, int],
    ) -> torch.Tensor:
        h, w = input_size
        x = rearrange(x, "1 h w d -> 1 d h w", h=h, w=w)
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = rearrange(x, "1 d h w -> (h w) d")
        x = self.readout(x)
        return x


class CustomQwen2VLVE(Qwen2VisionTransformer):
    """Thin wrapper around the Qwen2-VL used as a vision encoder.

    This mirrors the original HF-based vision encoder used in Kanana-V, but
    reuses vLLM's optimized `Qwen2VisionTransformer` building blocks.
    """

    config_class = Qwen2VLVisionConfig

    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__(
            vision_config=config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix="",
            use_data_parallel=False,
            attn_backend_override=None,
        )
        self.gradient_checkpointing = False

        # Kanana-V uses its own projector/abstractor instead of the Qwen2
        # built-in patch merger, so we drop the merger module to keep the
        # parameter set compatible with the original checkpoint.
        if hasattr(self, "merger"):
            del self.merger

    @classmethod
    def _from_config(cls, config: Qwen2VLVisionConfig) -> "CustomQwen2VLVE":
        """Drop-in replacement for the HF `_from_config` constructor."""
        return cls(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        """Run the vision transformer and optionally return intermediate states.

        Unlike the base `Qwen2VisionTransformer`, this wrapper exposes the
        pre-merger patch-level representations and a HF-style `BaseModelOutput`
        so that the existing projector / abstractor code can be reused.
        """
        assert return_dict, "Only return_dict=True is supported."

        # Patchify
        x = pixel_values.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)  # (num_patches, embed_dim)

        # Prepare grid and rotary embeddings – mirror base implementation.
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw_np = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw_np = grid_thw.cpu().numpy()

        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # Compute cu_seqlens in numpy then move to device, same as base model.
        cu_seqlens = np.repeat(
            grid_thw_np[:, 1] * grid_thw_np[:, 2],
            grid_thw_np[:, 0],
        ).cumsum(axis=0, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        cu_seqlens = torch.from_numpy(cu_seqlens).to(
            self.device,
            non_blocking=True,
        )

        # Shape to (S, B, D) with batch dimension 1 as expected by the blocks.
        x = x.unsqueeze(1)

        # Pre-compute seqlens for attention backend.
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        encoder_states = () if output_hidden_states else None

        for blk in self.blocks:
            if output_hidden_states:
                # Store patch-level states (S, D).
                encoder_states = encoder_states + (x.squeeze(1),)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    blk.__call__,
                    x,
                    cu_seqlens,
                    rotary_pos_emb_cos,
                    rotary_pos_emb_sin,
                    max_seqlen,
                )
            else:
                layer_outputs = blk(
                    x,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb_cos=rotary_pos_emb_cos,
                    rotary_pos_emb_sin=rotary_pos_emb_sin,
                    max_seqlen=max_seqlen,
                )
            x = layer_outputs

        # Final hidden state at patch level (S, D).
        hidden_states = x.squeeze(1)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
        )

    def get_num_tokens(self) -> int:
        # Not used in the current Kanana-V pipeline, kept for API compatibility.
        return -1


class KananaVProcessingInfo(BaseProcessingInfo):
    """Processing info for Kanana-V, declaring supported modalities and limits."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=9999999,
            image_height=9999999,
            num_frames=1,
        )
        return max_image_size

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
    ) -> tuple[ImageSize, int]:
        image_processor = self.ctx.get_hf_processor().image_processor

        import sys

        module = sys.modules[type(image_processor).__module__]
        smart_resize = module.smart_resize

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        target_width, target_height = self.get_image_size_with_most_features()
        num_vision_tokens = self._get_vision_info(
            image_width=target_width,
            image_height=target_height,
            num_frames=1,
        )[1]
        return {"image": num_vision_tokens}


class KananaVDummyInputsBuilder(BaseDummyInputsBuilder[KananaVProcessingInfo]):
    """Utility for building dummy inputs for profiling / benchmarking."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        image_token = "<image>\n"
        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        return {
            "image": self._get_dummy_images(
                width=4096, height=2160, num_images=num_images
            ),
        }


class KananaVMultiModalProcessor(BaseMultiModalProcessor[KananaVProcessingInfo]):
    """vLLM multimodal processor for Kanana-V (text + image)."""

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: MultiModalDataItems,
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Run the underlying HF processor on text and image data."""
        # Text-only input is handled as a special case here.
        if not mm_data or not mm_data.get("images", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Images
        image_inputs = mm_data.get("images", [])
        pixel_sizes = []
        if image_inputs is not None:
            if not isinstance(image_inputs[0], Image.Image):
                image_inputs = [Image.fromarray(image) for image in image_inputs]

            image_processor = self.info.get_hf_processor().image_processor
            processor_output = [image_processor(image) for image in image_inputs]
            pixel_values = [o["pixel_values"] for o in processor_output]
            image_meta = [o["image_meta"] for o in processor_output]
            # list of dict -> dict of list
            image_meta = {k: [d[k] for d in image_meta] for k in image_meta[0]}

            for pixel_value in pixel_values:
                pixel_sizes.append(pixel_value.shape[0])
            # flattened pixel_values for single example (already includes batch dim)
            pixel_values = torch.concat(pixel_values, dim=0)

        # Text with image metadata
        text_tokens = self.info.get_tokenizer().encode_prompt(
            prompt,
            max_length=None,
            image_meta=image_meta if image_inputs is not None else None,
        )
        input_ids = text_tokens["input_ids"]
        media_token_id = self.info.get_hf_config().text_config.eos_token_id + 1
        # Replace placeholder token ids (-1) with media_token_id.
        input_ids = torch.where(input_ids == -1, media_token_id, input_ids)

        combined_outputs = dict(
            # Add batch dimension to input_ids.
            input_ids=input_ids.unsqueeze(0),
            # pixel_values already contain the batch dimension.
            pixel_values=pixel_values if image_inputs is not None else None,
            vision_grid_thw=torch.tensor(image_meta["vision_grid_thw"])
            if image_inputs is not None
            else None,
            image_token_thw=torch.tensor(image_meta["image_token_thw"])
            if image_inputs is not None
            else None,
            pixel_sizes=torch.tensor(pixel_sizes) if image_inputs is not None else None,
            tensor_type="pt",
            tokenization_kwargs=tok_kwargs,
        )
        return BatchFeature(combined_outputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        media_token_id = self.info.get_hf_config().text_config.eos_token_id + 1
        separator_token_id = 198

        def get_replacement(idx: int) -> Sequence[int]:
            out_item = out_mm_kwargs["image"][idx]
            image_token_thw = out_item["image_token_thw"].data.cpu().numpy()

            media_tokens = self.info.get_tokenizer().repeat_image_tokens(
                image_token_thw,
                with_row_separator=True,
                add_global_local_separator=True,
            )

            # Sanitize media tokens.
            media_tokens = [
                media_token_id if token == -1 or token == separator_token_id else token
                for token in media_tokens
            ]
            return media_tokens

        return [
            PromptReplacement(
                modality="image", target="<image>", replacement=get_replacement
            ),
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        pixel_sizes = hf_inputs.get("pixel_sizes", torch.empty(0))

        mm_fields_config = dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", pixel_sizes),
            vision_grid_thw=MultiModalFieldConfig.batched("image"),
            image_token_thw=MultiModalFieldConfig.batched("image"),
        )
        return mm_fields_config


@MULTIMODAL_REGISTRY.register_processor(
    KananaVMultiModalProcessor,
    info=KananaVProcessingInfo,
    dummy_inputs=KananaVDummyInputsBuilder,
)
class KananaVForConditionalGeneration(nn.Module, SupportsMultiModal):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        logger.info("Build vision model ...")
        self.vision_model = CustomQwen2VLVE._from_config(config.vision_config)

        logger.info("Build projector ...")
        self.abstractor = DynamicCAbstractor(
            config.projector_config, num_input_tokens=self.vision_model.get_num_tokens()
        )

        logger.info("Build LM ...")
        self.text_config = config.text_config
        language_model_config = vllm_config.with_hf_config(hf_config=self.text_config)
        language_model_config.quant_config = quant_config
        if self.text_config.architectures[0] == "LlamaForCausalLM":
            self.language_model = LlamaForCausalLM(
                vllm_config=language_model_config, prefix="model"
            )
        else:
            raise NotImplementedError("Not supported language model architecture")

        self.media_token_id = self.text_config.eos_token_id + 1
        self.separator_token_id = 198

    def forward_vision(
        self,
        pixel_values: torch.Tensor,
        image_metas: dict | None = None,
    ) -> torch.Tensor:
        """Run the vision backbone and return hidden states at the target layer."""
        vision_model_args = {
            "pixel_values": pixel_values,
            "return_dict": True,
            "output_hidden_states": True,
            "grid_thw": image_metas["vision_grid_thw"],
        }
        v_outputs = self.vision_model(**vision_model_args)
        layer_index = self.config.projector_config.feature_layer_index
        visual_features = self._get_visual_feature_at(
            v_outputs.hidden_states, layer_index
        )
        return visual_features

    def forward_projector(
        self,
        visual_features: torch.Tensor,
        image_metas: dict | None = None,
    ) -> torch.Tensor:
        """Run the projector / abstractor over vision features."""
        visual_embeds = self.abstractor(
            visual_features,
            grid_thw=image_metas["vision_grid_thw"],
        )["last_hidden_state"]
        return visual_embeds

    def forward_and_project_vision(
        self,
        pixel_values: torch.Tensor,
        image_metas: dict | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper for vision backbone + projector."""
        assert pixel_values is not None
        visual_features = self.forward_vision(pixel_values, image_metas=image_metas)
        visual_embeds = self.forward_projector(visual_features, image_metas=image_metas)
        return visual_embeds

    def _get_visual_feature_at(
        self,
        v_output: Sequence[torch.Tensor],
        layer_index: int | Sequence[int],
    ) -> torch.Tensor:
        if isinstance(layer_index, list):
            visual_features = torch.stack(v_output, dim=1)[
                :, layer_index
            ]  # [B, n_scales, L, dim]
        else:
            visual_features = v_output[layer_index]  # [B, L, dim]
        return visual_features

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        """Compute multimodal embeddings for image inputs.

        This expands media tokens into grids of visual tokens with row and
        global/local separator tokens, then replaces the media-token
        embeddings with projected visual features.
        """
        assert kwargs["pixel_values"] is not None
        pixel_values = kwargs["pixel_values"]
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.ndim == 2:
                kwargs["pixel_values"] = pixel_values
            if pixel_values.ndim != 3:
                raise ValueError(
                    f"pixel_values should be 2D or batched 3D tensor. "
                    f"Got ndim: {pixel_values.ndim} "
                    f"(shape={pixel_values.shape})"
                )
            kwargs["pixel_values"] = pixel_values.flatten(0, 1)
        else:
            kwargs["pixel_values"] = torch.concat(pixel_values)
        if kwargs["vision_grid_thw"].ndim == 3:
            kwargs["vision_grid_thw"] = kwargs["vision_grid_thw"].flatten(0, 1)
        visual_embeds = self.forward_and_project_vision(kwargs["pixel_values"], kwargs)

        merge_size = self.abstractor.merge_size
        batch_size = kwargs["vision_grid_thw"].size(0)
        multi_modal_embeddings = ()
        sample_index = 0
        sep_embed = self._embed_text_input_ids(
            torch.tensor([self.separator_token_id]).to(visual_embeds.device),
            self.language_model.embed_input_ids,
            is_multimodal=None,
            handle_oov_mm_token=False,
        )[0]
        sep_embed = torch.unsqueeze(torch.unsqueeze(sep_embed, 0), 1)
        for i in range(batch_size):
            t, h, w = (
                kwargs["vision_grid_thw"][i][0],
                kwargs["vision_grid_thw"][i][1] // merge_size,
                kwargs["vision_grid_thw"][i][2] // merge_size,
            )
            visual_embed = visual_embeds[sample_index : sample_index + t * h * w].view(
                t, h, w, -1
            )
            sep_embeds = sep_embed.expand(t, h, 1, -1)
            multi_modal_embeddings += (
                torch.cat([sep_embeds, visual_embed], dim=2).view(t * h * (w + 1), -1),
            )
            sample_index += t * h * w
        return multi_modal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs,
    ):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is None")

        outputs = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return outputs

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
