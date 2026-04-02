# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Cheers (UMM) model compatible with HuggingFace weights.

Cheers is a unified multimodal model for image understanding and generation.
For vLLM, we focus on the image understanding (vision-to-text) capabilities.
The image generation part (gen_projector, hi_gate, etc.) is not supported,
but the VAE encoder + decoder projector are required for image understanding.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processors.cheers import CheersProcessor
from vllm.utils.tensor_schema import TensorSchema

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .siglip import SiglipVisionModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


# ── VAE components (needed for image understanding pipeline) ────────


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class _AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        return x + self.proj_out(h_)


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _swish(self.norm1(x))
        h = self.conv1(h)
        h = _swish(self.norm2(h))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class _Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


_VAE_ENCODER_DEFAULTS = {
    "in_channels": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "z_channels": 32,
}
_VAE_DECODER_DEFAULTS = {
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "z_channels": 32,
}


def _cfg(config, key, defaults=None):
    """Access config attribute whether it's a dict or namespace object."""
    if isinstance(config, dict):
        if key in config:
            return config[key]
        if defaults and key in defaults:
            return defaults[key]
        raise KeyError(f"Key '{key}' not found in config dict: {list(config.keys())}")
    return getattr(config, key)


class CheersVAEEncoder(nn.Module):
    """VAE encoder from the Cheers/UMM model."""

    def __init__(self, config):
        super().__init__()
        d = _VAE_ENCODER_DEFAULTS
        ch = _cfg(config, "ch", d)
        ch_mult = _cfg(config, "ch_mult", d)
        num_res_blocks = _cfg(config, "num_res_blocks", d)
        z_channels = _cfg(config, "z_channels", d)
        in_channels = _cfg(config, "in_channels", d)
        num_resolutions = len(ch_mult)

        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != num_resolutions - 1:
                down.downsample = _Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, 1, 1)
        self._num_resolutions = num_resolutions
        self._num_res_blocks = num_res_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self._num_resolutions):
            for i_block in range(self._num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = _swish(self.norm_out(h))
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class CheersVAEDecoder(nn.Module):
    """VAE decoder (used inside VAEDecoderProjector)."""

    def __init__(self, config):
        super().__init__()
        d = _VAE_DECODER_DEFAULTS
        ch = _cfg(config, "ch", d)
        ch_mult = _cfg(config, "ch_mult", d)
        num_res_blocks = _cfg(config, "num_res_blocks", d)
        z_channels = _cfg(config, "z_channels", d)
        out_ch = _cfg(config, "out_ch", d)
        num_resolutions = len(ch_mult)

        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        block_in = ch * ch_mult[num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)
        self._num_resolutions = num_resolutions
        self._num_res_blocks = num_res_blocks

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.to(upscale_dtype)
        for i_level in reversed(range(self._num_resolutions)):
            for i_block in range(self._num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = _swish(self.norm_out(h))
        return self.conv_out(h)


class CheersVAEModel(nn.Module):
    """VAE model with encoder only (for image understanding)."""

    def __init__(self, config):
        super().__init__()
        enc_cfg = _cfg(config, "vae_encoder_config")
        self.encoder = CheersVAEEncoder(enc_cfg)
        self.ps = [2, 2]
        z_ch = _cfg(enc_cfg, "z_channels", _VAE_ENCODER_DEFAULTS)
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * z_ch,
            eps=1e-4,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.bn.eval()
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(
            mean,
            "... c (i pi) (j pj) -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.bn(z)


class CheersVAEDecoderProjector(nn.Module):
    """VAE decoder projector that converts latent back to pixel-like space."""

    def __init__(self, config):
        super().__init__()
        dec_cfg = _cfg(config, "vae_decoder_config")
        enc_cfg = _cfg(config, "vae_encoder_config")
        self.decoder = CheersVAEDecoder(dec_cfg)
        self.ps = [2, 2]
        z_ch = _cfg(enc_cfg, "z_channels", _VAE_ENCODER_DEFAULTS)
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * z_ch,
            eps=1e-4,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + 1e-4)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        z = z * s + m
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)


class CheersImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each image
        - w: Width of each image
    """

    type: Literal["pixel_values"]
    pixel_values: torch.Tensor  # Shape: (bn, 3, h, w)


CheersImageInputs: TypeAlias = CheersImagePixelInputs


class CheersUndProjector(nn.Module):
    """Understanding projector that maps vision features to LLM dimension
    with 2x2 spatial compression (4x token reduction)."""

    def __init__(
        self,
        image_embed_dim: int,
        text_embed_dim: int,
        compression_factor: tuple[int, int] = (2, 2),
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim
        self.compression_factor = compression_factor
        self.layernorm = nn.LayerNorm(image_embed_dim)
        hidden_size = image_embed_dim * (compression_factor[0] * compression_factor[1])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, text_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        height = width = int(x.size(1) ** 0.5)
        x = x.permute(0, 2, 1).unflatten(-1, (height, width))
        batch_size, dim, height, width = x.shape
        unfolded = x.unfold(
            2, self.compression_factor[0], self.compression_factor[0]
        ).unfold(3, self.compression_factor[1], self.compression_factor[1])
        unfolded = unfolded.contiguous().view(
            batch_size,
            dim,
            -1,
            self.compression_factor[0] * self.compression_factor[1],
        )
        unfolded = (
            unfolded.permute(0, 2, 3, 1)
            .contiguous()
            .view(
                batch_size,
                -1,
                dim * self.compression_factor[0] * self.compression_factor[1],
            )
        )
        return self.mlp(unfolded)


class CheersProcessingInfo(BaseProcessingInfo):
    """Processing information for Cheers model."""

    def get_hf_processor(self, **kwargs: object) -> CheersProcessor:
        from vllm.transformers_utils.processor import cached_get_image_processor

        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        tokenizer = self.get_tokenizer()

        return CheersProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        vit_config = hf_config.vision_representation_config
        patch_size = vit_config.patch_size
        image_size = vit_config.image_size
        num_patches = (image_size // patch_size) ** 2
        # After 2x2 compression, tokens reduce by 4x
        num_tokens = num_patches // 4
        return {"image": num_tokens}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vit_config = hf_config.vision_representation_config
        patch_size = vit_config.patch_size
        image_size = vit_config.image_size
        num_patches = (image_size // patch_size) ** 2
        return num_patches // 4


class CheersDummyInputsBuilder(BaseDummyInputsBuilder[CheersProcessingInfo]):
    """Build dummy inputs for Cheers model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<|image_pad|>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        hf_config = self.info.get_hf_config()
        vit_config = hf_config.vision_representation_config
        image_size = vit_config.image_size
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class CheersMultiModalProcessor(BaseMultiModalProcessor[CheersProcessingInfo]):
    """Multimodal processor for Cheers model."""

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        hf_config = self.info.get_hf_config()
        vit_config = hf_config.vision_representation_config
        patch_size = vit_config.patch_size
        image_size = vit_config.image_size

        tokenizer = self.info.get_tokenizer()
        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        if image_token_id is None:
            raise ValueError(
                "Image token '<|image_pad|>' not found in tokenizer vocabulary"
            )

        def get_replacement_cheers(item_idx: int):
            num_patches = (image_size // patch_size) ** 2
            num_tokens = num_patches // 4
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_cheers,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
        }


@MULTIMODAL_REGISTRY.register_processor(
    CheersMultiModalProcessor,
    info=CheersProcessingInfo,
    dummy_inputs=CheersDummyInputsBuilder,
)
class CheersForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP
):
    """
    Cheers: A unified multimodal model for image understanding and generation.

    For vLLM, we focus on the image understanding (vision-to-text) capabilities.
    The image generation part is not supported in vLLM.
    """

    requires_raw_input_tokens = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_representation.": "vision_representation.vision_model.",
            "model.und_projector.": "und_projector.",
            "model.vae_model.": "vae_model.",
            "model.vae_decoder_projector.": "vae_decoder_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image_pad|>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        if type(config).__name__ not in ("CheersConfig", "UMMConfig"):
            raise ValueError(
                f"Expected CheersConfig or UMMConfig, got {type(config).__name__}."
            )

        self.config = config
        self.multimodal_config = multimodal_config

        # The Cheers model's custom Qwen2Config defaults rope_theta to
        # 1_000_000, but this isn't stored in the JSON.  vLLM's standard
        # Qwen2Config defaults to 10_000, causing a 100× mismatch.
        # We must patch BOTH the attribute AND rope_parameters (which
        # patch_rope_parameters may have already populated from the wrong
        # default before __init__ runs).
        _CHEERS_ROPE_THETA = 1_000_000.0
        tc = config.text_config
        old_theta = getattr(tc, "rope_theta", None)
        if old_theta != _CHEERS_ROPE_THETA:
            logger.info(
                "Overriding text_config.rope_theta from %s to %s",
                old_theta,
                _CHEERS_ROPE_THETA,
            )
            tc.rope_theta = _CHEERS_ROPE_THETA
        rp = getattr(tc, "rope_parameters", None)
        if rp is not None and rp.get("rope_theta") != _CHEERS_ROPE_THETA:
            logger.info(
                "Overriding rope_parameters.rope_theta from %s to %s",
                rp.get("rope_theta"),
                _CHEERS_ROPE_THETA,
            )
            rp["rope_theta"] = _CHEERS_ROPE_THETA

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        vit_config = config.vision_representation_config

        with self._mark_tower_model(vllm_config, "image"):
            self.vae_model = CheersVAEModel(config)
            self.vae_decoder_projector = CheersVAEDecoderProjector(config)

            self.vision_representation = SiglipVisionModel(
                config=vit_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_representation"),
            )

            vit_hidden_size = vit_config.hidden_size
            llm_hidden_size = config.text_config.hidden_size

            self.und_projector = CheersUndProjector(
                image_embed_dim=vit_hidden_size,
                text_embed_dim=llm_hidden_size,
                compression_factor=(2, 2),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "und_projector"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> CheersImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        return CheersImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
        )

    def _process_image_input(
        self, image_input: CheersImageInputs
    ) -> tuple[torch.Tensor, ...]:
        """Process image inputs through VAE → SigLIP → projector pipeline.

        HF native path: pixel_values → VAE.encode(t=1.0) → vae_decoder_projector
                         → SigLIP → und_projector → text-space embeddings
        """
        pixel_values = image_input["pixel_values"]

        if pixel_values.ndim == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(
                batch_size * num_images, channels, height, width
            )

        with torch.no_grad():
            vae_dtype = next(self.vae_model.parameters()).dtype
            image_latent = self.vae_model.encode(pixel_values.to(dtype=vae_dtype))
            image_pixel_hat = self.vae_decoder_projector(image_latent)

        vision_features = self.vision_representation(image_pixel_hat)
        vision_embeds = self.und_projector(vision_features)

        return tuple(vision_embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

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

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, keeping VAE encoder/decoder projector for understanding."""
        skip_prefixes = [
            "model.time_embed.",
            "model.gen_projector.",
            "model.hi_gate.",
            "model.hi_projector.",
            "model.vae_model.decoder.",
        ]
        skip_keywords = [
            "text_loss_fc",
        ]

        filtered_weights = []
        for name, tensor in weights:
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            if any(kw in name for kw in skip_keywords):
                continue
            filtered_weights.append((name, tensor))

        loader = AutoWeightsLoader(self)
        return loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)
