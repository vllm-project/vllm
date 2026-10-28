# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x Omni: NaViT vision and waveform audio on the MuseSpark1x stack."""

import itertools
import math
from copy import copy
from typing import TYPE_CHECKING, cast

import regex as re
import torch
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.model_executor.models.vision import (
    get_load_balance_assignment,
    is_vit_use_data_parallel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .audio import AutoEncoder
from .layers import MuseSpark1xRMSNorm
from .processor import (
    MuseSpark1xOmniDummyInputsBuilder,
    MuseSpark1xOmniMultiModalProcessor,
    MuseSpark1xOmniProcessingInfo,
)
from .vision import NaViT, get_navit_patch_count, get_navit_video_patch_count

logger = init_logger(__name__)


OMNI_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_regex={
        re.compile(pattern): replacement
        for pattern, replacement in {
            r"^model\.language_model\.": "model.",
            r"^(model\.layers\.\d+)\.input_layernorm\.": (
                r"\1.self_attn.pre_qkv_proj_norm."
            ),
            r"^(model\.layers\.\d+)\.post_attention_residual_gate\.": (r"\1.add_attn."),
            r"^(model\.layers\.\d+)\.post_feedforward_residual_gate\.": (
                r"\1.add_ffn."
            ),
            r"^(model\.layers\.\d+)\.pre_feedforward_layernorm\.": (
                r"\1.pre_ffn_norm."
            ),
            r"^(model\.layers\.\d+)\.post_feedforward_layernorm\.": (
                r"\1.post_ffn_norm."
            ),
            r"^(model\.layers\.\d+)\.mlp\.experts\.post_expert_norm\.": (
                r"\1.feed_forward.post_expert_norm."
            ),
            r"^(model\.layers\.\d+)\.mlp\.gate\.e_score_correction_bias$": (
                r"\1.feed_forward.expert_bias"
            ),
            r"^(model\.layers\.\d+)\.mlp\.gate\.": r"\1.feed_forward.router.",
            r"^(model\.layers\.\d+)\.mlp\.pre_expert_proj\.": (
                r"\1.feed_forward.pooler."
            ),
            r"^(model\.layers\.\d+)\.mlp\.post_expert_proj\.": (
                r"\1.feed_forward.unpooler."
            ),
            r"^(model\.layers\.\d+)\.mlp\.": r"\1.feed_forward.",
            r"\.experts\.gate_up_proj_input_scale$": (".experts.w13_input_scale"),
            r"\.experts\.gate_up_proj_weight_scale_2$": (".experts.w13_weight_scale_2"),
            r"\.experts\.gate_up_proj_weight_scale$": (".experts.w13_weight_scale"),
            r"\.experts\.down_proj_input_scale$": ".experts.w2_input_scale",
            r"\.experts\.down_proj_weight_scale_2$": (".experts.w2_weight_scale_2"),
            r"\.experts\.down_proj_weight_scale$": ".experts.w2_weight_scale",
            r"\.experts\.gate_up_proj$": ".experts.w13_weight",
            r"\.experts\.down_proj$": ".experts.w2_weight",
            r"^model\.vision_adapter\.fc1\.": "vision_model.adapter.c_fc.",
            r"^model\.vision_adapter\.fc2\.": "vision_model.adapter.c_proj.",
            r"^model\.vision_tower\.patch_embedder\.patch_embedding\.": (
                "vision_model.conv1_linear."
            ),
            (
                r"^model\.vision_tower\.patch_embedder\."
                r"position_embedding_table\.weight$"
            ): "vision_model.positional_embedding_vlm",
            r"^model\.vision_tower\.layers\.": "vision_model.encoder.layers.",
            r"^model\.vision_tower\.": "vision_model.",
            r"^model\.vision_projection\.": "vision_projection.",
            r"^(vision_model\.encoder\.layers\.\d+)\.attn\.proj\.": (
                r"\1.attn.out_proj."
            ),
            r"^(vision_model\.encoder\.layers\.\d+)\.norm1\.": r"\1.ln_1.",
            r"^(vision_model\.encoder\.layers\.\d+)\.norm2\.": r"\1.ln_2.",
            r"^(vision_model\.encoder\.layers\.\d+)\.mlp\.fc1\.": r"\1.mlp.c_fc.",
            r"^(vision_model\.encoder\.layers\.\d+)\.mlp\.fc2\.": r"\1.mlp.c_proj.",
            **{
                rf"^model\.audio_tower\.layers\.(\d+)\.conv_blocks\.{block}\.": (
                    rf"audio_encoder.stages.\1.{block + 1}."
                )
                for block in range(2)
            },
            r"^(audio_encoder\.stages\.\d+\.\d+)\.conv\.": r"\1.conv.conv.",
            r"^(audio_encoder\.stages\.\d+\.\d+)\.mlp\.fc1\.": r"\1.linear1.",
            r"^(audio_encoder\.stages\.\d+\.\d+)\.mlp\.fc2\.": r"\1.linear2.",
            r"^model\.audio_tower\.layers\.(\d+)\.patch_embedder\.norm\.": (
                r"audio_encoder.stages.\1.0.norm."
            ),
            r"^model\.audio_tower\.layers\.(\d+)\.patch_embedder\.proj\.": (
                r"audio_encoder.stages.\1.0.linear."
            ),
            r"^model\.audio_tower\.post_proj_norm\.": "audio_encoder.dual_norm.",
            r"^model\.audio_tower\.proj\.": "audio_encoder.o_proj.",
            r"^model\.audio_tower\.norm\.": "audio_encoder.norm.",
        }.items()
    },
    orig_to_new_stacked={
        ".self_attn.q_proj.": (".self_attn.qkv_proj.", "q"),
        ".self_attn.k_proj.": (".self_attn.qkv_proj.", "k"),
        ".self_attn.v_proj.": (".self_attn.qkv_proj.", "v"),
        ".self_attn.gate_proj.": (".self_attn.qkv_proj.", "o_gate"),
        ".attn.q_proj.": (".attn.qkv_proj.", "q"),
        ".attn.k_proj.": (".attn.qkv_proj.", "k"),
        ".attn.v_proj.": (".attn.qkv_proj.", "v"),
    },
)


def register_omni_processor(cls: type) -> type:
    """Register the Omni multimodal processor for a vendor's Omni class."""
    return MULTIMODAL_REGISTRY.register_processor(
        MuseSpark1xOmniMultiModalProcessor,
        info=MuseSpark1xOmniProcessingInfo,
        dummy_inputs=MuseSpark1xOmniDummyInputsBuilder,
    )(cls)


class MuseSpark1xOmniMixin:
    """The Omni towers, their embedding merge and their weight mapping.

    Not a ``Protocol``: it precedes the text stack in the bases so ``__init__``
    runs after the stack is built, and a Protocol would swallow that chain.
    """

    supports_encoder_tp_data = True
    # Provided by the vendor text stack.
    model: nn.Module

    if TYPE_CHECKING:

        def _mark_tower_model(
            self,
            vllm_config: VllmConfig,
            modalities: set[str] | str,
            *,
            targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
        ): ...

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|patch|>"
        for kind in ("video", "audio"):
            if modality.startswith(kind):
                return f"<|{kind}|>"
        raise ValueError(f"Unsupported Muse Spark modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        hf_config = vllm_config.model_config.hf_config
        text_vllm_config = copy(vllm_config)
        text_vllm_config.model_config = copy(vllm_config.model_config)
        text_vllm_config.model_config.hf_config = hf_config.text_config
        super().__init__(vllm_config=text_vllm_config, prefix=prefix)  # type: ignore[call-arg]

        vision_config = hf_config.vision_config
        audio_config = hf_config.audio_config
        self.vllm_config = vllm_config
        adapter_output_dim = hf_config.projector_hidden_size
        llm_hidden_size = hf_config.text_config.hidden_size

        with set_current_vllm_config(vllm_config, prefix=prefix):
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                with set_model_tag("NaViTEncoder", is_encoder=True):
                    self.vision_model = NaViT(
                        vision_config,
                        projector_hidden_size=hf_config.projector_hidden_size,
                        prefix=maybe_prefix(prefix, "vision_model"),
                    )
                self.vision_projection = RowParallelLinear(
                    input_size=adapter_output_dim,
                    output_size=llm_hidden_size,
                    bias=False,
                    input_is_parallel=False,
                    quant_config=None,
                    prefix=maybe_prefix(prefix, "vision_projection"),
                )
                self.perception_emb_norm = MuseSpark1xRMSNorm(
                    llm_hidden_size, eps=1e-5, has_weight=False
                )

            with self._mark_tower_model(vllm_config, "audio"):
                with set_model_tag("AutoEncoder", is_encoder=True):
                    self.audio_encoder = AutoEncoder(
                        latent_channels=audio_config.out_hidden_size,
                        patch_size=math.prod(audio_config.patch_sizes),
                        patchify_sizes=tuple(audio_config.patch_sizes),
                        d_models=tuple(audio_config.hidden_sizes),
                        n_layers=tuple(audio_config.depths),
                        kernel_size=audio_config.conv_kernel_size,
                        mlp_ratio=audio_config.mlp_ratio,
                        norm_eps=audio_config.rms_norm_eps,
                        prefix=maybe_prefix(prefix, "audio_encoder"),
                    )
                if vllm_config.compilation_config.compile_mm_encoder:
                    self.audio_encoder._encode_prefill = (  # type: ignore[method-assign]
                        torch.compile(
                            self.audio_encoder._encode_prefill,
                            fullgraph=True,
                            dynamic=True,
                        )
                    )

        self._patch_height = vision_config.patch_size
        self._patch_width = vision_config.patch_size
        self._patch_temporal = vision_config.patch_temporal
        self._downsample_factor = vision_config.merge_size

        self._use_vision_dp = is_vit_use_data_parallel()
        self._adapter_output_dim = adapter_output_dim

        # FlashInfer's cute-dsl GEMM passes operands over DLPack, which refuses
        # grad-carrying tensors such as the towers' biases.
        cast(nn.Module, self).requires_grad_(False)

    def get_language_model(self) -> torch.nn.Module:
        return self

    def _process_image_input(
        self,
        images: torch.Tensor | list[torch.Tensor],
        target_hw: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Convert image inputs to a list of individual image tensors."""
        if isinstance(images, list):
            result: list[torch.Tensor] = []
            for img in images:
                if img.dim() in (3, 4):  # [C, H, W] or [N, C, H, W]
                    result.append(img)
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
            return result
        if images.dim() == 1:
            # The Rust frontend ships one flat channel-planar buffer per batch.
            if target_hw is None:
                raise ValueError(
                    "Flat image pixels require image_target_hw for reshaping"
                )
            hw = target_hw.tolist()
            counts = [3 * h * w for h, w in hw]
            if sum(counts) != images.shape[0]:
                raise ValueError(
                    f"Flat image pixels {images.shape[0]} do not match "
                    f"image_target_hw {hw}"
                )
            return [
                item.view(3, h, w) for item, (h, w) in zip(images.split(counts), hw)
            ]
        if images.dim() == 4:  # [B, C, H, W]
            return [images[i] for i in range(images.shape[0])]
        if images.dim() == 5:  # [B, N, C, H, W]
            return [images[i] for i in range(images.shape[0])]
        raise ValueError(f"Unexpected batched image shape: {images.shape}")

    def _process_video_input(
        self,
        videos: torch.Tensor | list[torch.Tensor],
        target_hw: torch.Tensor | None = None,
        frame_counts: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Convert video inputs to a list of individual video tensors."""
        if isinstance(videos, list):
            for v in videos:
                assert v.dim() == 4, (
                    f"Expected [T, C, H, W] video tensor, got {v.shape}"
                )
            return videos
        if videos.dim() == 1:
            # The Rust frontend ships one flat buffer per batch.
            if target_hw is None or frame_counts is None:
                raise ValueError(
                    "Flat video pixels require video_target_hw and "
                    "video_frame_counts for reshaping"
                )
            hw = target_hw.tolist()
            counts = [3 * t * h * w for t, (h, w) in zip(frame_counts.tolist(), hw)]
            if sum(counts) != videos.shape[0]:
                raise ValueError(
                    f"Flat video pixels {videos.shape[0]} do not match "
                    f"video_target_hw {hw} and frame counts "
                    f"{frame_counts.tolist()}"
                )
            return [
                item.view(t, 3, h, w)
                for item, t, (h, w) in zip(
                    videos.split(counts), frame_counts.tolist(), hw
                )
            ]
        assert videos.dim() == 5, (
            f"Expected [B, T, C, H, W] video tensor, got {videos.shape}"
        )
        return [videos[i] for i in range(videos.shape[0])]

    def _get_patch_count(self, item: torch.Tensor) -> int:
        """Number of output patches for an image or video tensor."""
        if item.ndim == 3:  # [C, H, W]
            return get_navit_patch_count(
                img_h=item.shape[1],
                img_w=item.shape[2],
                patch_h=self._patch_height,
                patch_w=self._patch_width,
                downsample_factor=self._downsample_factor,
            )
        if item.ndim == 4:  # [T, C, H, W] or [N, C, H, W]
            return get_navit_video_patch_count(
                video_t=item.shape[0],
                video_h=item.shape[2],
                video_w=item.shape[3],
                patch_t=self._patch_temporal,
                patch_h=self._patch_height,
                patch_w=self._patch_width,
                downsample_factor=self._downsample_factor,
            )
        raise ValueError(f"Unexpected vision item shape: {item.shape}")

    def _encode_vision(self, vision_items: list[torch.Tensor]) -> list[torch.Tensor]:
        """NaViT -> vision_projection -> perception_emb_norm, split per item."""
        if not vision_items:
            return []
        vision_params = next(self.vision_model.parameters(), None)
        if vision_params is not None:
            vision_items = [
                x.to(
                    device=vision_params.device,
                    dtype=vision_params.dtype,
                    non_blocking=True,
                )
                for x in vision_items
            ]

        if self._use_vision_dp:
            h_vision = self._run_dp_sharded_navit(vision_items)
        else:
            h_vision = self.vision_model(vision_items)

        h_vision, _ = self.vision_projection(h_vision)

        h_vision = self.perception_emb_norm(h_vision)

        patch_counts = [self._get_patch_count(item) for item in vision_items]
        assert h_vision.shape[0] == sum(patch_counts), (
            f"NaViT output {h_vision.shape[0]} patches != calculated "
            f"{sum(patch_counts)}. Per-item: {patch_counts}, "
            f"shapes: {[x.shape for x in vision_items]}"
        )
        return list(h_vision.split(patch_counts))

    def _encode_audio(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        sample_counts: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        audio_params = next(self.audio_encoder.parameters(), None)
        # reduce_data may stack same-length waveforms into [N, T].
        if isinstance(input_features, torch.Tensor):
            if input_features.dim() >= 2:
                input_features = [
                    input_features[i] for i in range(input_features.shape[0])
                ]
            elif sample_counts is not None and sample_counts.numel() > 1:
                # The Rust frontend ships one flat buffer per batch.
                counts = sample_counts.tolist()
                if sum(counts) != input_features.shape[0]:
                    raise ValueError(
                        f"Flat audio samples {input_features.shape[0]} do not "
                        f"match audio_sample_counts {counts}"
                    )
                input_features = list(input_features.split(counts))
            else:
                input_features = [input_features]
        results: list[torch.Tensor] = []
        for wav in input_features:
            if audio_params is not None:
                wav = wav.to(device=audio_params.device, dtype=audio_params.dtype)
            audio_embeds = self.audio_encoder([wav])
            if isinstance(audio_embeds, list) and len(audio_embeds) > 0:
                audio_embed = audio_embeds[0]
            else:
                audio_embed = audio_embeds
            # Audio takes tok_embeddings_norm, not perception_emb_norm.
            audio_embed = self.model.tok_embeddings_norm(audio_embed)
            results.append(audio_embed)
        return results

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Encode all modalities in the order they appear in the kwargs."""
        embeddings: list[torch.Tensor] = []
        for key, value in kwargs.items():
            if key == "images":
                embeddings.extend(
                    self._encode_vision(
                        self._process_image_input(
                            cast(torch.Tensor | list[torch.Tensor], value),
                            cast(
                                torch.Tensor | None,
                                kwargs.get("image_target_hw"),
                            ),
                        )
                    )
                )
            elif key == "videos":
                embeddings.extend(
                    self._encode_vision(
                        self._process_video_input(
                            cast(torch.Tensor | list[torch.Tensor], value),
                            cast(
                                torch.Tensor | None,
                                kwargs.get("video_target_hw"),
                            ),
                            cast(
                                torch.Tensor | None,
                                kwargs.get("video_frame_counts"),
                            ),
                        )
                    )
                )
            elif key == "input_features":
                embeddings.extend(
                    self._encode_audio(
                        cast(torch.Tensor | list[torch.Tensor], value),
                        cast(
                            torch.Tensor | None,
                            kwargs.get("audio_sample_counts"),
                        ),
                    )
                )
        return embeddings

    def _run_dp_sharded_navit(
        self,
        vision_items: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run NaViT with items load-balanced across TP ranks and one all-gather."""
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        patch_counts = [self._get_patch_count(item) for item in vision_items]

        shuffle_indices, gpu_sample_counts, gpu_loads = get_load_balance_assignment(
            patch_counts, tp_size
        )
        cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

        local_image_indices = shuffle_indices[
            cum_gpu_sample_counts[tp_rank] : cum_gpu_sample_counts[tp_rank + 1]
        ]
        local_vision_items = [vision_items[i] for i in local_image_indices]

        if len(local_vision_items) > 0:
            local_output = self.vision_model(local_vision_items)
        else:
            device = next(self.vision_model.parameters()).device
            dtype = next(self.vision_model.parameters()).dtype
            local_output = torch.empty(
                0, self._adapter_output_dim, device=device, dtype=dtype
            )

        max_output_per_rank = max(gpu_loads) if max(gpu_loads) > 0 else 0

        current_len = local_output.shape[0]
        if current_len < max_output_per_rank:
            padding = torch.empty(
                max_output_per_rank - current_len,
                local_output.shape[1],
                dtype=local_output.dtype,
                device=local_output.device,
            )
            local_output_padded = torch.cat([local_output, padding], dim=0)
        else:
            local_output_padded = local_output

        gathered = tensor_model_parallel_all_gather(
            local_output_padded.contiguous(), dim=0
        )

        original_order = [torch.Tensor()] * len(vision_items)
        for rank in range(tp_size):
            rank_start = rank * max_output_per_rank
            count = gpu_sample_counts[rank]
            if count == 0:
                continue
            rank_image_indices = shuffle_indices[
                cum_gpu_sample_counts[rank] : cum_gpu_sample_counts[rank + 1]
            ]
            embed_offset = rank_start
            for img_idx in rank_image_indices:
                img_patches = patch_counts[img_idx]
                original_order[img_idx] = gathered[
                    embed_offset : embed_offset + img_patches
                ]
                embed_offset += img_patches

        return torch.cat(original_order, dim=0)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix vision/audio embeddings into token embeddings.

        Training merges audio before tok_embeddings_norm and vision after it;
        audio is pre-normed in embed_multimodal, so norming first is equivalent.
        """
        h = self.model.embed_tokens(input_ids)
        h = self.model.tok_embeddings_norm(h)

        if multimodal_embeddings is not None and is_multimodal is not None:
            h = _merge_multimodal_embeddings(h, multimodal_embeddings, is_multimodal)

        return h

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector=["vision_projection.", "perception_emb_norm."],
            tower_model=["vision_model.", "audio_encoder."],
        )


__all__ = ["OMNI_WEIGHTS_MAPPER", "MuseSpark1xOmniMixin", "register_omni_processor"]
