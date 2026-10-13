# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EmbeddingGemma2: bidirectional Gemma4-derived multimodal embedding model."""

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModel, BatchFeature, PreTrainedConfig
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
    EmbeddingGemma2Config,
    EmbeddingGemma2TextConfig,
)
from transformers.models.embedding_gemma2.processing_embedding_gemma2 import (
    EmbeddingGemma2Processor,
)
from transformers.video_utils import VideoMetadata

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.gemma4 import Gemma4MLP
from vllm.model_executor.models.gemma4_mm import (
    _SUPPORTED_SOFT_TOKENS,
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
    _get_max_soft_tokens,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    default_pooling_type,
)
from vllm.model_executor.models.transformers.utils import (
    recursive_replace_linear,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)

_WINDOW_OFFSET = 1  # HF masks |q-k| <= W; vLLM ENCODER_ONLY masks <= W-1.
_VIDEO_FRAME_WIDTH = 672
_VIDEO_FRAME_HEIGHT = 480
_VIDEO_MAX_FRAMES = 32
_VIDEO_MAX_SOFT_TOKENS = 140


class EmbeddingGemma2Attention(nn.Module):
    """Bidirectional attention with sliding window offset for EmbeddingGemma2."""

    def __init__(
        self,
        config: EmbeddingGemma2TextConfig,
        layer_cfg,
        layer_idx: int,
        cache_config,
        quant_config: QuantizationConfig | None,
        max_position: int,
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        tp = get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, layer_cfg.num_key_value_heads // tp)
        self.head_dim = layer_cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            layer_cfg.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

        assert config.layer_types is not None
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        assert config.rope_parameters is not None
        rope_parameters = dict(config.rope_parameters[layer_type])

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        window = config.sliding_window if self.is_sliding else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=(window + _WINDOW_OFFSET)
            if window is not None
            else None,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)

        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = self.v_norm(v)
        v = v.flatten(-2, -1)

        attn_output = self.attn(q, k, v)
        out, _ = self.o_proj(attn_output)
        return out


class EmbeddingGemma2PLEBlock(nn.Module):
    """Per-layer-embedding residual block (mirrors HF ple_block)."""

    def __init__(
        self,
        config: EmbeddingGemma2TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.per_layer_input_gate = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size_per_layer_input,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.per_layer_input_gate",
            return_bias=False,
        )
        self.per_layer_projection = ReplicatedLinear(
            self.hidden_size_per_layer_input,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.per_layer_projection",
            return_bias=False,
        )
        self.post_per_layer_input_norm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.act_fn = get_act_fn(config.hidden_activation)

    def forward(
        self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        gate = self.per_layer_input_gate(hidden_states)
        gate = self.act_fn(gate)
        gated = gate * per_layer_input
        proj = self.per_layer_projection(gated)
        proj = self.post_per_layer_input_norm(proj)
        return residual + proj


class EmbeddingGemma2DecoderLayer(nn.Module):
    """Transformer decoder layer for EmbeddingGemma2 text model with PLE gating."""

    def __init__(
        self,
        config: EmbeddingGemma2TextConfig,
        idx: int,
        cache_config,
        quant_config: QuantizationConfig | None,
        max_position: int,
        prefix: str,
    ):
        super().__init__()
        lc = config.per_layer_config[idx]
        self.self_attn = EmbeddingGemma2Attention(
            config,
            lc,
            idx,
            cache_config,
            quant_config,
            max_position,
            f"{prefix}.self_attn",
        )
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.ple_block = EmbeddingGemma2PLEBlock(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.ple_block",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scalar: torch.Tensor
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        positions: torch.Tensor,
        h: torch.Tensor,
        per_layer_input: torch.Tensor,
    ) -> torch.Tensor:
        r = h
        h = (
            self.post_attention_layernorm(
                self.self_attn(positions, self.input_layernorm(h))
            )
            + r
        )
        r = h
        h = (
            self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
            + r
        )
        h = self.ple_block(h, per_layer_input)
        return h * cast(torch.Tensor, self.layer_scalar)


class EmbeddingGemma2TextPLE(nn.Module):
    """Per-layer embedding (PLE) projection and normalization module."""

    def __init__(
        self,
        config: EmbeddingGemma2TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_model_projection = ColumnParallelLinear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.per_layer_model_projection",
        )
        self.per_layer_model_projection_scale = config.hidden_size**-0.5
        self.per_layer_projection_norm = RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        proj, _ = self.per_layer_model_projection(inputs_embeds)
        proj = proj * self.per_layer_model_projection_scale
        proj = proj.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return self.per_layer_projection_norm(proj)


class EmbeddingGemma2TextModel(nn.Module):
    """Text embedding backbone for EmbeddingGemma2 with bidirectional attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config.text_config
        self.config = cfg
        max_pos = min(
            cfg.max_position_embeddings, vllm_config.model_config.max_model_len
        )
        self.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                EmbeddingGemma2DecoderLayer(
                    cfg,
                    i,
                    vllm_config.cache_config,
                    vllm_config.quant_config,
                    max_pos,
                    f"{prefix}.layers.{i}",
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ple = EmbeddingGemma2TextPLE(
            cfg,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.ple",
        )
        self.embedding_projection = ReplicatedLinear(
            cfg.hidden_size,
            cfg.embedding_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embedding_projection",
            return_bias=False,
        )
        self.embed_scale: torch.Tensor
        self.register_buffer(
            "embed_scale",
            torch.tensor(cfg.hidden_size**0.5),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_scale = cast(torch.Tensor, self.embed_scale)
        dtype = cast(torch.dtype, self.embed_tokens.weight.dtype)
        return self.embed_tokens(input_ids) * embed_scale.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        per_layer = self.ple(inputs_embeds)
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(positions, h, per_layer[..., i, :])
        h = self.norm(h)
        h = self.embedding_projection(h)
        return h


class EmbeddingGemma2ProcessingInfo(Gemma4ProcessingInfo):
    """Processing information and token budgeting for EmbeddingGemma2."""

    def get_hf_config(self) -> EmbeddingGemma2Config:
        return self.ctx.get_hf_config(EmbeddingGemma2Config)

    def get_hf_processor(  # type: ignore[override]
        self, **kwargs: object
    ) -> EmbeddingGemma2Processor:
        return self.ctx.get_hf_processor(EmbeddingGemma2Processor, **kwargs)

    def get_default_tok_params(self):
        # Base-class behavior: add_special_tokens=True (BOS + EOS)
        return BaseProcessingInfo.get_default_tok_params(self)

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        config = self.get_hf_config()
        merged_kwargs = self.ctx.get_merged_mm_kwargs({})
        val, _ = _get_max_soft_tokens(merged_kwargs)
        if isinstance(val, int) and val in _SUPPORTED_SOFT_TOKENS:
            tokens_per_image = val + 2
        else:
            tokens_per_image = (
                getattr(config.vision_config, "default_output_length", 256) + 2
            )
        tokens: dict[str, int] = {"image": tokens_per_image}
        if config.audio_config is not None:
            processor = self.get_hf_processor()
            tokens["audio"] = processor.audio_seq_length + 2

        processor = self.get_hf_processor()
        video_proc = getattr(processor, "video_processor", None)
        max_frames = getattr(video_proc, "max_frames", 32)
        mm_config = self.ctx.model_config.get_multimodal_config()

        # The loader's max_frames cap is 128 when media_io_kwargs can raise it
        video_media_io = (
            getattr(mm_config, "media_io_kwargs", {}).get("video", {})
            if mm_config
            else {}
        )
        media_io_max_frames = video_media_io.get("max_frames")
        if media_io_max_frames is not None:
            max_frames = min(int(media_io_max_frames), 128)
        elif "max_frames" in merged_kwargs:
            max_frames = min(int(merged_kwargs["max_frames"]), 128)

        video_opts = mm_config.limit_per_prompt.get("video") if mm_config else None
        if video_opts is not None:
            num_frames_opt = getattr(video_opts, "num_frames", None)
            if isinstance(num_frames_opt, int):
                max_frames = min(max_frames, num_frames_opt)

        video_max_soft = merged_kwargs.get(
            "max_soft_tokens",
            getattr(video_proc, "max_soft_tokens", _VIDEO_MAX_SOFT_TOKENS),
        )
        if video_max_soft is not None and video_max_soft not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"Unsupported max_soft_tokens={video_max_soft} for video. "
                f"Valid values are {_SUPPORTED_SOFT_TOKENS}."
            )
        if not isinstance(video_max_soft, int):
            video_max_soft = _VIDEO_MAX_SOFT_TOKENS

        # Token allocation per video frame verified against HF EmbeddingGemma2Processor:
        # - Without timestamps: boi (1) + video_token * num_soft_tokens + eoi (1)
        #   Total = num_soft_tokens + 2
        # - With timestamps: timestamp prefix string (e.g. ' 01:23 ' = up to 7 tokens)
        #   Total = num_soft_tokens + 2 + 7
        add_ts = getattr(video_proc, "add_timestamps", False) or merged_kwargs.get(
            "add_timestamps", False
        )
        ts_slack = 7 if add_ts else 0
        tokens["video"] = max_frames * (video_max_soft + 2 + ts_slack)
        return tokens

    def get_image_repl(  # type: ignore[override]
        self,
        *,
        image_width: int,
        image_height: int,
        processor: EmbeddingGemma2Processor | None,
        max_soft_tokens: int | None = None,
    ) -> PromptUpdateDetails:
        if processor is None:
            processor = self.get_hf_processor()

        num_soft = self._compute_num_soft_tokens(
            image_width,
            image_height,
            max_soft_tokens=max_soft_tokens,
        )
        config = self.get_hf_config()
        assert config.boi_token_id is not None
        assert processor.image_token_id is not None
        assert config.eoi_token_id is not None
        token_ids = (
            [config.boi_token_id]
            + [processor.image_token_id] * num_soft
            + [config.eoi_token_id]
        )
        return PromptUpdateDetails.select_token_id(token_ids, processor.image_token_id)

    def get_audio_repl(  # type: ignore[override]
        self,
        *,
        audio_len: int,
        processor: EmbeddingGemma2Processor | None,
    ) -> PromptUpdateDetails:
        if processor is None:
            processor = self.get_hf_processor()
        sampling_rate = processor.feature_extractor.sampling_rate
        num_tokens = self._compute_audio_num_tokens(
            audio_len, sampling_rate, processor.audio_seq_length
        )
        config = self.get_hf_config()
        assert config.boa_token_id is not None
        assert processor.audio_token_id is not None
        assert config.eoa_token_index is not None
        token_ids = (
            [config.boa_token_id]
            + [processor.audio_token_id] * num_tokens
            + [config.eoa_token_index]
        )
        return PromptUpdateDetails.select_token_id(token_ids, processor.audio_token_id)

    def get_video_repl(  # type: ignore[override]
        self,
        *,
        timestamps: list[float],
        num_soft_tokens_per_frame: list[int],
        processor: EmbeddingGemma2Processor,
        add_timestamps: bool = False,
    ) -> PromptUpdateDetails:
        tokenizer = self.ctx.get_tokenizer()
        config = self.get_hf_config()

        boi_token_id = config.boi_token_id
        eoi_token_id = config.eoi_token_id
        video_token_id = processor.video_token_id
        assert boi_token_id is not None
        assert eoi_token_id is not None
        assert video_token_id is not None

        all_token_ids: list[int] = []
        if not add_timestamps:
            for n_tokens in num_soft_tokens_per_frame:
                all_token_ids.append(boi_token_id)
                all_token_ids.extend([video_token_id] * n_tokens)
                all_token_ids.append(eoi_token_id)
        else:
            for i, (ts, n_tokens) in enumerate(
                zip(timestamps, num_soft_tokens_per_frame)
            ):
                minutes = int(ts // 60)
                seconds = int(ts % 60)
                ts_str = f"{minutes:02d}:{seconds:02d}"
                prefix = f" {ts_str} " if i > 0 else f"{ts_str} "
                ts_token_ids = tokenizer.encode(prefix, add_special_tokens=False)
                all_token_ids.extend(ts_token_ids)
                all_token_ids.append(boi_token_id)
                all_token_ids.extend([video_token_id] * n_tokens)
                all_token_ids.append(eoi_token_id)

        return PromptUpdateDetails.select_token_id(all_token_ids, video_token_id)


class EmbeddingGemma2DummyInputsBuilder(Gemma4DummyInputsBuilder):
    """Dummy input builder for EmbeddingGemma2 profiling and testing."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        num_videos = mm_counts.get("video", 0)
        processor = self.info.get_hf_processor()
        text = (
            (processor.image_token * num_images)
            + (processor.audio_token * num_audios if processor.audio_token else "")
            + (processor.video_token * num_videos)
        )
        return text

    def _get_dummy_videos(  # type: ignore[override]
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides=None,
    ) -> list[VideoItem]:
        num_frames = _VIDEO_MAX_FRAMES
        width = _VIDEO_FRAME_WIDTH
        height = _VIDEO_FRAME_HEIGHT
        videos = super(Gemma4DummyInputsBuilder, self)._get_dummy_videos(
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=overrides,
        )
        video_items: list[VideoItem] = []
        for video in videos:
            video_num_frames = video.shape[0]
            video_metadata = {
                "fps": 1.0,
                "duration": float(video_num_frames),
                "total_num_frames": video_num_frames,
                "frames_indices": list(range(video_num_frames)),
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_items.append((video, video_metadata))
        return video_items


class EmbeddingGemma2MultiModalProcessor(Gemma4MultiModalProcessor):
    """Multimodal input processor for EmbeddingGemma2."""

    info: EmbeddingGemma2ProcessingInfo

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )

        if not hf_data:
            return self._finalize_hf_mm_data(
                hf_data, hf_kwargs, passthrough_data, BatchFeature()
            )

        prompt_text = hf_data.pop("text")
        assert isinstance(prompt_text, str)

        processor = self.info.get_hf_processor()
        video_outputs: dict[str, Any] = {}

        if videos := hf_data.pop("videos", []):
            assert isinstance(videos, list)

            all_video_pixel_values: list[torch.Tensor] = []
            all_video_position_ids: list[torch.Tensor] = []
            video_num_soft_tokens_per_video: list[list[int]] = []
            video_timestamps_per_video: list[list[float]] = []
            video_frame_counts: list[int] = []

            for item in videos:
                video_array, metadata = item

                if hf_kwargs.get("num_frames") not in (-1, None):
                    raise ValueError(
                        "EmbeddingGemma2 does not accept num_frames; "
                        "use max_frames/fps."
                    )

                if isinstance(metadata, dict):
                    if not metadata.get("do_sample_frames", True) and any(
                        k in hf_kwargs
                        for k in ("fps", "max_frames", "overflow_strategy")
                    ):
                        logger.warning_once(
                            "For file/URL-loaded videos, "
                            "fps/max_frames/overflow_strategy in mm_processor_kwargs "
                            "have no effect because frames are decoded and sampled by "
                            "media_io. Please specify these sampling options via "
                            "media_io_kwargs instead."
                        )
                    valid_meta_fields = {
                        f.name for f in dataclasses.fields(VideoMetadata)
                    }
                    meta_dict = {
                        k: v for k, v in metadata.items() if k in valid_meta_fields
                    }
                    metadata_obj = VideoMetadata(**meta_dict)
                else:
                    if getattr(metadata, "do_sample_frames", True) is False and any(
                        k in hf_kwargs
                        for k in ("fps", "max_frames", "overflow_strategy")
                    ):
                        logger.warning_once(
                            "For file/URL-loaded videos, "
                            "fps/max_frames/overflow_strategy in mm_processor_kwargs "
                            "have no effect because frames are decoded and sampled by "
                            "media_io. Please specify these sampling options via "
                            "media_io_kwargs instead."
                        )
                    metadata_obj = metadata

                video_mm_kwargs = dict(hf_kwargs)
                video_mm_kwargs.pop("num_frames", None)
                video_mm_kwargs["return_metadata"] = True
                if isinstance(metadata, dict) and "do_sample_frames" in metadata:
                    video_mm_kwargs["do_sample_frames"] = metadata["do_sample_frames"]
                elif hasattr(metadata, "do_sample_frames"):
                    video_mm_kwargs["do_sample_frames"] = metadata.do_sample_frames

                single_video_out = self.info.ctx.call_hf_processor(
                    self.info.get_hf_processor(**video_mm_kwargs),
                    dict(
                        text=processor.video_token,
                        videos=[video_array],
                        video_metadata=[metadata_obj],
                    ),
                    video_mm_kwargs,
                )

                all_video_pixel_values.append(single_video_out["pixel_values_videos"])
                all_video_position_ids.append(single_video_out["video_position_ids"])

                num_frames = int(single_video_out["num_frames_per_video"][0])
                video_frame_counts.append(num_frames)

                num_soft = single_video_out.get("num_soft_tokens_per_video")
                if num_soft is not None:
                    soft_val = num_soft[0]
                    if isinstance(soft_val, torch.Tensor):
                        soft_val = soft_val.item()
                    soft_tokens_list = [int(soft_val)] * num_frames
                elif "video_position_ids" in single_video_out and num_frames > 0:
                    pos = single_video_out["video_position_ids"]
                    pooling_k2 = (
                        getattr(processor.video_processor, "pooling_kernel_size", 3)
                        ** 2
                    )
                    valid_patches = (pos[0] != -1).all(dim=-1).sum().item()
                    soft_tokens_list = [valid_patches // pooling_k2] * num_frames
                elif "input_ids" in single_video_out and num_frames > 0:
                    vtok = (
                        (
                            single_video_out["input_ids"][0]
                            == processor.tokenizer.video_token_id
                        )
                        .sum()
                        .item()
                    )
                    soft_tokens_list = [vtok // num_frames] * num_frames
                else:
                    raise ValueError(
                        f"Unable to determine soft tokens for video frame: "
                        f"keys={list(single_video_out.keys())}"
                    )

                video_num_soft_tokens_per_video.append(soft_tokens_list)

                # Use returned video_metadata if available, else input metadata_obj
                ret_meta = single_video_out.get("video_metadata")
                meta_for_ts = (
                    ret_meta[0] if (ret_meta and len(ret_meta) > 0) else metadata_obj
                )

                add_ts = hf_kwargs.get(
                    "add_timestamps",
                    getattr(processor.video_processor, "add_timestamps", False),
                )
                if add_ts:
                    try:
                        ts = meta_for_ts.timestamps
                    except ValueError as e:
                        raise ValueError(
                            "Asked to build a prompt with frame timestamps, but no "
                            "valid fps/frames_indices was available in video metadata."
                        ) from e
                    video_timestamps_per_video.append([float(t) for t in ts])
                else:
                    try:
                        ts = meta_for_ts.timestamps
                        video_timestamps_per_video.append([float(t) for t in ts])
                    except ValueError:
                        video_timestamps_per_video.append([0.0] * num_frames)

            video_outputs = {
                "pixel_values_videos": torch.cat(all_video_pixel_values, dim=0),
                "pixel_position_ids_videos": torch.cat(all_video_position_ids, dim=0),
                "video_frame_counts": torch.tensor(
                    video_frame_counts, dtype=torch.long
                ),
                "video_num_soft_tokens": video_num_soft_tokens_per_video,
                "video_timestamps": video_timestamps_per_video,
            }

        if hf_data:
            remaining_counts: dict[str, int] = {}
            if "images" in hf_data:
                images_list = hf_data["images"]
                remaining_counts["image"] = (
                    len(images_list) if isinstance(images_list, list) else 1
                )
            if "audio" in hf_data:
                audio_list = hf_data["audio"]
                remaining_counts["audio"] = (
                    len(audio_list) if isinstance(audio_list, (list, tuple)) else 1
                )
            remaining_prompt_text = self.dummy_inputs.get_dummy_text(remaining_counts)

            call_kwargs = dict(hf_kwargs)
            call_kwargs.pop("num_frames", None)
            processed_data = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**call_kwargs),
                dict(text=remaining_prompt_text, **hf_data),
                call_kwargs,
            )
        else:
            processed_data = BatchFeature()

        if "image_position_ids" in processed_data:
            processed_data["pixel_position_ids"] = processed_data.pop(
                "image_position_ids"
            )

        if "input_features" in processed_data:
            input_features = processed_data.pop("input_features")
            input_features_mask = processed_data.pop("input_features_mask")
            if not isinstance(input_features, list):
                unpadded_features = []
                unpadded_masks = []
                for i in range(input_features.shape[0]):
                    mask = input_features_mask[i]
                    valid_len = int(mask.sum().item())
                    unpadded_features.append(input_features[i, :valid_len])
                    unpadded_masks.append(mask[:valid_len])
                processed_data["input_features_padded"] = unpadded_features
                processed_data["input_features_mask"] = unpadded_masks
            else:
                processed_data["input_features_padded"] = input_features
                processed_data["input_features_mask"] = input_features_mask

        processed_data.update(video_outputs)
        processed_data.pop("num_frames_per_video", None)
        processed_data.pop("video_position_ids", None)

        return self._finalize_hf_mm_data(
            hf_data, hf_kwargs, passthrough_data, processed_data
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        prompt_updates = []

        if "image" in mm_items:
            image_token_id = hf_processor.image_token_id

            def get_replacement_image(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)
                image_size = images.get_image_size(item_idx)
                merged_kwargs = self.info.ctx.get_merged_mm_kwargs(
                    hf_processor_mm_kwargs,
                )
                val, _ = _get_max_soft_tokens(merged_kwargs)
                max_soft_tokens = (
                    val
                    if isinstance(val, int) and val in _SUPPORTED_SOFT_TOKENS
                    else None
                )
                return self.info.get_image_repl(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                    max_soft_tokens=max_soft_tokens,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_replacement_image,
                )
            )

        if "video" in mm_items:
            video_token_id = hf_processor.video_token_id
            assert video_token_id is not None

            def get_replacement_video(item_idx: int):
                out_item = out_mm_kwargs["video"][item_idx]
                raw_ts = out_item["video_timestamps"].data
                if isinstance(raw_ts, torch.Tensor):
                    ts_list: list[Any] = raw_ts.tolist()
                elif isinstance(raw_ts, (list, tuple)):
                    ts_list = list(raw_ts)
                else:
                    ts_list = [raw_ts]
                timestamps = [float(t) for t in ts_list]

                raw_soft = out_item["video_num_soft_tokens"].data
                if isinstance(raw_soft, torch.Tensor):
                    soft_list: list[Any] = raw_soft.tolist()
                elif isinstance(raw_soft, (list, tuple)):
                    soft_list = list(raw_soft)
                else:
                    soft_list = [raw_soft]
                num_soft = [int(n) for n in soft_list]

                add_ts = hf_processor_mm_kwargs.get(
                    "add_timestamps",
                    getattr(hf_processor.video_processor, "add_timestamps", False),
                )
                return self.info.get_video_repl(
                    timestamps=timestamps,
                    num_soft_tokens_per_frame=num_soft,
                    processor=hf_processor,
                    add_timestamps=add_ts,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="video",
                    target=[video_token_id],
                    replacement=get_replacement_video,
                )
            )

        if "audio" in mm_items:
            audio_token_id = hf_processor.audio_token_id
            assert audio_token_id is not None

            def get_replacement_audio(item_idx: int):
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)
                return self.info.get_audio_repl(
                    audio_len=audio_len,
                    processor=hf_processor,
                )

            prompt_updates.append(
                PromptReplacement(
                    modality="audio",
                    target=[audio_token_id],
                    replacement=get_replacement_audio,
                )
            )

        return prompt_updates


def _embedding_gemma2_weights_mapper(
    config_or_layers: PreTrainedConfig | int | None = None,
) -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_stacked={
            ".self_attn.q_proj.weight": (".self_attn.qkv_proj.weight", "q"),
            ".self_attn.k_proj.weight": (".self_attn.qkv_proj.weight", "k"),
            ".self_attn.v_proj.weight": (".self_attn.qkv_proj.weight", "v"),
            ".self_attn.q_proj.bias": (".self_attn.qkv_proj.bias", "q"),
            ".self_attn.k_proj.bias": (".self_attn.qkv_proj.bias", "k"),
            ".self_attn.v_proj.bias": (".self_attn.qkv_proj.bias", "v"),
            ".mlp.gate_proj.weight": (".mlp.gate_up_proj.weight", 0),
            ".mlp.up_proj.weight": (".mlp.gate_up_proj.weight", 1),
            ".mlp.gate_proj.bias": (".mlp.gate_up_proj.bias", 0),
            ".mlp.up_proj.bias": (".mlp.gate_up_proj.bias", 1),
        }
    )


@MULTIMODAL_REGISTRY.register_processor(
    EmbeddingGemma2MultiModalProcessor,
    info=EmbeddingGemma2ProcessingInfo,
    dummy_inputs=EmbeddingGemma2DummyInputsBuilder,
)
@default_pooling_type(seq_pooling_type="MEAN", tok_pooling_type="ALL")
class EmbeddingGemma2Model(nn.Module, SupportsMultiModal, VllmModelForPooling):
    """Multimodal pooling embedding model supporting text, vision, and audio."""

    is_pooling_model = True

    hf_to_vllm_mapper = _embedding_gemma2_weights_mapper()
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # TODO(lucianomartins): Extract Gemma4EncoderMixin into a shared module
    # once generative model tests are migrated.
    # TODO(lucianomartins): Investigate @support_torch_compile integration for encoder.
    _parse_and_validate_image_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_image_input
    )
    _parse_and_validate_video_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_video_input
    )
    _parse_and_validate_audio_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_audio_input
    )
    _parse_and_validate_multimodal_inputs = (
        Gemma4ForConditionalGeneration._parse_and_validate_multimodal_inputs
    )
    _process_image_input = Gemma4ForConditionalGeneration._process_image_input
    _process_video_input = Gemma4ForConditionalGeneration._process_video_input
    _process_audio_input = Gemma4ForConditionalGeneration._process_audio_input
    embed_multimodal = Gemma4ForConditionalGeneration.embed_multimodal
    _encoder_chunk = staticmethod(Gemma4ForConditionalGeneration._encoder_chunk)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.model_dtype = vllm_config.model_config.dtype
        self.vllm_config = vllm_config
        self.hf_to_vllm_mapper = _embedding_gemma2_weights_mapper(config)
        lora_config = vllm_config.lora_config
        self._enable_mm_lora = bool(
            lora_config is not None and lora_config.enable_tower_connector_lora
        )

        tower_quant: QuantizationConfig | None
        if quant_config and quant_config.get_name() in [
            "bitsandbytes",
            "torchao",
            "compressed-tensors",
        ]:
            tower_quant = quant_config
        else:
            vision_cfg = getattr(config, "vision_config", None)
            quantizable = (
                vision_cfg is not None
                and vision_cfg.hidden_size % 64 == 0
                and vision_cfg.intermediate_size % 64 == 0
            )
            tower_quant = quant_config if quantizable else None

        # ---- Vision tower (shared by image and video) ----
        self.embed_vision: Gemma4MultimodalEmbedder | None
        if getattr(config, "vision_config", None) is not None:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.vision_tower = AutoModel.from_config(config=config.vision_config)
                self.embed_vision = Gemma4MultimodalEmbedder(
                    config.vision_config,
                    config.text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_vision"),
                )
                recursive_replace_linear(
                    self.vision_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )
        else:
            self.vision_tower = None
            self.embed_vision = None

        # ---- Audio tower ----
        self.embed_audio: Gemma4MultimodalEmbedder | None
        if getattr(config, "audio_config", None) is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = AutoModel.from_config(config=config.audio_config)
                self.audio_tower.post_init()
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config,
                    config.text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_audio"),
                )
                recursive_replace_linear(
                    self.audio_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "audio_tower"),
                )
        else:
            self.audio_tower = None
            self.embed_audio = None

        # ---- Language model ----
        with self._mark_language_model(vllm_config):
            self.language_model = EmbeddingGemma2TextModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        assert vllm_config.model_config.pooler_config is not None
        self.pooler = DispatchPooler.for_embedding(
            vllm_config.model_config.pooler_config
        )

    def get_language_model(self) -> nn.Module:  # type: ignore[override]
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions, inputs_embeds=inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ignore_prefixes = [
            "embed_vision.embedding.",
            "embed_audio.embedding.",
        ]
        if self.audio_tower is None:
            ignore_prefixes.extend(["audio_tower.", "embed_audio."])
        if self.vision_tower is None:
            ignore_prefixes.extend(["vision_tower.", "embed_vision."])
        return AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=ignore_prefixes,
        ).load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int = 0) -> str | None:
        if modality == "image":
            return "<|image|>"
        if modality == "audio":
            return "<|audio|>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")
