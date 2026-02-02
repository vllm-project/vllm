# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    BatchFeature,
    Qwen3Config,
)

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.whisper_utils import (
    ISO639_1_SUPPORTED_LANGS,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.transformers_utils.processors.funasr_processor import FunASRFeatureExtractor
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
    _require_is_multimodal,
)
from .qwen3 import Qwen3Model
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)

logger = init_logger(__name__)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, nout, dim=-1):
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super().forward(x)
        return super().forward(x.transpose(self.dim, -1)).transpose(self.dim, -1)


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        normalize_before=True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        hidden_states,
        mask,
        cache=None,
        mask_shfit_chunk=None,
        mask_att_chunk_encoder=None,
    ):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        if self.in_size == self.size:
            hidden_states = residual + self.self_attn(
                hidden_states,
                mask,
                mask_shfit_chunk=mask_shfit_chunk,
                mask_att_chunk_encoder=mask_att_chunk_encoder,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states,
                mask,
                mask_shfit_chunk=mask_shfit_chunk,
                mask_att_chunk_encoder=mask_att_chunk_encoder,
            )

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)

        return hidden_states, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder


class MultiHeadedAttentionSANM(nn.Module):
    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        kernel_size,
        sanm_shift=0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.out_proj = ColumnParallelLinear(
            input_size=n_feat,
            output_size=n_feat,
            bias=True,
        )
        self.linear_q_k_v = ColumnParallelLinear(
            input_size=in_feat,
            output_size=n_feat * 3,
            bias=True,
        )
        self.attn = None

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        b, t, d = x.size()
        q_k_v, _ = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float("inf")
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = attn
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        out, _ = self.out_proj(x)  # (batch, time1, d_model)
        return out

    def forward(
        self, hidden_states, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None
    ):
        q_h, k_h, v_h, v = self.forward_qkv(hidden_states)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory


class SinusoidalPositionEncoder(torch.nn.Module):
    def __init__(self, d_model=80):
        super().__init__()

    def encode(
        self,
        positions: torch.Tensor = None,
        depth: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype, device=device)
        ) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype)
            * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, hidden_states):
        batch_size, timesteps, input_dim = hidden_states.size()
        positions = torch.arange(1, timesteps + 1, device=hidden_states.device)[None, :]
        position_encoding = self.encode(positions, input_dim, hidden_states.dtype).to(
            hidden_states.device
        )

        return hidden_states + position_encoding


class SenseVoiceEncoderSmall(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        kernel_size: int = 11,
        sanm_shift: int = 0,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = SinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            kernel_size,
            sanm_shift,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            kernel_size,
            sanm_shift,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                )
                for i in range(num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(output_size)

        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        maxlen = xs_pad.shape[1]
        masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]

        xs_pad *= self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, idim, hidden_units):
        super().__init__()
        self.w_1 = ColumnParallelLinear(
            input_size=idim,
            output_size=hidden_units,
            bias=True,
        )
        self.w_2 = RowParallelLinear(
            input_size=hidden_units,
            output_size=idim,
            bias=True,
        )
        self.activation = _ACTIVATION_REGISTRY["relu"]

    def forward(self, hidden_states):
        hidden_states, _ = self.w_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.w_2(hidden_states)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, None, None)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)

        return hidden_states


class FunASRAudioAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // tp_size

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scaling = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
            prefix=f"{prefix}.qkv",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            prefix=f"{prefix}.out_proj",
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            scale=self.scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        bs, seq_length, _ = hidden_states.size()
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bs, seq_length, -1, self.head_dim)
        k = k.view(bs, seq_length, -1, self.head_dim)
        v = v.view(bs, seq_length, -1, self.head_dim)

        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        attn_output = attn_output.view(bs, seq_length, -1)
        output, _ = self.out_proj(attn_output)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        downsample_rate=2,
        encoder_dim=1280,
        llm_dim=4096,
        ffn_dim: int = 2048,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self.k = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = ColumnParallelLinear(
            input_size=self.encoder_dim * self.k,
            output_size=ffn_dim,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.linear2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=self.llm_dim,
            bias=True,
        )

        self.blocks = None
        if kwargs.get("n_layer", 2) > 0:
            self.blocks = nn.ModuleList(
                [
                    EncoderLayer(
                        llm_dim,
                        FunASRAudioAttention(
                            kwargs.get("attention_heads", 8),
                            llm_dim,
                            prefix=f"{prefix}.self_attn",
                        ),
                        PositionwiseFeedForward(
                            llm_dim,
                            llm_dim // 4,
                        ),
                    )
                    for _ in range(kwargs.get("n_layer", 2))
                ]
            )

    def forward(self, hidden_states, ilens=None):
        batch_size, seq_len, dim = hidden_states.size()
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_num, 0, 0), value=0.0)
        seq_len = hidden_states.size(1)

        hidden_states = hidden_states.contiguous()
        hidden_states = hidden_states.view(batch_size, chunk_num, dim * self.k)
        hidden_states, _ = self.linear1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states, _ = self.linear2(hidden_states)

        olens = None
        olens = (ilens - 1) // self.k + 1

        if self.blocks is not None:
            for layer, block in enumerate(self.blocks):
                hidden_states = block(hidden_states)
        return hidden_states, olens


class FunASRAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins
        - t: Time frames (M)
    """

    input_features: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b", "nmb", "t"),
    ]
    speech_lengths: Annotated[
        list[torch.Tensor] | None,
        TensorShape("m"),
    ]


class FunASREncoder(nn.Module):
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ):
        super().__init__()
        self.audio_encoder = SenseVoiceEncoderSmall(
            input_size=560, **vllm_config.model_config.hf_config.audio_encoder_conf
        )
        self.audio_adaptor = Transformer(
            downsample_rate=1,
            use_low_frame_rate=True,
            ffn_dim=2048,
            llm_dim=1024,
            encoder_dim=512,
            n_layer=2,
            freeze=True,
            prefix=maybe_prefix(prefix, "audio_encoder"),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with mapping from HuggingFace format."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attn.qkv.", "self_attn.q_proj.", "q"),
            ("self_attn.qkv.", "self_attn.k_proj.", "k"),
            ("self_attn.qkv.", "self_attn.v_proj.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class FunASRModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = FunASREncoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "encoder")
        )
        self.decoder = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "decoder")
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
        speech: torch.Tensor | list[torch.Tensor] | None,
        speech_lengths: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        self.feat_permute = False

        if self.feat_permute:
            encoder_out, encoder_out_lens = self.encoder.audio_encoder(
                speech.permute(0, 2, 1), speech_lengths
            )
        else:
            encoder_out, encoder_out_lens = self.encoder.audio_encoder(
                speech, speech_lengths
            )

        encoder_out, encoder_out_lens = self.encoder.audio_adaptor(
            encoder_out, encoder_out_lens
        )
        return encoder_out


class FunASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Qwen3Config:
        return self.ctx.get_hf_config(Qwen3Config)

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # Because the encoder prompt is padded

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs: object) -> FunASRFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, FunASRFeatureExtractor)
        return feature_extractor

    def get_target_channels(self) -> int:
        return 1

    def get_num_audio_tokens(self) -> int:
        return self.get_hf_config().max_source_positions


class FunASRDummyInputsBuilder(BaseDummyInputsBuilder[FunASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|AUDIO|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class FunASRMultiModalProcessor(BaseMultiModalProcessor[FunASRProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.info.get_target_channels(),
        )

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = dict(audio=mm_data.pop("audios"))
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            speech_lengths=MultiModalFieldConfig.batched("audio"),
            fake_token_len=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")

        audio_token_id = vocab[audio_token]

        out_mm_data = out_mm_kwargs.get_data()

        fake_token_len = out_mm_data.get("fake_token_len")
        if fake_token_len is None:
            audio_output_lengths = []
        else:
            assert isinstance(fake_token_len, torch.Tensor)
            # _, audio_output_lens = _get_feat_extract_output_lengths(
            #    feature_attention_mask.sum(-1)
            # )

            audio_output_lengths = fake_token_len.tolist()

        def get_replacement_qwen2_audio(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape) == 2, "audio_embeds must be a 2D tensor"
                num_features = audio_embeds.shape[0]

            # if num_features == 0:
            #    audios = mm_items.get_items("audio", AudioProcessorItems)
            #    audio_len = audios.get_audio_length(item_idx)

            #    raise ValueError(
            #        f"The audio (len={audio_len}) is too short "
            #        "to be represented inside the model"
            #    )

            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    FunASRMultiModalProcessor,
    info=FunASRProcessingInfo,
    dummy_inputs=FunASRDummyInputsBuilder,
)
class FunASRForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "linear_q.": "q_proj.",
            "linear_k.": "k_proj.",
            "linear_v.": "v_proj.",
            "linear_out.": "out_proj.",
        }
    )

    supports_transcription_only = True
    supports_segment_timestamp = True
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            # TODO language should be optional and can be guessed.
            # For now we default to en. See
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py#L1520
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "en"
        return super().validate_language(language)

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,  # not needed here
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        # processor = cached_processor_from_config(model_config)
        if language is None:
            raise ValueError(
                "Language must be specified when creating the funasr prompt"
            )

        funasr_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：<|AUDIO|><|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        prompt = {
            "prompt": funasr_prompt,
            "multi_modal_data": {
                "audio": (audio, stt_config.sample_rate),
            },
        }
        return cast(PromptType, prompt)

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)

        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        processor = cached_processor_from_config(model_config)
        hop_length = processor.feature_extractor.hop_length
        assert hop_length is not None
        return math.ceil(audio_duration_s * stt_config.sample_rate / hop_length)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = FunASRModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        logit_scale = getattr(config, "logit_scale", 1.0)

        if config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        decoder_outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        return decoder_outputs

    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        speech = audio_input["input_features"]
        speech_lengths = audio_input["speech_lengths"]
        enc_output = self.model.get_encoder_outputs(
            speech=speech, speech_lengths=speech_lengths
        )

        return enc_output

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.model.decoder.embed_input_ids(input_ids)

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> FunASRAudioInputs:
        input_features = kwargs.pop("input_features", None)
        speech_lengths = kwargs.pop("speech_lengths", None)

        if input_features is not None:
            input_features = json_map_leaves(lambda x: x.to(self.dtype), input_features)

        if speech_lengths is not None:
            speech_lengths = json_map_leaves(lambda x: x.to(self.dtype), speech_lengths)

        return FunASRAudioInputs(
            input_features=input_features, speech_lengths=speech_lengths
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
        )

        # add fake zeros bias for k_proj to state_dict
        weights = _create_fake_bias_for_k_proj(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _create_fake_bias_for_k_proj(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """
    Create full zeros bias for k_proj weight in self-attn and x-attn layers.
    So that the bias for k_proj in qkv_proj can be initialized with zeros.
    """
    for name, weight in weights:
        if name.endswith(".k_proj.weight"):
            bias = torch.zeros(weight.size(0))
            bias_name = name.replace("weight", "bias")
            yield from [(name, weight), (bias_name, bias)]
        else:
            yield name, weight
