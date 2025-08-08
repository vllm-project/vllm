# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/MoonshotAI/Kimi-Audio/blob/master/finetune_codes/modeling_kimia.py
"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypedDict, Union, Tuple, List, Dict

import os
import numpy as np
import torch
import torch.nn as nn
import transformers

from transformers import AutoTokenizer, BatchFeature
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from vllm.config import VllmConfig
from ...transformers_utils.configs import KimiAudioConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from .moonaudio import MoonshotKimiaForCausalLM
from ...transformers_utils.processors import KimiAudioProcessor, WhisperEncoder
from ...transformers_utils.tokenizers import Glm4Tokenizer
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, maybe_prefix,
                    merge_multimodal_embeddings)

from packaging import version

assert version.parse(transformers.__version__) >= version.parse("4.34.1")

if version.parse(transformers.__version__) >= version.parse("4.35.0"):
    from transformers.utils import is_flash_attn_2_available as is_flash_attn_available
else:
    from transformers.utils import is_flash_attn_available

if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
else:
    raise RuntimeError("flash attention must be installed")

from vllm.logger import init_logger

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from .utils import maybe_prefix
from vllm.model_executor.layers.logits_processor import LogitsProcessor

logger = init_logger(__name__)


class KimiAudioMultiModalProjector(nn.Module):
    """Equivalent to VQAdaptor"""
    
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)
    

# === Audio Inputs === #
class KimiAudioInputs(TypedDict):
    whisper_input_feature: List[torch.Tensor]
    """Shape: `(num_audios, seq_len, feature_dim)`"""
    
    is_continuous_mask: torch.Tensor
    """Shape: `(num_audios, seq_len)`"""

    audio_input_ids: Optional[torch.Tensor]


# === Processing Info === #
class KimiAudioProcessingInfo(BaseProcessingInfo):
    
    def get_hf_config(self) -> KimiAudioConfig:
        return self.ctx.get_hf_config(KimiAudioConfig)
    
    def get_tokenizer(self) -> AutoTokenizer:
        return super().get_tokenizer()
    
    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        return self.ctx.get_hf_processor(KimiAudioProcessor, **kwargs)
    
    def get_audio_tokenizer(self) -> Any:
        audio_tokenizer = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
        return audio_tokenizer.to(torch.cuda.current_device())
    
    def get_feature_extractor(
            self,
            *,
            sampling_rate: Optional[int] = None
    ) -> WhisperEncoder:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperEncoder)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # no limits on count of audio
        return {"audio": None}
    
    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        """每个音频项最多产生多少tokens"""
        # 基于whisper特征长度估算
        # 通常30秒音频 -> ~750 tokens
        return {"audio": 750}


# === Dummy Inputs Builder === #
class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        
        tokenizer = self.info.get_tokenizer()
        config = self.info.get_hf_config()
        
        # 使用Kimi-Audio的特殊tokens
        # 从config中获取特殊token ids
        media_begin = tokenizer.decode([config.kimia_media_begin])
        media_end = tokenizer.decode([config.kimia_media_end])
        media_pad = tokenizer.decode([config.kimia_media_pad])
        
        # 构建dummy文本
        dummy_segments = []
        for i in range(num_audios):
            # 模拟音频占位符
            # media_begin + 一些padding + media_end
            audio_placeholder = f"{media_begin}{media_pad * 100}{media_end}"
            dummy_segments.append(audio_placeholder)
        
        # 添加一些文本提示
        if num_audios > 0:
            return "Please transcribe the following audio: " + " ".join(dummy_segments)
        return "Hello, how can I help you?"
    
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        
        # 创建dummy音频数据
        # Whisper需要16kHz采样率的音频
        audio_duration = 3.0  # 3秒dummy音频
        sample_rate = 16000
        audio_len = int(audio_duration * sample_rate)
        
        return {
            "audio": self._get_dummy_audios(
                length=audio_len, 
                num_audios=num_audios
            )
        }


# === Multi-Modal Processor === #
class KimiAudioMultiModalProcessor(BaseMultiModalProcessor[KimiAudioProcessingInfo]):

    @property
    def whisper_processor(self, **kwargs):
        if self._whisper_processor is None:
            self._whisper_processor = WhisperEncoder()
        return self._whisper_processor
    
    @property
    def audio_tokenizer(self, **kwargs):
        if self._audio_tokenizer is None:
            self._audio_tokenizer = self.info.get_audio_tokenizer()
        return self._audio_tokenizer
    
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)
    
    def _process_audio_to_features(
        self,
        audio_data: np.ndarray,
        sampling_rate: int = 16000
    ) -> torch.Tensor:
        """将音频转换为Whisper特征"""
        # 使用Whisper处理器提取特征
        inputs = self.whisper_processor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        # 获取mel特征
        mel_features = inputs.input_features
        
        # 这里需要通过Whisper编码器获取特征
        # 在实际实现中，这会调用WhisperEncoder
        # 返回shape: (1, seq_len, feature_dim)
        return mel_features
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data.get("audios", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), return_tensors="pt")
        
        feature_extractor = self.info.get_feature_extractor()
        mm_kwargs = dict(
            **mm_kwargs,
            Sampling_rate=feature_extractor.sampling_rate,
        )
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
    
    def _create_continuous_mask(
        self, 
        input_ids: torch.Tensor,
        config: KimiAudioConfig
    ) -> torch.Tensor:
        """创建标记音频位置的mask"""
        # 找到media_begin和media_end之间的位置
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        media_begin_indices = (input_ids == config.kimia_media_begin).nonzero(as_tuple=True)[0]
        media_end_indices = (input_ids == config.kimia_media_end).nonzero(as_tuple=True)[0]
        
        for begin, end in zip(media_begin_indices, media_end_indices):
            mask[begin+1:end] = True
        
        return mask.unsqueeze(0)  # Add batch dimension
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_nums = hf_inputs.get("audio", {}).get("nums", [])
        if audio_nums > self.info.get_supported_mm_limits():
            raise ValueError(
                f"Audio count {audio_nums} exceeds the supported limit "
                f"{self.info.get_supported_mm_limits()}"
            )
        return dict(
            whisper_input_feature=MultiModalFieldConfig.batched("audio"),
            is_continuous_mask=MultiModalFieldConfig.batched("audio"),
            text_input_ids=MultiModalFieldConfig.batched("audio"),
        )
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # 获取特殊token
        media_begin_token = tokenizer.decode([vocab.kimia_media_begin])
        media_pad_token = tokenizer.decode([vocab.kimia_media_pad])
        media_end_token = tokenizer.decode([vocab.kimia_media_end])
        
        def get_replacement_kimi_audio(item_idx: int):
            # 获取这个音频的特征长度
            whisper_features = out_mm_kwargs.get("whisper_input_feature", [])
            if item_idx < len(whisper_features):
                feature_len = whisper_features[item_idx].shape[1]
            else:
                feature_len = 100  # 默认长度
            
            # 创建替换序列
            # media_begin + feature_len个media_pad + media_end
            replacement_ids = (
                [vocab.kimia_media_begin] + 
                [vocab.kimia_media_pad] * feature_len + 
                [vocab.kimia_media_end]
            )
            
            return PromptUpdateDetails.select_token_id(
                replacement_ids,
                # 使用media_pad作为embedding token
                embed_token_id=vocab.kimia_media_pad,
            )
        
        # 查找prompt中的音频占位符并替换
        return [
            PromptReplacement(
                modality="audio",
                target=f"{media_begin_token}.*?{media_end_token}",  # 正则匹配
                replacement=get_replacement_kimi_audio,
            )
        ]


# === Main Model Class === #
@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder
)
class KimiAudioForConditionalGeneration(nn.Module, SupportsMultiModal):
    
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_media_end|>"
        
        raise ValueError("Only audio modality is supported")
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiAudioConfig = model_config.hf_config
        self.config = config
        
        self.audio_tower = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
        encoder_path = os.path.join(model_config.model, "whisper-large-v3")
        mel_batch_size = getattr(config, "mel_batch_size", 20)
        self.multi_modal_projector = WhisperEncoder(
            encoder_path,
            mel_batch_size=mel_batch_size,
        )
        self.multi_modal_model = MoonshotKimiaForCausalLM(
            config=config,
            prefix=maybe_prefix(prefix, "multi_modal_model"),
        )
        
        self.lm_head = self.multi_modal_model.lm_head
        self.mimo_output = self.multi_modal_model.mimo_output
        
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            self.vocab_size,
            logit_scale
        )
        
        # self.tp_rank = get_tensor_model_parallel_rank()
        # self.tp_world_size = get_tensor_model_parallel_world_size()
        
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
    
    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)
    
    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[KimiAudioInputs]:
        whisper_input_feature = kwargs.pop('whisper_input_feature', None)
        is_continuous_mask = kwargs.pop('is_continuous_mask', None)
        audio_input_ids = kwargs.pop('audio_input_ids', None)

        if whisper_input_feature is None:
            return None
        
        whisper_input_feature = self._validate_and_reshape_mm_tensor(
            whisper_input_feature, 'whisper_input_feature'
        )
        
        if is_continuous_mask is not None:
            is_continuous_mask = self._validate_and_reshape_mm_tensor(
                is_continuous_mask, 'is_continuous_mask'
            )
        else:
            # Create a default continuous mask
            batch_size, seq_len = whisper_input_feature.shape[:2]
            is_continuous_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        return KimiAudioInputs(
            whisper_input_feature=whisper_input_feature,
            is_continuous_mask=is_continuous_mask,
            audio_input_ids=audio_input_ids,
        )
    
    def _process_audio_input(self, audio_input: KimiAudioInputs) -> torch.Tensor:
        whisper_input_feature = audio_input["whisper_input_feature"]
        is_continuous_mask = audio_input["is_continuous_mask"]
        
        # Step 1: Process through Kimi-Audio's Whisper encoder
        # Input: raw audio features
        # Output: reshaped features (B, T//4, D*4)
        continuous_feature = self.audio_tower.tokenize_waveform(whisper_input_feature)
        continuous_feature = continuous_feature.reshape(
            continuous_feature.shape[0],
            int(continuous_feature.shape[1] // 4),
            continuous_feature.shape[2] * 4,
        )
        return continuous_feature
    
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model
    
    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        
        processed_features = self._process_audio_input(audio_input)
        return processed_features
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            # Use Kimi-Audio's media begin token ID for merging
            kimia_media_begin_id = getattr(self.config, 'kimia_media_begin', 151661)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                kimia_media_begin_id
            )
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            audio_input = self._parse_and_validate_audio_input(**kwargs)
            if audio_input is None:
                inputs_embeds = None
            else:
                multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
        
        outputs = self.multi_modal_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        text_logits, audio_logits = outputs.logits

        return CausalLMOutputWithPast(
            loss=None,
            logits=text_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        **kwargs: object,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata, **kwargs)
        return logits
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)