# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import torch
import torch.nn as nn
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import RefSlidingWindowAttention
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processors.unlimited_ocr import UnlimitedOCRHFProcessor

from .deepseek_ocr import (
    DeepseekOCRDummyInputsBuilder,
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
)
from .deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2ForCausalLM, DeepseekV2Model


class UnlimitedOCRProcessingInfo(DeepseekOCRProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(UnlimitedOCRHFProcessor, **kwargs)


class UnlimitedOCRAttention(nn.Module):
    """Normal MHA implementation used by Deepseek v1."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        attn_cache_config = copy.copy(cache_config)
        if attn_cache_config is not None:
            attn_cache_config.sliding_window = None

        self.attn = RefSlidingWindowAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            sliding_window=config.sliding_window,
            num_kv_heads=self.num_kv_heads,
            cache_config=attn_cache_config,
            quant_config=quant_config,
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
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class UnlimitedOCRDecoderLayer(DeepseekV2DecoderLayer):
    attn_cls = UnlimitedOCRAttention


class UnlimitedOCRModel(DeepseekV2Model):
    layer_cls = UnlimitedOCRDecoderLayer


class UnlimitedOCRLanguageForCausalLM(DeepseekV2ForCausalLM):
    model_cls = UnlimitedOCRModel


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    def _init_language_model(
        self, vllm_config: VllmConfig, hf_config: PretrainedConfig, prefix: str
    ):
        return UnlimitedOCRLanguageForCausalLM(
            vllm_config=vllm_config.with_hf_config(hf_config),
            prefix=prefix,
        )
