# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 MTP model for SM100 (Blackwell)."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP as DeepSeekMTPBase
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMultiTokenPredictor as DeepSeekMultiTokenPredictorBase,
)
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMultiTokenPredictorLayer as DeepSeekMultiTokenPredictorLayerBase,
)
from vllm.model_executor.models.deepseek_mtp import SharedHead as SharedHeadBase
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .layer import DeepseekV32DecoderLayer
from .model import remap_weight_name

logger = init_logger(__name__)


class SharedHead(SharedHeadBase):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.norm.weight, self.norm.variance_epsilon)


class DeepSeekMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayerBase):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        nn.Module.__init__(self)

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV32DecoderLayer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=int(prefix.rsplit(".", 1)[-1]),
            topk_indices_buffer=topk_indices_buffer,
            prefix=prefix,
        )


class DeepSeekMultiTokenPredictor(DeepSeekMultiTokenPredictorBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)


@support_torch_compile
class DeepSeekMTP(DeepSeekMTPBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        assert hasattr(self.config, "index_topk")
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
            logger.info("Using bfloat16 kv-cache for DeepSeekV3.2")
        self.model = DeepSeekMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()
        # Keep the original loader from applying the fused FP4 indexer remap.
        self.is_fp4_ckpt = False

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            layer = layer.mtp_block
            assert isinstance(layer, DeepseekV32DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        del intermediate_tensors
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = super().load_weights(weights)
        for layer in self.model.layers.values():
            layer.mtp_block.fuse_indexer_weights()
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        name = super()._rewrite_spec_layer_name(spec_layer, name)
        return remap_weight_name(name)


@torch.compile
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(mean_sq + eps)
    x = x * rrms
    x = x * w.to(torch.float32)
    return x.to(orig_dtype)
