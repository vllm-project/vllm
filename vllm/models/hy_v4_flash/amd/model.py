# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash backbone using the public HYV4 checkpoint contract (ROCm)."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.hy_v4.amd.hc import HYV4HCLayer
from vllm.models.hy_v4.amd.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
)
from vllm.models.hy_v4.weight_utils import HYV4_MLP_MAPPER
from vllm.models.hy_v4_flash.common.model import (
    HYV4FlashForCausalLMMixin,
    HYV4FlashModelMixin,
)
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .attention import HYV4FlashAttention
from .moe import HYV4FlashFeedForward, HYV4FlashMoE


def validate_platform() -> None:
    if not current_platform.is_rocm():
        raise NotImplementedError("The HYV4 Flash AMD implementation requires ROCm")


class HYV4FlashDecoderLayer(HYV4DecoderLayer):
    def __init__(
        self,
        config: HYV4FlashConfig,
        vllm_config: VllmConfig,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        ihc_sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.enable_ihc = config.enable_ihc
        assert not ihc_sequence_parallel
        self.defer_reduction = False
        self.self_attn = HYV4FlashAttention(
            vllm_config,
            config,
            self.layer_idx,
            topk_indices_buffer,
            f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        dense = (config.mlp_layer_types or ())[self.layer_idx] == "dense"
        mlp_cls = HYV4FlashFeedForward if dense else HYV4FlashMoE
        self.mlp = mlp_cls(config, quant_config, f"{prefix}.mlp")
        self.block_type = "feedforward" if dense else "moe"
        self.hc_attn_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )


class HYV4FlashModel(HYV4FlashModelMixin, HYV4Model):
    decoder_layer_cls = HYV4FlashDecoderLayer
    supports_ihc_sequence_parallel = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.init_engram(vllm_config, prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.engram_hash is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        if input_ids is None:
            raise ValueError("Flash Engram requires original input token IDs")
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        hashes = input_ids.new_empty(
            (
                len(self.config.engram_layer_ids),
                input_ids.numel(),
                self.engram_hash.offsets.shape[1],
            ),
            dtype=torch.int32,
        )
        self.engram_hash.prepare_hashes_with_output(
            input_ids,
            lookback_token_ids,
            self.layers[0].self_attn.mla_attn.layer_name,
            hashes,
        )
        self.prepare_engram_lookups(hashes, hidden.dtype)
        hash_indices = {
            layer_id - 1: index
            for index, layer_id in enumerate(self.config.engram_layer_ids)
        }
        for index, layer in enumerate(self.layers):
            if index in hash_indices:
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                hidden = layer.engram_layer(hidden, hashes[hash_indices[index]])
            hidden, _ = layer(positions, hidden, None)
        return self.norm(self.hc_head(hidden))


class HYV4FlashForCausalLM(HYV4FlashForCausalLMMixin, HYV4ForCausalLM):
    model_cls = HYV4FlashModel
    supports_lora = False  # type: ignore[assignment]
    supports_pp = False  # type: ignore[assignment]
    requires_raw_input_tokens = True
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    hf_to_vllm_mapper = HYV4_MLP_MAPPER | WeightsMapper(
        orig_to_new_substr={".compressed_indexer.": ".indexer."},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_platform()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.init_weight_loading(vllm_config)
