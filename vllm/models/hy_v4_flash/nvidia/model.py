# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash backbone using the public HYV4 checkpoint contract."""

from itertools import islice

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.model_executor.models.utils import WeightsMapper, copy_pp_buffers
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.common.ops.sequence_parallel import (
    is_sequence_parallel_active,
    sp_all_gather,
    sp_shard,
)
from vllm.models.hy_v4.common.mrope import set_token_rope
from vllm.models.hy_v4.common.sequence_parallel import IHC_SP_MIN_TOKENS
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
    IHCCarry,
)
from vllm.models.hy_v4.weight_utils import HYV4_MLP_MAPPER
from vllm.models.hy_v4_flash.common.model import (
    HYV4FlashForCausalLMMixin,
    HYV4FlashModelMixin,
)
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .attention import HYV4FlashAttention
from .moe import HYV4FlashFeedForward, HYV4FlashMoE


def validate_platform() -> None:
    if not current_platform.is_cuda():
        raise NotImplementedError("HYV4 Flash requires NVIDIA CUDA")


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
        self.defer_reduction = ihc_sequence_parallel and self.enable_ihc
        self.self_attn = HYV4FlashAttention(
            vllm_config,
            config,
            self.layer_idx,
            topk_indices_buffer,
            f"{prefix}.self_attn",
            defer_reduction=self.defer_reduction,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        dense = (config.mlp_layer_types or ())[self.layer_idx] == "dense"
        mlp_cls = HYV4FlashFeedForward if dense else HYV4FlashMoE
        self.mlp = mlp_cls(
            config,
            quant_config,
            f"{prefix}.mlp",
            defer_reduction=self.defer_reduction,
        )
        self.block_type = "feedforward" if dense else "moe"
        self.hc_attn_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )
        self.init_ihc_boundaries()

    def _deferred_reduce(
        self,
        hidden_states: torch.Tensor,
        sequence_parallel: bool,
        sub_block: nn.Module,
    ) -> torch.Tensor:
        if not self.defer_reduction:
            return hidden_states
        norm = sub_block.post_reduce_norm
        if (
            not sequence_parallel
            and norm is not None
            and sub_block.fuse_post_reduce_norm
        ):
            return fused_allreduce_rms_norm(hidden_states, None, norm)[0]
        hidden_states = super()._deferred_reduce(
            hidden_states, sequence_parallel, sub_block
        )
        return norm(hidden_states) if norm is not None else hidden_states


class HYV4FlashModel(HYV4FlashModelMixin, HYV4Model):
    decoder_layer_cls = HYV4FlashDecoderLayer
    supports_ihc_sequence_parallel = True

    def _allows_ihc_sequence_parallel(self, vllm_config: VllmConfig) -> bool:
        # DFly and other speculative paths remain unverified; gate before
        # construction so their original reductions and fusions are preserved.
        return (
            super()._allows_ihc_sequence_parallel(vllm_config)
            and vllm_config.speculative_config is None
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.init_engram(vllm_config, prefix)
        local_layers = list(islice(self.layers, self.start_layer, self.end_layer))
        # SWA and sparse layers rotate with different thetas, and each rotary
        # module holds its own per-forward rows for the fused RoPE kernels.
        self.rotary_embs = list(
            {
                id(layer.self_attn.rotary_emb): layer.self_attn.rotary_emb
                for layer in local_layers
            }.values()
        )
        # Engram hashing reads the batch layout from this stage's first sparse
        # layer.
        sparse = [
            layer for layer in local_layers if not layer.self_attn.is_window_layer
        ]
        self.engram_metadata_layer = (
            (sparse or local_layers)[0].self_attn.mla_attn.layer_name
            if local_layers
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden = intermediate_tensors["hidden_states"]
        if self.engram_hash is None:
            for rotary_emb in self.rotary_embs[1:]:
                set_token_rope(rotary_emb, positions, hidden)
            return super().forward(input_ids, positions, intermediate_tensors, hidden)
        if input_ids is None:
            raise ValueError("Flash Engram requires original input token IDs")
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            copy_pp_buffers(intermediate_tensors, self.pp_buffers)
        for rotary_emb in self.rotary_embs:
            set_token_rope(rotary_emb, positions, hidden)
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
            self.engram_metadata_layer,
            hashes,
        )
        hash_indices = {
            layer_id - 1: index
            for index, layer_id in enumerate(self.config.engram_layer_ids)
        }
        full_num_tokens = positions.shape[-1]
        sequence_parallel = is_sequence_parallel_active(
            self.ihc_sequence_parallel, full_num_tokens, IHC_SP_MIN_TOKENS
        )
        if sequence_parallel:
            hidden = sp_shard(hidden)
        fused = self.uses_fused_ihc()
        carry: IHCCarry | None = None
        for index in range(self.start_layer, self.end_layer):
            layer = self.layers[index]
            if index in hash_indices:
                if carry is not None:
                    hidden = self.layers[index - 1].hc_mlp_layer.post(*carry)
                    carry = None
                if sequence_parallel:
                    # Engram's hashes and head-partial reduction use global rows.
                    hidden = sp_all_gather(hidden)[:full_num_tokens]
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                hidden = layer.engram_layer(hidden, hashes[hash_indices[index]])
                if sequence_parallel:
                    hidden = sp_shard(hidden)
            if fused:
                carry = layer.forward_ihc_fused(positions, hidden, carry)
            else:
                hidden, _ = layer(positions, hidden, None)
        is_last_rank = get_pp_group().is_last_rank
        if carry is not None:
            output = (
                self.hc_head.fused_post(*carry, self.norm) if is_last_rank else None
            )
            if output is not None:
                if sequence_parallel:
                    output = sp_all_gather(output)[:full_num_tokens]
                return output
            hidden = layer.hc_mlp_layer.post(*carry)
        if not is_last_rank:
            buffers = {k: v[: positions.shape[-1]] for k, v in self.pp_buffers.items()}
            return IntermediateTensors({"hidden_states": hidden.flatten(1), **buffers})
        hidden = self.norm(self.hc_head(hidden))
        if sequence_parallel:
            hidden = sp_all_gather(hidden)[:full_num_tokens]
        return hidden


class HYV4FlashForCausalLM(HYV4FlashForCausalLMMixin, HYV4ForCausalLM, SupportsMRoPE):
    model_cls = HYV4FlashModel
    supports_lora = False  # type: ignore[assignment]
    requires_raw_input_tokens = True
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    hf_to_vllm_mapper = HYV4_MLP_MAPPER | WeightsMapper(
        orig_to_new_substr={".compressed_indexer.": ".indexer."},
        orig_to_new_stacked={
            ".self_attn.q_a_proj.": (".self_attn.fused_qkv_a_proj.", 0),
            ".self_attn.kv_a_proj_with_mqa.": (".self_attn.fused_qkv_a_proj.", 1),
            ".indexer.key_value_projection.": (".indexer.key_projection.", 0),
            ".indexer.key_gate_projection.": (".indexer.key_projection.", 1),
        },
    )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        # Published configs use M-RoPE; text tokens share one position per row.
        return torch.arange(len(input_tokens)).repeat(3, 1), 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_platform()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.init_weight_loading(vllm_config)
