# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, replace
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig

from .gigachat4 import GigaChat4Attention, GigaChat4GatedNorm, GigaChat4MLP
from .qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
    DFlashGroupedConv,
)
from .utils import WeightsMapper, maybe_prefix


class GigaChat4DFlash2Attention(GigaChat4Attention):
    causal = False

    @property
    def attn(self):
        return self.mla_attn

    def project_context_kv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.fused_qkv_a_proj.weight[self.q_lora_rank :]
        latent = F.linear(hidden_states, weight)
        kv_c, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)
        _, k_pe = self.rotary_emb(positions, torch.empty_like(k_pe), k_pe)
        return kv_c, k_pe


class GigaChat4DFlash2DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.branch_multiplier != 1.0 or config.linear_attn_output_mult != 1.0:
            raise ValueError(
                "GigaChat4 DFlash2 requires unit branch/output multipliers"
            )
        if quant_config is not None:
            raise ValueError("GigaChat4 DFlash2 requires unquantized draft weights")
        if config.index_topk:
            raise ValueError("GigaChat4 DFlash2 draft attention does not use DSA")
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_vllm_config = replace(
            vllm_config,
            model_config=speculative_config.draft_model_config,
            quant_config=quant_config,
            parallel_config=replace(
                vllm_config.parallel_config,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
            ),
        )
        self.self_attn = GigaChat4DFlash2Attention(
            draft_vllm_config,
            None,
            None,
            maybe_prefix(prefix, "self_attn"),
            non_causal_multi_token_decode=True,
        )
        self.mlp = GigaChat4MLP(
            config, config.intermediate_size, quant_config, maybe_prefix(prefix, "mlp")
        )
        self.input_layernorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.post_attention_layernorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.post_self_attn_layernorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.post_feedforward_layernorm = GigaChat4GatedNorm(config, config.hidden_size)
        draft = config.dflash_config
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft["conv_kernel_size"]),
            group_size=int(draft["conv_group_size"]),
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class GigaChat4DFlash2Model(DFlash2Qwen3Model):
    decoder_layer_cls = GigaChat4DFlash2DecoderLayer
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        spec = vllm_config.speculative_config
        assert spec is not None
        draft = spec.draft_model_config.hf_config.dflash_config
        if draft.get("attention_mode") != "mla":
            raise ValueError("GigaChat4 DFlash2 requires MLA draft attention")
        if draft.get("shift_label", True) is not True:
            raise ValueError("GigaChat4 DFlash2 requires shift_label=True")
        if draft.get("context_input_norm", "none") != "none":
            raise ValueError("GigaChat4 DFlash2 requires context_input_norm='none'")
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        self.norm = GigaChat4GatedNorm(self.config, self.config.hidden_size)
        self.hidden_norm = GigaChat4GatedNorm(self.config, self.config.hidden_size)
        self.block_size = self.layers[0].attention_conv.block_size

    def context_kv_modules(self) -> list[nn.Module]:
        modules = [self.fc, self.hidden_norm]
        for layer in self.layers:
            attn = layer.self_attn
            modules.extend(
                [attn.fused_qkv_a_proj, attn.kv_a_layernorm, attn.rotary_emb, attn.attn]
            )
        return modules

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        context_states = self.hidden_norm(context_states)
        for i, layer in enumerate(self.layers):
            kv_c, k_pe = layer.self_attn.project_context_kv(
                context_positions, context_states
            )
            slot_mapping = (
                context_slot_mapping[i]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = layer.self_attn.attn
            attn.update_kv_cache(
                kv_c,
                k_pe,
                attn.kv_cache,
                slot_mapping,
                None,
                attn.kv_cache_dtype,
                attn._k_scale,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if input_embeds is None else input_embeds
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        # Giga row i predicts i+1; the shared proposer samples rows 1..K.
        blocks = hidden_states.view(-1, self.block_size, self.config.hidden_size)
        blocks = torch.roll(blocks, shifts=1, dims=1)
        return blocks.reshape_as(hidden_states)


class GigaChat4DFlash2ForCausalLM(DFlash2Qwen3ForCausalLM):
    speculator_config_class = DFlash2SpeculatorConfig
    model_cls = GigaChat4DFlash2Model

    def process_weights_after_loading(self) -> None:
        # MLA consumes the loaded projections directly, without fused MHA buffers.
        pass
