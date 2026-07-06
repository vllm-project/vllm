# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
    _try_load_fp8_indexer_wk,
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.sparse_utils import register_phys_shadow

from .attention import DeepseekV32Attention
from .fused_ops import fused_allreduce_rms_norm


class DeepseekV32DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.use_mha = False

        self.self_attn = DeepseekV32Attention(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            # Defer the MoE cross-rank all-reduce; it is fused into the next
            # layer's input_layernorm (or the final norm) via
            # fused_allreduce_rms_norm. self.mlp.experts is the MoERunner.
            self.mlp.experts.moe_config.skip_final_all_reduce = True
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=False,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First layer: hidden_states is the (already reduced) embedding.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # The previous layer's MLP/MoE output is left un-reduced; fuse its
            # all-reduce into this input_layernorm.
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        # self_attn's o_proj runs reduce_results=False; fuse its all-reduce with
        # the post-attention RMSNorm.
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        # MLP/MoE runs un-reduced; its all-reduce is fused into the next layer's
        # input_layernorm (or the model's final norm).
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV32Model(torch.nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        from vllm.platforms import current_platform

        self.device = current_platform.device_type

        self.vocab_size = config.vocab_size
        # DSA is always sparse (has index_topk); allocate the shared top-k
        # buffer the indexer writes and the sparse MLA backend reads.
        self.is_v32 = True
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
        register_phys_shadow(topk_indices_buffer)

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV32DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.aux_hidden_state_layers = tuple[int, ...]()
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # Last layer's MoE output is un-reduced; fuse its all-reduce into norm.
        hidden_states, _ = fused_allreduce_rms_norm(hidden_states, residual, self.norm)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # DSA-only: MLA (fused_qkv_a_proj) + the fused indexer wk/weights_proj +
        # routed experts. No MHA (qkv_proj) or ROCm shared-expert-fusion paths.
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

        pp_missing_layer_names = get_pp_missing_layer_names(self)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        _pending_wk_fp8: dict = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # MTP / nextn layers are loaded by the MTP model, not here.
            if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                continue
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Experts are handled below; skip here before the name rewrite.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping  # type: ignore[assignment]
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)  # type: ignore[assignment]
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    """DSA causal LM — DeepSeek V2/V3 orchestration with the DSA backbone.

    Serves DeepSeek V3.2 and any architecture reusing DSA (e.g. GLM-5.2).
    """

    model_cls = DeepseekV32Model

    def set_moe_parameters(self):
        # Same as the base, but keyed on the MoE block type rather than the
        # decoder-layer type (DeepseekV32DecoderLayer is a plain nn.Module).
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)
