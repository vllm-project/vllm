# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZEDA-GLM-4.7-Flash-Dynamic (Glm4MoeLitePlusPlus) model.

Thin override of vLLM's Glm4MoeLite that adds Zero-Compute Expert (ZCE)
support by:
- recreating the router gate to emit logits for all experts
  (n_routed_experts + sum(zce_nums) = 64 + 32 = 96),
- syncing the 96-dim e_score_correction_bias onto the Ascend MoE runner and
  its routed_experts (GLM uses sigmoid + bias + group topk routing),
- setting zero_expert_num / zero_expert_type on routed_experts so that
  vLLM-Ascend's AscendUnquantizedFusedMoEMethod.apply() invokes
  zero_experts_compute() to remap zero-expert slots (>= n_routed_experts) to
  expert 0 with weight 0 (graph-friendly, no per-expert Python loop).

Reuses the core FusedMoE / weight loading / forward; no eager dispatch.
Reference: vllm-ascend PR #14905 (Qwen3 MoE++).
"""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.glm4_moe_lite import (
    Glm4MoeLite,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteModel,
)
from vllm.platforms import current_platform

from .utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class Glm4MoeLitePlusPlusDecoderLayer(Glm4MoeLiteDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            topk_indices_buffer=topk_indices_buffer,
        )
        if not isinstance(self.mlp, Glm4MoeLite):
            return

        hf_config = config if config is not None else vllm_config.model_config.hf_config
        zce_nums = list(getattr(hf_config, "zce_nums", []) or [])
        zce_types = list(getattr(hf_config, "zce_types", []) or [])
        total_num_experts = hf_config.n_routed_experts + sum(zce_nums)
        if total_num_experts <= hf_config.n_routed_experts or not zce_nums:
            return

        new_gate = nn.Linear(
            hf_config.hidden_size,
            total_num_experts,
            bias=False,
            dtype=torch.float32,
        )
        new_bias = nn.Parameter(
            torch.zeros(total_num_experts, dtype=torch.float32)
        )
        new_gate.e_score_correction_bias = new_bias
        self.mlp.gate = new_gate

        experts = self.mlp.experts
        experts.e_score_correction_bias = new_bias
        if hasattr(experts, "routed_experts"):
            experts.routed_experts.e_score_correction_bias = new_bias
            experts.routed_experts.zero_expert_num = int(sum(zce_nums))
            experts.routed_experts.zero_expert_type = (
                zce_types[0] if zce_types else "zero"
            )
        if hasattr(experts, "router") and experts.router is not None:
            experts.router.e_score_correction_bias = new_bias


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Glm4MoeLitePlusPlusModel(Glm4MoeLiteModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.vocab_size = config.vocab_size
        is_v32 = hasattr(config, "index_topk")
        if is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

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
            lambda prefix: Glm4MoeLitePlusPlusDecoderLayer(
                vllm_config=vllm_config,
                config=config,
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


class Glm4MoeLitePlusPlusForCausalLM(Glm4MoeLiteForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4MoeLiteForCausalLM, self).__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        self.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        if self.use_mha:
            self.packed_modules_mapping["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]

        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = Glm4MoeLitePlusPlusModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.num_moe_layers = (
            self.config.num_hidden_layers - self.config.first_k_dense_replace
        )
        self.set_moe_parameters()
