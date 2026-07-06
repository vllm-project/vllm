# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MixtureOfExperts,
    DeepseekV2MoE,
    _try_load_fp8_indexer_wk,
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import (
    get_pp_missing_layer_names,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .kernels import fused_eh_norm
from .fused_ops import fused_allreduce_rms_norm
from .model import DeepseekV32DecoderLayer


class DeepseekV32MultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        # bf16 skinny GEMM for the eh_proj, B300+B200 measured: GLM-5.2
        # (6144, 12288) wins at M <= 3, DSv3.2 (7168, 14336) at M <= 2;
        # cuBLAS holds above (the 151/205MB weights stream at ~5.7TB/s).
        eh_dispatch = {(6144, 12288): 3, (7168, 14336): 2}
        eh_weight = getattr(self.eh_proj, "weight", None)
        self._eh_skinny_max = (
            eh_dispatch.get(tuple(eh_weight.shape), 0)
            if current_platform.is_device_capability_family(100)
            and hasattr(torch.ops._C, "bf16_skinny_gemm")
            and eh_weight is not None
            and eh_weight.dtype == torch.bfloat16
            else 0
        )

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
            vllm_config,
            prefix,
            config=config,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # Fused: zero pos-0 embeds + enorm(embeds) + hnorm(prev) + cat -> [N, 2H].
        eh_input = fused_eh_norm(
            positions,
            inputs_embeds,
            previous_hidden_states,
            self.enorm.weight,
            self.hnorm.weight,
            self.enorm.variance_epsilon,
        )
        if (
            eh_input.shape[0] <= self._eh_skinny_max
            and eh_input.dtype == torch.bfloat16
        ):
            hidden_states = ops.bf16_skinny_gemm(eh_input, self.eh_proj.weight)
        else:
            hidden_states = self.eh_proj(eh_input)
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        # mtp_block's MoE output is left un-reduced (skip_final_all_reduce);
        # fuse its all-reduce into the final norm like the main model does at
        # layer boundaries (norm(AR(h), residual), one flashinfer kernel).
        #
        # Recycle the POST-final-norm hidden into the next draft step. The
        # residual-add is fused into the final RMSNorm so it is computed
        # exactly once, and the result is returned for both tuple positions:
        # the draft-logits hidden (compute_logits applies the LM head only) and
        # the recycled previous_hidden_states. Recycling the pre-final-norm
        # hidden mismatches the draft model's hnorm and lowers MTP acceptance;
        # post-norm recycle matches deepseek_mtp.py (PR #45895). The tuple form
        # is understood by both the V2 speculator (isinstance-tuple check) and
        # the legacy proposer (model_returns_tuple is True for the
        # DeepSeekMTPModel architecture).
        hidden_states, _ = fused_allreduce_rms_norm(
            hidden_states, residual, self.shared_head.norm
        )
        return hidden_states, hidden_states


class DeepseekV32MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepseekV32MultiTokenPredictorLayer(
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

    def set_skip_topk(self, skip: bool):
        # index_share_for_mtp_iteration: step 0 computes top-k, steps 1+ reuse.
        for layer in self.layers.values():
            self_attn = getattr(layer.mtp_block, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "skip_topk"):
                self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather the top-k index rows at ``slot_ids`` to the front of the buffer."""
        num_slots = slot_ids.numel()
        for layer in self.layers.values():
            self_attn = getattr(layer.mtp_block, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "topk_indices_buffer"):
                topk_indices_buffer = self_attn.topk_indices_buffer
                topk_indices_buffer[:num_slots] = topk_indices_buffer[slot_ids]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        # hidden_states is already post-final-norm (produced in the layer
        # forward and recycled as-is); apply the LM head only, without a
        # second RMSNorm.
        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)


class DeepseekV32MTP(nn.Module, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekV32MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, DeepseekV2MoE):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
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
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
        )

        pp_missing_layer_names = get_pp_missing_layer_names(self)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        _pending_wk_fp8: dict = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            )
            name = self._rewrite_spec_layer_name(spec_layer, name)

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
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight
                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    is_expert_weight = False
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping  # type: ignore[assignment]
                        if weight_name not in chunk_name:
                            continue
                        is_expert_weight = True
                        name_mapped = chunk_name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            continue
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        name = maybe_remap_kv_scale_name(name, params_dict)  # type: ignore[assignment]
                        if name is None:
                            continue
                        if (
                            spec_layer != self.model.mtp_start_layer_idx
                            and ".layers" not in name
                        ):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint."
                )
        return loaded_params
