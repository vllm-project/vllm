# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2026 BharatGen AI team. All rights reserved.
#
# This code has been modified to accommodate Param2MoE's GQA-based MoE architecture.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# limitations under the License.
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def _is_expert_bias_name(name: str) -> bool:
    """True when the weight is the MoE router's per-expert score bias."""
    return name.endswith(".mlp.gate.expert_bias")


def _zero_mean_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return t
    return t - t.mean()


def _rename_and_normalize_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Translate HuggingFace Param2MoE weight names to vLLM internal names
    and zero-mean the expert-bias tensor so the router stays balanced.

    Mapping table (HF → vLLM):
      model.word_embeddings.*              → model.embed_tokens.*
      *.attention.query_key_value.*        → *.self_attn.qkv_proj.*
      *.attention.dense.*                  → *.self_attn.o_proj.*
      *.attention.query_layernorm.*        → *.self_attn.q_layernorm.*
      *.attention.key_layernorm.*          → *.self_attn.k_layernorm.*
      *.mlp.gate.expert_bias               → *.mlp.gate.e_score_correction_bias
        (also zero-meant for load balance)
    """
    for name, w in weights:
        # Embedding table
        name = name.replace("model.word_embeddings.", "model.embed_tokens.")
        # Fused QKV projection  (HF: query_key_value → vLLM: qkv_proj)
        name = name.replace(".attention.query_key_value.", ".self_attn.qkv_proj.")
        # Output projection  (HF: dense → vLLM: o_proj)
        name = name.replace(".attention.dense.", ".self_attn.o_proj.")
        # Per-head query norm
        name = name.replace(".attention.query_layernorm.", ".self_attn.q_layernorm.")
        # Per-head key norm
        name = name.replace(".attention.key_layernorm.", ".self_attn.k_layernorm.")
        # Catch any remaining .attention. → .self_attn. prefixes
        # (e.g. future bias params on the projection layers)
        name = name.replace(".attention.", ".self_attn.")

        # Expert-score bias: rename + zero-mean
        if name.endswith(".mlp.gate.expert_bias"):
            name = name.replace(
                ".mlp.gate.expert_bias",
                ".mlp.gate.e_score_correction_bias",
            )
            w = _zero_mean_tensor(w)

        yield name, w


class Param2MoEAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) for Param2MoE.

    Notable differences from a vanilla GQA layer:
      * The checkpoint fuses Q, K, V into a single ``query_key_value`` weight.
        vLLM receives it already renamed to ``qkv_proj`` by the weight-name
        translator and splits it during ``load_weights``.
      * Optional per-head RMS norms on Q and K (``use_qk_norm=True``).
    """

    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.use_qk_norm: bool = getattr(config, "use_qk_norm", False)

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0, (
            f"num_attention_heads ({self.num_heads}) must be divisible "
            f"by tensor-parallel world size ({tp_size})."
        )
        assert self.num_kv_heads % tp_size == 0, (
            f"num_key_value_heads ({self.num_kv_heads}) must be divisible "
            f"by tensor-parallel world size ({tp_size})."
        )
        self.num_local_heads = self.num_heads // tp_size
        self.num_local_kv_heads = self.num_kv_heads // tp_size

        # Sizes after TP split (used in forward to split qkv output)
        self.q_size_local = self.num_local_heads * self.head_dim
        self.kv_size_local = self.num_local_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=getattr(config, "use_qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=getattr(config, "use_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if self.use_qk_norm:
            self.q_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # `partial_rotary_factor` defaults to 1.0 (full RoPE) if not in config
        partial_rotary_factor: float = getattr(config, "partial_rotary_factor", 1.0)
        rope_dim = int(self.head_dim * partial_rotary_factor)

        rope_parameters: dict = {
            "rope_type": "default",
            "base": config.rope_theta,
        }
        if config.rope_scaling is not None:
            rope_parameters.update(config.rope_scaling)
            # Normalise key: some checkpoints use "type", vLLM wants "rope_type"
            if "type" in rope_parameters and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters.pop("type")

        self.rotary_emb = get_rope(
            rope_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Fused QKV projection → split into local Q / K / V
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size_local, self.kv_size_local, self.kv_size_local],
            dim=-1,
        )

        # 2. Optional per-head QK norms
        #    Reshape to (T, num_local_heads, head_dim), norm, reshape back.
        if self.use_qk_norm:
            T = q.shape[0]
            q = self.q_layernorm(q.view(T, self.num_local_heads, self.head_dim)).view(
                T, self.q_size_local
            )
            k = self.k_layernorm(
                k.view(T, self.num_local_kv_heads, self.head_dim)
            ).view(T, self.kv_size_local)

        # 3. Rotary position embeddings
        q, k = self.rotary_emb(positions, q, k)

        # 4. Paged attention
        attn_output = self.attn(q, k, v)

        # 5. Output projection
        output, _ = self.o_proj(attn_output)
        return output


class Param2MoEMLP(nn.Module):
    """SwiGLU feed-forward block used for dense layers."""

    def __init__(
        self,
        intermediate_size: int,
        config,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Param2MoEMoEBlock(nn.Module):
    """
    Mixture-of-Experts block for Param2MoE.

    Routing:
      * Sigmoid scoring  (config.score_function = "sigmoid")
      * Grouped top-k   (n_group, topk_group)
      * Per-expert bias  (gate.expert_bias → e_score_correction_bias)
      * routed_scaling_factor normalisation

    One set of shared (always-active) experts is added on top.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size

        self.num_experts: int = config.num_experts
        self.top_k: int = config.num_experts_per_tok
        self.routed_scaling_factor: float = getattr(
            config, "routed_scaling_factor", 1.0
        )

        self.n_group: int | None = getattr(config, "n_group", None)
        self.topk_group: int | None = getattr(config, "topk_group", None)
        self.use_grouped_topk: bool = (
            self.n_group is not None and self.topk_group is not None
        )

        self.norm_expert_prob: bool = getattr(config, "norm_topk_prob", True)
        self.score_function: str = getattr(config, "score_function", "sigmoid")

        self.gate = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
        )

        if getattr(config, "moe_router_enable_expert_bias", True):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None  # type: ignore[assignment]

        self.num_shared_experts: int = getattr(config, "num_shared_experts", 1)
        if self.num_shared_experts > 0:
            # If moe_shared_expert_intermediate_size is present in the config
            # it already encodes the TOTAL intermediate size across all shared
            # experts (i.e. it equals moe_intermediate_size * num_shared_experts).
            # Do NOT multiply again.  Fall back to computing the product only
            # when the dedicated field is absent.
            if (
                hasattr(config, "moe_shared_expert_intermediate_size")
                and config.moe_shared_expert_intermediate_size is not None
            ):
                shared_int: int = config.moe_shared_expert_intermediate_size
            else:
                shared_int = config.moe_intermediate_size * self.num_shared_experts
            self.shared_experts = Param2MoEMLP(
                intermediate_size=shared_int,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None  # type: ignore[assignment]

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=self.norm_expert_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.score_function,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
            use_grouped_topk=self.use_grouped_topk,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def maybe_get_fused_moe(self) -> SharedFusedMoE:
        return self.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router: both input and weight must be float32 for numerical
        # stability (mirrors the original Param2MoEGate behaviour).
        # The gate nn.Linear weight lives in the model dtype (bfloat16),
        # so we must cast both explicitly via F.linear instead of calling
        # self.gate() which would hit a dtype mismatch.
        router_logits = F.linear(
            hidden_states.float(),
            self.gate.weight.float(),
        ).to(hidden_states.dtype)

        final_hidden = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        if self.shared_experts is not None:
            shared_output, expert_output = final_hidden
        else:
            shared_output, expert_output = None, final_hidden

        if shared_output is not None:
            expert_output = expert_output + shared_output

        if self.tp_size > 1:
            expert_output = self.experts.maybe_all_reduce_tensor_model_parallel(
                expert_output
            )

        return expert_output.view(num_tokens, hidden_dim)


class Param2MoEDecoderLayer(nn.Module):
    """
    Single transformer decoder block.

    Dense for the first ``first_k_dense_replace`` layers; MoE thereafter.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        hidden_size = config.hidden_size
        # Derive the layer index from the prefix (e.g. "model.layers.3")
        layer_idx = int(prefix.split(".")[-1])

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Param2MoEAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        first_k_dense: int = getattr(config, "first_k_dense_replace", 1)
        is_moe_layer = config.num_experts is not None and layer_idx >= first_k_dense

        if is_moe_layer:
            self.mlp = Param2MoEMoEBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Param2MoEMLP(  # type: ignore[assignment]
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm + attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Pre-norm + MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Param2MoEModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.tie_word_embeddings: bool = getattr(config, "tie_word_embeddings", False)

        # Embedding  (HF name: word_embeddings → vLLM name: embed_tokens)
        if get_pp_group().is_first_rank or (
            self.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Param2MoEDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(hidden_states, positions, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """
        Custom weight loader for the inner Param2MoEModel.

        Receives weights that have already been renamed/normalised by the
        outer model and whose ``model.`` prefix has been stripped by
        ``AutoWeightsLoader``.  Handles:
          1. Fused QKV split (query_key_value → qkv_proj q/k/v shards).
          2. gate_proj + up_proj → gate_up_proj stacking (dense + shared-exp).
          3. Routed-expert weights via the fused-MoE mapping.
          4. All remaining weights via their default loader.
        """
        config = self.config
        num_heads: int = config.num_attention_heads
        num_kv_heads: int = config.num_key_value_heads
        head_dim: int = config.head_dim or (config.hidden_size // num_heads)
        q_split = num_heads * head_dim
        kv_split = num_kv_heads * head_dim

        stacked_params_mapping = [
            # (vllm_param_name, ckpt_weight_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            # ------------------------------------------------------------------
            # 1. Fused QKV: split into q / k / v shards for QKVParallelLinear
            # ------------------------------------------------------------------
            if name.endswith(".self_attn.qkv_proj.weight"):
                if name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                q_w = loaded_weight[:q_split, :]
                k_w = loaded_weight[q_split : q_split + kv_split, :]
                v_w = loaded_weight[q_split + kv_split :, :]
                weight_loader(param, q_w, "q")
                weight_loader(param, k_w, "k")
                weight_loader(param, v_w, "v")
                loaded_params.add(name)
                continue

            # ------------------------------------------------------------------
            # 2. gate_proj / up_proj → gate_up_proj (dense MLP + shared-exp.)
            # ------------------------------------------------------------------
            matched_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:  # routed experts handled below
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                if new_name not in params_dict:
                    continue
                if is_pp_missing_parameter(new_name, self):
                    continue

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(new_name)
                matched_stacked = True
                break

            if matched_stacked:
                continue

            # ------------------------------------------------------------------
            # 3. Routed expert weights → fused-MoE kernel layout
            # ------------------------------------------------------------------
            matched_expert = False
            for (
                param_name,
                weight_name,
                expert_id,
                shard_id,
            ) in expert_params_mapping:
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(new_name, self):
                    continue
                if new_name not in params_dict:
                    continue

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(new_name)
                matched_expert = True
                break

            if matched_expert:
                continue

            # ------------------------------------------------------------------
            # 4. All other weights: direct load (layernorms, embed_tokens, …)
            # ------------------------------------------------------------------
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(param, loaded_weight)
            except Exception as e:
                raise RuntimeError(
                    f"[param2moe] Failed to load weight '{name}' "
                    f"with shape {tuple(loaded_weight.shape)} "
                    f"into param type {type(param).__name__}: {e}"
                ) from e
            loaded_params.add(name)

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )


class Param2MoEMixtureOfExperts(MixtureOfExperts):
    """Implements the vLLM MixtureOfExperts protocol for Param2MoE."""

    expert_weights: list[torch.Tensor]

    def extract_moe_parameters(self, example_moe: Param2MoEMoEBlock | None) -> None:
        if example_moe is None:
            raise RuntimeError(
                "No Param2MoEMoEBlock found in model.layers. "
                "Check first_k_dense_replace and num_experts in config."
            )
        self.num_logical_experts = example_moe.num_experts
        self.num_routed_experts = example_moe.num_experts
        self.num_shared_experts = example_moe.num_shared_experts

        self.num_physical_experts = self.num_logical_experts
        self.num_local_physical_experts = self.num_logical_experts
        self.num_redundant_experts = 0

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts

        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts

            fused = moe.experts
            if hasattr(fused, "n_local_physical_experts"):
                fused.n_local_physical_experts = num_local_physical_experts
            if hasattr(fused, "n_physical_experts"):
                fused.n_physical_experts = num_physical_experts
            if hasattr(fused, "n_redundant_experts"):
                fused.n_redundant_experts = self.num_redundant_experts
            if hasattr(fused, "update_expert_map"):
                fused.update_expert_map()

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.expert_weights.clear()
        for layer_idx, layer in enumerate(self.moe_layers):
            if hasattr(layer, "get_expert_weights"):
                self.expert_weights.append(layer.get_expert_weights())
            if hasattr(layer, "set_eplb_state"):
                layer.set_eplb_state(
                    moe_layer_idx=layer_idx,
                    expert_load_view=expert_load_view,
                    logical_to_physical_map=logical_to_physical_map,
                    logical_replica_count=logical_replica_count,
                )


class Param2MoEForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, Param2MoEMixtureOfExperts
):
    """
    vLLM-native Param2MoE CausalLM.

    Uses Grouped-Query Attention (GQA) with a Sigmoid-scored,
    grouped-topk Mixture-of-Experts MLP.
    """

    # LoRA packed-module mapping. The fused gate_up_proj handles
    # gate_proj and up_proj from the checkpoint.
    packed_modules_mapping = {
        "qkv_proj": ["query_key_value"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Modules eligible for LoRA adaptation.
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]

    # Embedding layers and their weight-tying counterparts.
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Modules that need vocab-size padding for LoRA.
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = Param2MoEModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.tie_word_embeddings: bool = getattr(config, "tie_word_embeddings", False)
        if get_pp_group().is_last_rank:
            if self.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()
            self.logits_processor = None  # type: ignore[assignment]

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.expert_weights: list[torch.Tensor] = []
        self.num_moe_layers: int = 0
        self.moe_layers: list = []
        self.moe_mlp_layers: list = []

        example_moe: Param2MoEMoEBlock | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Param2MoEMoEBlock):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
                self.num_moe_layers += 1

        if self.config.num_experts is not None:
            self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if not get_pp_group().is_last_rank:
            return None
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(_rename_and_normalize_weights(weights))
