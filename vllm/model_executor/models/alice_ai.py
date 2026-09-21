# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production-oriented AliceAI using upstream KDA and AttnRes kernels."""

from collections.abc import Iterable
from copy import copy
from itertools import islice
from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextModel,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.utils import WeightsMapper, extract_layer_index
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
)
from vllm.models.kimi_k3.nvidia.ops import attn_res as upstream_attn_res
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig

_SPLIT_KDA_CONV_SHARDS = {
    ".linear_attn.q_conv1d": (".linear_attn.conv1d", 0),
    ".linear_attn.k_conv1d": (".linear_attn.conv1d", 1),
    ".linear_attn.v_conv1d": (".linear_attn.conv1d", 2),
}


def _load_split_kda_conv_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    params: dict[str, nn.Parameter],
    loaded: set[str],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Load split AliceAI conv weights through the upstream fused-conv ABI."""
    for name, weight in weights:
        for source_name, (target_name, shard_id) in _SPLIT_KDA_CONV_SHARDS.items():
            if source_name not in name:
                continue
            mapped_name = name.replace(source_name, target_name, 1)
            param = params.get(mapped_name)
            if param is None:
                raise ValueError(f"No target parameter for {name!r}: {mapped_name!r}")
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                raise ValueError(
                    f"Target parameter {mapped_name!r} has no weight loader"
                )
            weight_loader(param, weight, shard_id)
            loaded.add(mapped_name)
            break
        else:
            yield name, weight


def get_attn_res_block_size(hf_config: object) -> int:
    """Return the architecture-rebase AttnRes grouping, not legacy metadata."""
    block_size = getattr(hf_config, "block_attn_res_block_size", None)
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError(
            "AliceAI requires a positive integer block_attn_res_block_size"
        )
    if block_size <= 0:
        raise ValueError("block_attn_res_block_size must be positive")
    return block_size


def _make_kimi_kda_config(config: Qwen3NextConfig) -> KimiLinearConfig:
    """Adapt the flat AliceAI KDA geometry to the existing upstream Kimi layer."""
    num_heads = config.linear_num_key_heads
    num_value_heads = config.linear_num_value_heads
    head_dim = config.linear_key_head_dim
    value_head_dim = config.linear_value_head_dim
    if num_heads != num_value_heads or head_dim != value_head_dim:
        raise ValueError(
            "AliceAI KDA requires matching key/value head counts and dimensions"
        )

    adapted = copy(config)
    adapted.linear_attn_config = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "short_conv_kernel_size": config.linear_conv_kernel_dim,
        "use_full_rank_gate": False,
        "gate_lower_bound": None,
    }
    return cast(KimiLinearConfig, adapted)


def _make_alice_ai_full_attention_config(
    config: Qwen3NextConfig,
) -> Qwen3NextConfig:
    adapted = copy(config)
    adapted.attn_output_gate = True
    adapted.qkv_bias = False
    adapted.dual_chunk_attention_config = None
    adapted.is_causal = True
    return adapted


def _mix_attn_res(
    prefix: torch.Tensor,
    residual_bank: torch.Tensor,
    norm_weight: torch.Tensor,
    query_weight: torch.Tensor,
    *,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    """Use the upstream Kimi primitive without mutating AliceAI residual state."""
    return upstream_attn_res(
        prefix,
        None,
        residual_bank,
        norm_weight,
        query_weight,
        None,
        num_blocks,
        -1,
        eps,
        0.0,
    )


class _AttentionResidualFinalMixer(nn.Module):
    def __init__(self, hidden_size: int, rms_norm_eps: float, prefix: str) -> None:
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.res_proj = ReplicatedLinear(
            hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.res_proj",
        )
        self.res_norm_weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        residual_bank: torch.Tensor,
        num_completed_sources: int,
        partial: torch.Tensor | None,
    ) -> torch.Tensor:
        if partial is None:
            if num_completed_sources < 1:
                raise RuntimeError("AttnRes requires at least one residual source")
            prefix = residual_bank[:, num_completed_sources - 1]
            num_blocks = num_completed_sources - 1
        else:
            prefix = partial
            num_blocks = num_completed_sources
        return _mix_attn_res(
            prefix,
            residual_bank,
            self.res_norm_weight,
            self.res_proj.weight.squeeze(0),
            num_blocks=num_blocks,
            eps=self.rms_norm_eps,
        )


class AliceAIDecoderBlock(Qwen3NextDecoderLayer):
    """AliceAI topology with upstream Qwen, Kimi KDA, MoE and AttnRes operations."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        if config.num_experts <= 0:
            raise ValueError("AliceAI requires sparse experts")
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError("AliceAI does not support sequence-parallel MoE")

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.attn_res_block_size = get_attn_res_block_size(config)
        self.attn_res_rms_norm_eps = float(config.rms_norm_eps)
        self.use_attn_reduce_scatter_for_moe = False

        if layer_type == "linear_attention":
            self.linear_attn = KimiGatedDeltaNetAttention(
                _make_kimi_kda_config(config),
                vllm_config,
                prefix=f"{prefix}.linear_attn",
            )
            self.linear_attn.o_norm.eps = float(config.rms_norm_eps)
        elif layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                _make_alice_ai_full_attention_config(config),
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen3NextSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = False

        if self.layer_idx > 0:
            self.attn_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.attn_res_proj",
            )
            self.attn_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.mlp_res_proj = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.mlp_res_proj",
        )
        self.mlp_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))

    def _mix_attention_residuals(
        self,
        residual_bank: torch.Tensor,
        num_completed_sources: int,
        partial: torch.Tensor | None,
        *,
        projection: ReplicatedLinear,
        norm_weight_name: str,
    ) -> torch.Tensor:
        if partial is None:
            if num_completed_sources < 1:
                raise RuntimeError("AttnRes requires at least one residual source")
            prefix = residual_bank[:, num_completed_sources - 1]
            num_blocks = num_completed_sources - 1
        else:
            prefix = partial
            num_blocks = num_completed_sources
        return _mix_attn_res(
            prefix,
            residual_bank,
            getattr(self, norm_weight_name),
            projection.weight.squeeze(0),
            num_blocks=num_blocks,
            eps=self.attn_res_rms_norm_eps,
        )

    def _run_attention(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            output = torch.empty_like(hidden_states)
            self.linear_attn(hidden_states, positions, output)
            return output
        return self.self_attn(hidden_states=hidden_states, positions=positions)

    def forward_attn_res(
        self,
        positions: torch.Tensor,
        residual_bank: torch.Tensor,
        num_completed_sources: int,
        partial: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        if self.layer_idx > 0 and self.layer_idx % self.attn_res_block_size == 0:
            if partial is None:
                raise RuntimeError("cannot finalize an empty AttnRes block")
            residual_bank[:, num_completed_sources].copy_(partial)
            num_completed_sources += 1
            partial = None

        if self.layer_idx == 0:
            hidden_states = residual_bank[:, 0]
        else:
            hidden_states = self._mix_attention_residuals(
                residual_bank,
                num_completed_sources,
                partial,
                projection=self.attn_res_proj,
                norm_weight_name="attn_res_norm_weight",
            )
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self._run_attention(hidden_states, positions)
        partial = attention_output if partial is None else partial + attention_output

        hidden_states = self._mix_attention_residuals(
            residual_bank,
            num_completed_sources,
            partial,
            projection=self.mlp_res_proj,
            norm_weight_name="mlp_res_norm_weight",
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        return num_completed_sources, partial + mlp_output


class AliceAIModel(Qwen3NextModel):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".linear_attn.a_log_bias": ".linear_attn.A_log",
        },
        orig_to_new_stacked={
            ".linear_attn.q_proj": (".linear_attn.in_proj_qkvgfab", 0),
            ".linear_attn.k_proj": (".linear_attn.in_proj_qkvgfab", 1),
            ".linear_attn.v_proj": (".linear_attn.in_proj_qkvgfab", 2),
            ".linear_attn.b_proj": (".linear_attn.in_proj_qkvgfab", 3),
            ".linear_attn.f_a_proj": (".linear_attn.in_proj_qkvgfab", 4),
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        self.attn_res_block_size = get_attn_res_block_size(config)
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=AliceAIDecoderBlock,
        )
        self.attnres_final = _AttentionResidualFinalMixer(
            config.hidden_size,
            float(config.rms_norm_eps),
            prefix=f"{prefix}.attnres_final",
        )
        spec_config = vllm_config.speculative_config
        needs_pre_final_hidden = spec_config is not None and spec_config.method == "mtp"
        if needs_pre_final_hidden:
            # AliceAI MTP was trained on hidden states before the target's final norm.
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def _init_residual_bank(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual_bank = hidden_states.new_empty(
            (
                hidden_states.shape[0],
                (self.config.num_hidden_layers + self.attn_res_block_size - 1)
                // self.attn_res_block_size,
                hidden_states.shape[-1],
            )
        )
        residual_bank[:, 0].copy_(hidden_states)
        return residual_bank

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            raise NotImplementedError("AliceAI does not support pipeline parallelism")
        if self.aux_hidden_state_layers:
            raise NotImplementedError(
                "AliceAI does not support auxiliary hidden states"
            )
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise NotImplementedError("AliceAI requires pipeline_parallel_size=1")

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        residual_bank = self._init_residual_bank(hidden_states)

        num_completed_sources = 1
        partial = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            num_completed_sources, partial = layer.forward_attn_res(
                positions, residual_bank, num_completed_sources, partial
            )
        if partial is None:
            raise RuntimeError("AttnRes produced an empty final block")
        hidden_states = self.attnres_final(
            residual_bank, num_completed_sources, partial
        )
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states)
        return self.norm(hidden_states)


class AliceAIForCausalLM(Qwen3NextForCausalLM):
    packed_modules_mapping = {
        **Qwen3NextForCausalLM.packed_modules_mapping,
        "in_proj_qkvgfab": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
        ],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # AttnRes calls bypass the per-layer forward hooks used by offloading.
        offloader = get_offloader()
        if isinstance(offloader, PrefetchOffloader):
            raise NotImplementedError("AliceAI does not support prefetch offloading")
        if (
            isinstance(offloader, UVAOffloader)
            and offloader.cpu_offload_max_bytes > 0
            and not offloader.uva_offloading
        ):
            raise NotImplementedError("AliceAI CPU offloading requires UVA")

        super().__init__(vllm_config=vllm_config, prefix=prefix, model_cls=AliceAIModel)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        manually_loaded: set[str] = set()
        weights = _load_split_kda_conv_weights(
            weights,
            dict(self.named_parameters()),
            manually_loaded,
        )
        loaded = super().load_weights(weights)
        loaded.update(manually_loaded)
        expected_target_weights = {
            name
            for name, _ in self.named_parameters()
            if ".attnres_final." in name
            or ".attn_res_" in name
            or ".mlp_res_" in name
            or name.endswith(".e_score_correction_bias")
        }
        if missing := expected_target_weights - loaded:
            raise ValueError(
                "AliceAI checkpoint is missing target-specific weights: "
                + ", ".join(sorted(missing))
            )
        return loaded
