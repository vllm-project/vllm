# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only AliceAI model."""

from collections.abc import Iterable
from copy import copy
from itertools import islice
from typing import cast

import torch
from torch import nn
from transformers import PreTrainedConfig, Qwen3NextConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
    resolve_layer_fused_shared_expert,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_quark_ocp_mx_group_size,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    _should_replicate_misaligned_shared_expert,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
)
from vllm.models.kimi_k3.nvidia.ops import attn_res
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

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


def _make_kimi_kda_config(config: PreTrainedConfig) -> KimiLinearConfig:
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
    config: PreTrainedConfig,
) -> Qwen3NextConfig:
    adapted = copy(config)
    adapted.attn_output_gate = True
    adapted.qkv_bias = False
    adapted.dual_chunk_attention_config = None
    adapted.is_causal = True
    return cast(Qwen3NextConfig, adapted)


class _AttentionResidualFinal(nn.Module):
    def __init__(self, hidden_size: int, rms_norm_eps: float, prefix: str) -> None:
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.res_proj = ReplicatedLinear(
            input_size=hidden_size,
            output_size=1,
            bias=False,
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
        return attn_res(
            prefix=prefix,
            delta=None,
            blocks=residual_bank,
            norm_weight=self.res_norm_weight,
            qk_weight=self.res_proj.weight.squeeze(0),
            output_norm_weight=None,
            num_blocks=num_blocks,
            block_write_idx=-1,
            eps=self.rms_norm_eps,
            output_norm_eps=0.0,
        )


class AliceAISparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_ep_group().device_group.size()
        self.n_routed_experts = config.num_experts
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )

        self.is_fused_shared_expert_enabled = False
        if config.shared_expert_intermediate_size > 0:
            self.is_fused_shared_expert_enabled = resolve_layer_fused_shared_expert(
                quant_config=quant_config,
                prefix=prefix,
                shared_expert_name="shared_expert",
            )

        self.replicate_shared_expert = False
        if not self.is_fused_shared_expert_enabled:
            shared_expert_group_size = get_quark_ocp_mx_group_size(
                quant_config=quant_config,
                layer_name=f"{prefix}.shared_expert.down_proj",
            )
            self.replicate_shared_expert = _should_replicate_misaligned_shared_expert(
                intermediate_size=config.shared_expert_intermediate_size,
                tp_size=self.tp_size,
                group_size=shared_expert_group_size,
                enable_expert_parallel=parallel_config.enable_expert_parallel,
                is_sequence_parallel=False,
            )

        self.enable_eplb = parallel_config.enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.num_experts,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.shared_expert_gate = ReplicatedLinear(
            input_size=config.hidden_size,
            output_size=1,
            bias=False,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if (
            self.is_fused_shared_expert_enabled
            or config.shared_expert_intermediate_size <= 0
        ):
            self.shared_expert = None
        else:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                disable_tp=self.replicate_shared_expert,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoEFactory(
            shared_experts=(
                None if self.replicate_shared_expert else self.shared_expert
            ),
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            n_shared_experts=1 if self.shared_expert is None else None,
            fuse_shared_experts=self.is_fused_shared_expert_enabled,
            shared_expert_prefix=f"{prefix}.shared_expert",
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        shared_output = (
            self.shared_expert(hidden_states)
            if self.replicate_shared_expert and self.shared_expert is not None
            else None
        )
        output = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        if shared_output is not None:
            output += shared_output
        return output.view(orig_shape)


class AliceAIDecoderBlock(nn.Module):
    """AliceAI topology with upstream Qwen, Kimi KDA, MoE and AttnRes operations."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

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

        if layer_type == "linear_attention":
            self.linear_attn = KimiGatedDeltaNetAttention(
                config=_make_kimi_kda_config(config),
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
            )
            self.linear_attn.o_norm.eps = float(config.rms_norm_eps)
        elif layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config=_make_alice_ai_full_attention_config(config),
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = AliceAISparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )

        if self.layer_idx > 0:
            self.attn_res_proj = ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=1,
                bias=False,
                prefix=f"{prefix}.attn_res_proj",
            )
            self.attn_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.mlp_res_proj = ReplicatedLinear(
            input_size=config.hidden_size,
            output_size=1,
            bias=False,
            prefix=f"{prefix}.mlp_res_proj",
        )
        self.mlp_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))

    def _apply_attention_residual(
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
        return attn_res(
            prefix=prefix,
            delta=None,
            blocks=residual_bank,
            norm_weight=getattr(self, norm_weight_name),
            qk_weight=projection.weight.squeeze(0),
            output_norm_weight=None,
            num_blocks=num_blocks,
            block_write_idx=-1,
            eps=self.attn_res_rms_norm_eps,
            output_norm_eps=0.0,
        )

    def _apply_attention(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            return self.linear_attn(hidden_states=hidden_states, positions=positions)
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
            hidden_states = self._apply_attention_residual(
                residual_bank=residual_bank,
                num_completed_sources=num_completed_sources,
                partial=partial,
                projection=self.attn_res_proj,
                norm_weight_name="attn_res_norm_weight",
            )
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self._apply_attention(
            hidden_states=hidden_states, positions=positions
        )
        partial = attention_output if partial is None else partial + attention_output

        hidden_states = self._apply_attention_residual(
            residual_bank=residual_bank,
            num_completed_sources=num_completed_sources,
            partial=partial,
            projection=self.mlp_res_proj,
            norm_weight_name="mlp_res_norm_weight",
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        return num_completed_sources, partial + mlp_output


@support_torch_compile
class AliceAIModel(nn.Module):
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
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.attn_res_block_size = get_attn_res_block_size(config)
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )

        def get_layer(prefix: str) -> AliceAIDecoderBlock:
            return AliceAIDecoderBlock(
                vllm_config=vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            num_hidden_layers=config.num_hidden_layers,
            layer_fn=get_layer,
            prefix=f"{prefix}.layers",
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            layers=self.layers, moe_cls=AliceAISparseMoeBlock, moe_name="mlp"
        )
        self.norm = GemmaRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.attnres_final = _AttentionResidualFinal(
            hidden_size=config.hidden_size,
            rms_norm_eps=float(config.rms_norm_eps),
            prefix=f"{prefix}.attnres_final",
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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
                positions=positions,
                residual_bank=residual_bank,
                num_completed_sources=num_completed_sources,
                partial=partial,
            )
        if partial is None:
            raise RuntimeError("AttnRes produced an empty final block")
        hidden_states = self.attnres_final(
            residual_bank=residual_bank,
            num_completed_sources=num_completed_sources,
            partial=partial,
        )
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights=weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=self.config.num_experts,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights=weights, mapper=self.hf_to_vllm_mapper)


class AliceAIForCausalLM(
    nn.Module, HasInnerState, IsHybrid, SupportsLoRA, MixtureOfExperts
):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"mtp.": None})

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
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

        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.model = AliceAIModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.moe_layers = [layer.mlp.experts for layer in self.model.layers]
        moe = self.model.layers[self.model.start_layer].mlp
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = moe.n_logical_experts
        self.num_physical_experts = moe.n_physical_experts
        self.num_local_physical_experts = moe.n_local_physical_experts
        self.num_routed_experts = moe.n_routed_experts
        self.num_redundant_experts = moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            moe = layer.mlp
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(lm_head=self.lm_head, hidden_states=hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            model_dtype=vllm_config.model_config.dtype,
            mamba_cache_dtype=vllm_config.cache_config.mamba_cache_dtype,
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
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            num_heads=config.linear_num_key_heads,
            head_dim=config.linear_key_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)
        manually_loaded: set[str] = set()
        weights = _load_split_kda_conv_weights(
            weights=weights,
            params=dict(self.named_parameters()),
            loaded=manually_loaded,
        )
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights=weights)
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
