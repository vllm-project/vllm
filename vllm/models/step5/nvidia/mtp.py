# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.sequence import IntermediateTensors

from .layernorm import Step5RMSNorm, get_step5_norm_dtype
from .model import (
    _STEP5_WEIGHT_SHARDS,
    STEP5_PACKED_MODULES_MAPPING,
    Step5DecoderLayer,
    Step5MoE,
    _reset_fused_qkv_indexer_load_state,
    _set_step5_moe_protocol_metadata,
    _step5_sp_padding,
    _validate_fused_qkv_indexer_weights,
)

logger = init_logger(__name__)


def _get_mtp_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        return vllm_config
    return replace(vllm_config, model_config=speculative_config.draft_model_config)


def _get_mtp_config(vllm_config: VllmConfig) -> PreTrainedConfig:
    return _get_mtp_vllm_config(vllm_config).model_config.hf_text_config


def _build_mtp_norm(config: PreTrainedConfig) -> nn.Module:
    return Step5RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        zero_centered=config.zero_centered,
        dtype=get_step5_norm_dtype(config),
    )


class Step5MTPSharedHead(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Keep MTP normalization and residual precision aligned with the target
        # model configuration.
        self.fp32_residual_connection = config.fp32_residual_connection
        self.norm = _build_mtp_norm(config)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.bfloat16)
        return self.norm(hidden_states)


class Step5MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        mtp_vllm_config = _get_mtp_vllm_config(vllm_config)
        config = mtp_vllm_config.model_config.hf_text_config
        self.fp32_residual_connection = config.fp32_residual_connection
        quant_config = vllm_config.quant_config
        self.enorm = _build_mtp_norm(config)
        self.hnorm = _build_mtp_norm(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = Step5MTPSharedHead(
            config=config, quant_config=quant_config, prefix=f"{prefix}.shared_head"
        )
        self.mtp_block = Step5DecoderLayer(
            mtp_vllm_config,
            prefix=f"{prefix}.mtp_block",
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
        if (
            inputs_embeds.is_cuda
            and previous_hidden_states.device == inputs_embeds.device
            and inputs_embeds.ndim == previous_hidden_states.ndim == 2
            and inputs_embeds.shape == previous_hidden_states.shape
            and inputs_embeds.stride(-1) == previous_hidden_states.stride(-1) == 1
            and 0 < inputs_embeds.shape[-1] <= 16384
            and inputs_embeds.dtype == torch.bfloat16
            and previous_hidden_states.dtype in (torch.bfloat16, torch.float32)
            and self.fp32_residual_connection
        ):
            from .ops.mtp_norm import norm_concat

            eh_input = norm_concat(
                inputs_embeds,
                previous_hidden_states,
                self.enorm.weight,
                self.hnorm.weight,
                self.enorm.variance_epsilon,
                self.enorm.zero_centered,
                self.fp32_residual_connection,
            )
        else:
            if self.fp32_residual_connection:
                inputs_embeds = inputs_embeds.to(torch.bfloat16)
                previous_hidden_states = previous_hidden_states.to(torch.bfloat16)
            eh_input = torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1
            )

        hidden_states = self.eh_proj(eh_input)
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.float32)
        padding = None
        if self.mtp_block.use_sequence_parallel:
            padding = _step5_sp_padding(hidden_states)
            hidden_states = sp_shard(hidden_states)
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, padding=padding
        )
        hidden_states = self.mtp_block._add_for_residual(hidden_states, residual)
        if self.mtp_block.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
        return hidden_states


class Step5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_mtp_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers <= 0:
            raise ValueError(
                "Step5 MTP requires num_nextn_predict_layers > 0, got "
                f"{self.num_mtp_layers}."
            )
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Step5MultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

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
        return self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head,
            mtp_layer.shared_head(hidden_states),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def _get_step5_mtp_moe_blocks(
    model: Step5MultiTokenPredictor,
) -> list[Step5MoE]:
    blocks: list[Step5MoE] = []
    for predictor_layer in model.layers.values():
        mtp_block = getattr(predictor_layer, "mtp_block", None)
        moe = getattr(mtp_block, "moe", None)
        if isinstance(moe, Step5MoE):
            blocks.append(moe)
    return blocks


def _is_step5_mtp_dense(model: Step5MultiTokenPredictor) -> bool:
    """Return whether every draft transformer block is dense."""
    return not _get_step5_mtp_moe_blocks(model)


class Step5MTP(nn.Module, MixtureOfExperts, SupportsMultiModal):
    supports_multimodal_embeddings = True
    has_own_lm_head = True
    packed_modules_mapping = STEP5_PACKED_MODULES_MAPPING
    _enable_weights_track_by_default = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # The target VL model maps quantization exclusions to language_model.*.
        # Draft modules live at model.* and must not mutate the target config.
        if vllm_config.quant_config is not None:
            quant_config = deepcopy(vllm_config.quant_config)
            quant_config.apply_vllm_mapper(
                WeightsMapper(orig_to_new_prefix={"language_model.": ""})
            )
            vllm_config = replace(vllm_config, quant_config=quant_config)
        self.config = _get_mtp_config(vllm_config)
        self.vllm_config = vllm_config
        self.model = Step5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Expose the first draft head to the generic sharing policy. Each MTP
        # layer retains its own checkpoint head for compute_logits.
        self.lm_head = self.model.layers[
            str(self.model.mtp_start_layer_idx)
        ].shared_head.head
        self.num_moe_layers: int
        self.num_expert_groups: int
        self.num_shared_experts: int
        self.num_logical_experts: int
        self.num_physical_experts: int
        self.num_local_physical_experts: int
        self.num_routed_experts: int
        self.num_redundant_experts: int
        if not _is_step5_mtp_dense(self.model):
            raise NotImplementedError("Step5 MTP supports dense draft layers only")
        # Dense MTP forwards must not inherit the target model's DP collectives.
        self.is_dense_mtp = True
        self.moe_layers = []
        _set_step5_moe_protocol_metadata(self, None)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids, self.model.embed_input_ids, is_multimodal=is_multimodal
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
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

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.mtp_validation import (
            is_mtp_completeness_check_enabled,
        )
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )

        validate_completeness = is_mtp_completeness_check_enabled()
        if validate_completeness:
            _reset_fused_qkv_indexer_load_state(self)
        params_dict = dict(self.named_parameters())

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("layers."):
                name = f"model.{name}"
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if "embed_tokens" not in name and spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            for param_name, weight_name, shard_id in _STEP5_WEIGHT_SHARDS:
                if weight_name not in name or "experts" in name or "moe" in name:
                    continue
                replaced_name = name.replace(weight_name, param_name)
                if replaced_name not in params_dict:
                    continue
                name = replaced_name
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias")
                    and name not in params_dict
                    or "tok_embeddings" in name
                ):
                    continue

                if spec_layer is not None and ".transformer." in name:
                    name = name.replace(".transformer.", ".")
                if "shared_head" in name:
                    name = name.replace("shared_head.output", "shared_head.head")
                if "embed_tokens" in name:
                    name = "model.embed_tokens.weight"
                if name not in params_dict:
                    logger.warning_once(
                        "Skipping unexpected MTP checkpoint weight: %s", name
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        if validate_completeness:
            loaded_params.update(_validate_fused_qkv_indexer_weights(self))
        missing_params = set(params_dict).difference(loaded_params)
        if missing_params and validate_completeness:
            param_name_example = min(missing_params)
            raise RuntimeError(
                "Some parameters like "
                f"{param_name_example} are not in the checkpoint and will falsely "
                "use random initialization"
            )
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int | None, name: str) -> str:
        """Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        return name
