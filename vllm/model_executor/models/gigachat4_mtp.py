# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.index_group import (
    SparseMLAIndexGroupBuilder,
    get_sparse_mla_index_group_max_rows,
)

from .gigachat4 import (
    GigaChat4DecoderLayer,
    GigaChat4ForCausalLM,
    GigaChat4GatedNorm,
    _prepare_gigachat4_fp8_config,
)
from .qwen3_next_mtp import should_torch_compile_mtp_drafter
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)


class GigaChat4MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor | None,
        index_group_builder: SparseMLAIndexGroupBuilder | None,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.enorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.hnorm = GigaChat4GatedNorm(config, config.hidden_size)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.mtp_block = GigaChat4DecoderLayer(
            vllm_config,
            topk_indices_buffer,
            index_group_builder,
            prefix,
            is_nextn=True,
        )
        self.shared_head = nn.Module()
        self.shared_head.norm = GigaChat4GatedNorm(config, config.hidden_size)
        self.shared_head.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.shared_head.head",
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = torch.cat((inputs_embeds, previous_hidden_states), dim=-1)
        hidden_states = self.eh_proj(hidden_states)
        # The next draft step consumes the state before shared_head.norm.
        return self.mtp_block(positions, hidden_states)


class GigaChat4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        topk_indices_buffer = None
        index_group_builder = None
        if config.use_dsa:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=current_platform.device_type,
            )
            index_group_builder = SparseMLAIndexGroupBuilder(
                topk_indices_buffer, get_sparse_mla_index_group_max_rows(vllm_config)
            )
        self.layers = nn.ModuleDict(
            {
                str(idx): GigaChat4MultiTokenPredictorLayer(
                    vllm_config,
                    topk_indices_buffer,
                    index_group_builder,
                    f"{prefix}.layers.{idx}",
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
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        layer_idx = self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers
        return self.layers[str(layer_idx)](
            positions, previous_hidden_states, inputs_embeds
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        layer_idx = self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers
        head = self.layers[str(layer_idx)].shared_head
        hidden_states = head.norm(hidden_states)
        return self.logits_processor(head.head, hidden_states)


@support_torch_compile(enable_if=should_torch_compile_mtp_drafter)
class GigaChat4MTP(nn.Module):
    has_own_embed_tokens = True
    has_own_lm_head = True
    packed_modules_mapping = GigaChat4ForCausalLM.packed_modules_mapping
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".self_attn.q_a_proj.": (".self_attn.fused_qkv_a_proj.", 0),
            ".self_attn.kv_a_proj_with_mqa.": (".self_attn.fused_qkv_a_proj.", 1),
            ".mlp.gate_proj.": (".mlp.gate_up_proj.", 0),
            ".mlp.up_proj.": (".mlp.gate_up_proj.", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        spec = vllm_config.speculative_config
        assert spec is not None
        vllm_config = replace(
            vllm_config,
            model_config=spec.draft_model_config,
            quant_config=_prepare_gigachat4_fp8_config(vllm_config.quant_config),
            parallel_config=replace(
                vllm_config.parallel_config,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
            ),
        )
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = GigaChat4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

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
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _nextn_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        native_vocab_weights: set[str] = set()
        for name, weight in weights:
            if name == "model.embed_tokens.weight":
                if name not in native_vocab_weights:
                    yield name, weight
                continue
            if name == "lm_head.weight":
                for head_idx in self.model.layers:
                    head_name = f"model.layers.{head_idx}.shared_head.head.weight"
                    if head_name not in native_vocab_weights:
                        yield head_name, weight
                continue
            layer_idx = get_spec_layer_idx_from_weight_name(self.config, name)
            if layer_idx is None:
                continue
            prefix = f"model.layers.{layer_idx}."
            suffix = name.removeprefix(prefix)
            if suffix.startswith("embed_tokens."):
                if layer_idx != self.config.num_hidden_layers:
                    continue
                name = f"model.{suffix}"
            elif not suffix.startswith(
                ("enorm.", "hnorm.", "eh_proj.", "shared_head.")
            ):
                name = f"{prefix}mtp_block.{suffix}"
            if suffix in ("embed_tokens.weight", "shared_head.head.weight"):
                native_vocab_weights.add(name)
            yield name, weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            self._nextn_weights(weights), mapper=self.hf_to_vllm_mapper
        )
