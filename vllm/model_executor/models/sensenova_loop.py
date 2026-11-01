# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text inference for fixed-count SenseNova Loop checkpoints with optional mHC."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.model_executor.layers.sensenova_loop_mhc import MHCRouter
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from .qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLMBase,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from .utils import AutoWeightsLoader, maybe_prefix


class SenseNovaLoopDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str) -> None:
        super().__init__(vllm_config, layer_type, prefix)
        config = vllm_config.model_config.hf_text_config
        self.physical_index = config.loop_physical_indices[self.layer_idx]
        self.mhc = config.mhc
        self.mhc_active = self.mhc is not None and (
            self.mhc["start"] <= self.physical_index < self.mhc["end"]
        )
        if self.mhc_active:
            self.mhc_attn = MHCRouter(config.hidden_size, self.mhc)
            self.mhc_ffn = MHCRouter(config.hidden_size, self.mhc)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.mhc_active:
            return super().forward(hidden_states, residual, positions, **kwargs)
        if residual is not None:
            hidden_states = hidden_states + residual
        streams = self.mhc["streams"]
        if self.physical_index == self.mhc["start"]:
            hidden_states = hidden_states.unsqueeze(-2).expand(-1, streams, -1)
        else:
            hidden_states = hidden_states.unflatten(-1, (streams, -1))
        branch, post, mix = self.mhc_attn(hidden_states)
        branch = self.input_layernorm(branch.contiguous())
        if self.layer_type == "linear_attention":
            branch = self.linear_attn(hidden_states=branch)
        else:
            branch = self.self_attn(hidden_states=branch, positions=positions)
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        branch, post, mix = self.mhc_ffn(hidden_states)
        branch = self.mlp(self.post_attention_layernorm(branch.contiguous()))
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        if self.physical_index == self.mhc["end"] - 1:
            hidden_states = hidden_states.mean(-2)
        else:
            hidden_states = hidden_states.flatten(-2)
        return hidden_states, None


class SenseNovaLoopModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = None
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            SenseNovaLoopDecoderLayer(vllm_config, layer_type, f"{prefix}.layers.{i}")
            for i, layer_type in enumerate(self.config.layer_types)
        )
        self.norm = Qwen3_5RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions)
        if residual is None:
            return self.norm(hidden_states)
        return self.norm(hidden_states, residual)[0]

    def share_loop_parameters(self) -> None:
        first: dict[int, nn.Module] = {}
        for physical, layer in zip(self.config.loop_physical_indices, self.layers):
            if physical not in first:
                first[physical] = layer
                continue
            for name, param in first[physical].named_parameters():
                parent, _, leaf = name.rpartition(".")
                setattr(layer.get_submodule(parent), leaf, param)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=Qwen3_5Model.hf_to_vllm_mapper
        )


class SenseNovaLoopForConditionalGeneration(
    nn.Module, HasInnerState, IsHybrid, SupportsMRoPE
):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_copy_func()

    get_mrope_input_positions = Qwen3_5ForCausalLMBase.get_mrope_input_positions

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = self.model_config.hf_text_config
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.quant_config is not None
            or vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None
            or getattr(self.config, "layer_scale", False)
        ):
            raise ValueError(
                "SenseNova Loop currently requires unquantized text inference "
                "without pipeline parallelism, LoRA, speculation or layer_scale"
            )
        self.quant_config = None
        self.scheduler_config = vllm_config.scheduler_config
        self.model = SenseNovaLoopModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        destinations: dict[int, list[int]] = {}
        for execution, physical in enumerate(self.config.loop_physical_indices):
            destinations.setdefault(physical, []).append(execution)

        def expanded_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                name = name.replace("model.language_model.", "model.", 1)
                if name == "model.loop_gate.logits":
                    continue
                if name.startswith(("model.visual.", "visual.", "mtp.")):
                    continue
                if name.startswith("model.layers."):
                    _, _, index, suffix = name.split(".", 3)
                    for execution in destinations[int(index)]:
                        yield f"model.layers.{execution}.{suffix}", weight
                else:
                    yield name, weight

        return AutoWeightsLoader(self).load_weights(expanded_weights())

    def process_weights_after_loading(self) -> None:
        # Parameters share storage; attention modules and their caches stay distinct.
        self.model.share_loop_parameters()
