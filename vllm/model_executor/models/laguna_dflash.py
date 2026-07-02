# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash speculator for Laguna target models.

Laguna DFlash uses a uniform drafter layer flavor (`layer_types` all full
or all sliding). The draft checkpoint shares token embedding and lm_head
weights with the target model through the generic spec-decode proposer.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.multimodal.inputs import NestedTensors

from .laguna import LagunaDecoderLayer
from .qwen3_dflash import DFlashQwen3Model
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


def _get_dflash_layer_types(config) -> tuple[str, ...]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        raise ValueError("Laguna DFlash config requires `layer_types`.")
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"DFlash layer_types length {len(layer_types)} does not match "
            f"num_hidden_layers {config.num_hidden_layers}."
        )
    # Laguna DFlash checkpoints use a uniform drafter attention flavor.
    if len(set(layer_types)) > 1:
        raise NotImplementedError(
            "Laguna DFlash drafter requires a uniform `layer_types` "
            f"(got {sorted(set(layer_types))})."
        )
    return tuple(layer_types)


@support_torch_compile
class DFlashLagunaModel(DFlashQwen3Model, EagleModelMixin):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        target_layer_ids = self.config.dflash_config["target_layer_ids"]
        if not target_layer_ids:
            raise ValueError(
                "Laguna DFlash config requires non-empty "
                "`dflash_config.target_layer_ids`."
            )

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.mask_token_id = self.config.dflash_config.get("mask_token_id")
        self.register_buffer(
            "mask_embedding",
            torch.zeros(
                self.config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )
        self.has_separate_mask_embedding = False

        self.layer_types = _get_dflash_layer_types(self.config)
        target_layer_count = self.config.target_layer_count
        self.layers = nn.ModuleList(
            [
                LagunaDecoderLayer(
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx}"),
                    config=self.config,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    layer_idx=layer_idx,
                    attention_prefix=maybe_prefix(
                        prefix, f"layers.{layer_idx + target_layer_count}"
                    ),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        for layer in self.layers:
            if getattr(layer.self_attn, "sliding_window", None) is not None:
                # DFlash inserts verifier-context K/V at absolute cache slots.
                # Keep full KV allocation; SWA remains a compute-time limit.
                layer.self_attn.attn.sliding_window = None
        num_features_to_use = len(target_layer_ids)
        target_hidden_size = vllm_config.model_config.get_hidden_size()
        fc_input_size = target_hidden_size * num_features_to_use
        self.num_aux_slices = num_features_to_use
        self.aux_hidden_norms = nn.ModuleList(
            [
                RMSNorm(
                    fc_input_size // num_features_to_use,
                    eps=self.config.rms_norm_eps,
                )
                for _ in range(num_features_to_use)
            ]
        )
        self.fc = ReplicatedLinear(
            input_size=fc_input_size,
            output_size=self.config.hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def _build_context_kv_buffers(
        self,
        layers_attn: list[nn.Module],
        has_bias: bool,
    ) -> None:
        self._kv_weights = torch.stack(
            [a.qkv_proj.weight[a.q_size :] for a in layers_attn], dim=0
        ).contiguous()
        if has_bias:
            self._kv_biases: torch.Tensor | None = torch.stack(
                [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
            ).contiguous()
        else:
            self._kv_biases = None
        self._input_layernorm_weights = torch.stack(
            [layer.input_layernorm.weight.data for layer in self.layers], dim=0
        ).contiguous()
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed_context_states = torch.empty(
            (num_layers, num_ctx, context_states.shape[-1]),
            dtype=context_states.dtype,
            device=context_states.device,
        )
        ops.rms_norm(
            normed_context_states,
            context_states.unsqueeze(0).expand(num_layers, -1, -1),
            self._input_layernorm_weights,
            self._rms_norm_eps,
        )
        all_kv_flat = torch.bmm(
            normed_context_states,
            self._kv_weights.transpose(1, 2),
        )
        if self._kv_biases is not None:
            all_kv_flat += self._kv_biases[:, None, :]
        all_kv = (
            all_kv_flat.view(num_layers, num_ctx, 2, num_kv_heads, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]
        all_v = all_kv[1]
        return all_k, all_v

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        all_k_normed = torch.empty_like(all_k)
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashLagunaForCausalLM(nn.Module, SupportsEagle3):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            raise ValueError("Laguna DFlash config requires `draft_vocab_size`.")
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "Laguna DFlash shares the target lm_head and requires "
                "`draft_vocab_size` to match the target vocabulary size "
                f"({self.config.draft_vocab_size} != {target_vocab_size})."
            )
        self.model = DFlashLagunaModel(
            vllm_config=vllm_config,
            prefix="model",
        )

        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize each verifier hidden-state slice, concatenate them, then
        # project into the drafter hidden size used as DFlash context.
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        num_slices = self.model.num_aux_slices
        slice_size = hidden_states.shape[-1] // num_slices
        slices = hidden_states.view(hidden_states.shape[0], num_slices, slice_size)
        normed = torch.empty_like(slices)
        for i, norm in enumerate(self.model.aux_hidden_norms):
            normed[:, i, :] = norm(slices[:, i, :])
        hidden_states = normed.reshape(hidden_states.shape[0], -1)
        result = self.model.fc(hidden_states)
        result = self.model.hidden_norm(result)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        loader = AutoWeightsLoader(self)
        loaded_weight_names = loader.load_weights(model_weights.items())
        loaded_weight_names.add("lm_head.weight")
        loaded_weight_names.add("model.embed_tokens.weight")
        self.model._build_fused_kv_buffers()
        return loaded_weight_names
