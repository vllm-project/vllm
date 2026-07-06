# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from vllm.multimodal.inputs import NestedTensors

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        layer_idx: int = 0,
    ) -> None:
        # Eagle3 has its own unfused decoder forward (linears reduce themselves).
        super().__init__(vllm_config, prefix=prefix, config=config, reduce_results=True)

        config = config or vllm_config.model_config.hf_config
        quant_config = self.get_quant_config(vllm_config)

        # First layer uses 2*hidden_size (embeds + hidden_states concatenated)
        # Subsequent layers use hidden_size (only hidden_states, no embeds)
        qkv_input_size = 2 * self.hidden_size if layer_idx == 0 else self.hidden_size

        # Parallel drafting checkpoints may have attention bias enabled
        qkv_bias = getattr(config, "attention_bias", False)

        # Override qkv_proj with correct input size and bias setting
        self.self_attn.qkv_proj = QKVParallelLinear(
            qkv_input_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "qkv_proj"),
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

        if getattr(config, "norm_before_residual", False):
            self._residual_norm = self._norm_before_residual
        else:
            self._residual_norm = self._norm_after_residual

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Use drafter's quantization config instead of verifier's."""
        return get_draft_quant_config(vllm_config)

    def _norm_before_residual(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.hidden_norm(hidden_states)
        residual = hidden_states
        return hidden_states, residual

    def _norm_after_residual(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            # First layer: concatenate embeds with hidden_states
            embeds = self.input_layernorm(embeds)
            hidden_states, residual = self._residual_norm(hidden_states=hidden_states)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            # Subsequent layers: process hidden_states and residuals only
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "hidden_states": 0,
        "input_embeds": 0,
    }
)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size

        # Get drafter's quantization config
        self.quant_config = get_draft_quant_config(vllm_config)

        eagle_config = getattr(self.config, "eagle_config", None) or {}
        if "use_aux_hidden_state" in eagle_config:
            self.use_aux_hidden_state = eagle_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True
        self.norm_before_fc = bool(
            eagle_config.get(
                "norm_before_fc", getattr(self.config, "norm_before_fc", False)
            )
        )

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            self.num_aux_hidden_states = getattr(
                self.config, "num_aux_hidden_states", None
            )
            if self.num_aux_hidden_states is None:
                eagle_config = getattr(self.config, "eagle_config", None) or {}
                layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
                self.num_aux_hidden_states = len(layer_ids) if layer_ids else 3

            target_hidden_size = getattr(
                self.config, "target_hidden_size", self.config.hidden_size
            )
            self.fc_input_size = target_hidden_size * self.num_aux_hidden_states

            if self.norm_before_fc:
                self.input_norm = RMSNorm(
                    self.fc_input_size,
                    eps=self.config.rms_norm_eps,
                )
            else:
                self.input_norm = None

            use_fc_norm = getattr(self.config, "fc_norm", False)
            if use_fc_norm:
                self.fc_norm = nn.ModuleList(
                    [
                        RMSNorm(target_hidden_size, eps=self.config.rms_norm_eps)
                        for _ in range(self.num_aux_hidden_states)
                    ]
                )
            else:
                self.fc_norm = None

            self.fc = ReplicatedLinear(
                input_size=self.fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )

        self.norm_output = getattr(self.config, "norm_output", False)
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        torch._assert(
            hidden_states.shape[-1] == input_embeds.shape[-1],
            "hidden_states and input_embeds must have the same last dimension",
        )

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)

        # norm_output variant uses the post-norm hidden states.
        aux_output = hidden_states if self.norm_output else hidden_prenorm

        return hidden_states, aux_output

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"midlayer.": "layers.0."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Eagle3LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        # Ensure draft_vocab_size is set
        # default to the base vocab size when absent
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

        # Store target layer count in draft config for
        # proper layer_types indexing in draft models
        self.config.target_layer_count = target_layer_num
        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

        self.use_parallel_drafting = vllm_config.speculative_config.parallel_drafting

        if self.use_parallel_drafting:
            self.register_buffer(
                "mask_hidden",
                torch.zeros(1, self.model.fc_input_size),
                persistent=False,
            )

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
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        # combine multiple auxiliary hidden states returned by eagle3

        if self.model.norm_before_fc:
            hidden_states = self.model.input_norm(hidden_states)

        # `norm_before_fc` adds a single RMSNorm before the FC layer, whereas `fc_norm`
        # applies separate RMSNorms to each chunk of the hidden states.
        if self.model.fc_norm is not None:
            chunks = hidden_states.chunk(self.model.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.model.fc_norm, chunks)],
                dim=-1,
            )

        return self.model.fc(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        includes_mask_hidden = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "mask_hidden" in name:
                # Load mask_hidden directly into buffer
                if not self.use_parallel_drafting:
                    logger.warning(
                        "mask_hidden found in weights but "
                        "model is not configured for parallel drafting. "
                        "Skipping loading mask_hidden."
                    )
                    continue
                self.mask_hidden.copy_(loaded_weight.view(1, -1))
                includes_mask_hidden = True
                continue
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        if not includes_mask_hidden and self.use_parallel_drafting:
            raise ValueError(
                "mask_hidden not found in weights but "
                "model is configured for parallel drafting. "
                "Please provide mask_hidden in the weights."
            )

        skip_substrs = ["mask_hidden"]
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if not self.model.norm_before_fc:
            skip_substrs.append("input_norm.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
