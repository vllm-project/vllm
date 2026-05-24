# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gumiho speculative drafter (EAGLE-like transformer + parallel MLP heads).

Reference:
    Li et al., "Gumiho: A Hybrid Architecture to Prioritize Early Tokens in
    Speculative Decoding", ICML 2025. https://arxiv.org/abs/2503.10135
    Code: https://github.com/AMD-AIG-AIMA/Gumiho

Gumiho is a hybrid speculative decoding drafter:

* The first two speculative tokens are produced autoregressively by a small
  transformer draft head (a 2-layer Llama variant in the released checkpoint),
  exactly like EAGLE.
* Every additional speculative token is produced *in parallel* by independent
  MLP heads, conditioned on the embeddings and hidden states of the first two
  draft tokens.

The transformer head is implemented in :class:`GumihoLlamaModel`, while the
outer :class:`GumihoLlamaForCausalLM` owns the MLP heads and the LM head, and
exposes :meth:`generate_mlp_draft_token_ids` which is called by
:class:`vllm.v1.spec_decode.gumiho.GumihoProposer` after the first two
sequential drafting steps.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

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
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.llama_eagle import LlamaDecoderLayer

from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    is_pp_missing_parameter,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class GumihoResBlock(nn.Module):
    """SiLU-linear block with optional residual projection.

    Mirrors the ``ResBlock`` used by the released Gumiho training code.
    The linear weight is zero-initialised so the freshly constructed block is
    equivalent to the identity, which matches the original initialisation
    scheme.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, dtype=dtype)
        if input_size != output_size:
            self.res_connection = nn.Linear(input_size, output_size, dtype=dtype)
        else:
            self.res_connection = nn.Identity()
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_connection(x) + self.act(self.linear(x))


class GumihoNoResBlock(nn.Module):
    """SiLU-linear block without residual connection (Gumiho ``noResBlock``)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, dtype=dtype)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


@support_torch_compile
class GumihoLlamaModel(nn.Module):
    """Transformer draft head of Gumiho.

    Generates the first two speculative tokens autoregressively. The input is
    the concatenation of the target token embedding and the previous hidden
    state, projected back to ``hidden_size`` by ``self.fc``.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size

        self.quant_config = get_draft_quant_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    vllm_config,
                    i == 0,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                    config=self.config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.fc = ReplicatedLinear(
            input_size=self.config.hidden_size * 2,
            output_size=self.config.hidden_size,
            bias=getattr(self.config, "gumiho_fc_bias", False),
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

        self.add_para_norm = getattr(self.config, "add_para_norm", False)
        if self.add_para_norm:
            self.enorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            self.hnorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def fuse_inputs(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate token embedding and previous hidden, project to hidden.

        Equivalent to: ``fc(cat([embed(input_ids), hidden_states], dim=-1))``,
        with optional separate RMSNorm on each component when
        ``add_para_norm=True`` (matches the training-time option).
        """
        input_embeds = self.embed_tokens(input_ids)
        if self.add_para_norm:
            fused = torch.cat(
                [self.enorm(input_embeds), self.hnorm(hidden_states)],
                dim=-1,
            )
        else:
            fused = torch.cat((input_embeds, hidden_states), dim=-1)
        return self.fc(fused)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.fuse_inputs(input_ids, hidden_states)
        # First token in a sequence has no prior hidden state to fuse with.
        hidden_states = hidden_states.masked_fill((positions == 0).unsqueeze(-1), 0)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states = hidden_states + residual
        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
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
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GumihoLlamaForCausalLM(LlamaForCausalLM):
    """Gumiho drafter: transformer head + parallel MLP heads + LM head.

    The transformer head produces the first two speculative tokens through
    :class:`GumihoLlamaModel`. For each speculative token beyond the second,
    one parallel MLP head in ``self.mlp`` is invoked via
    :meth:`generate_mlp_draft_token_ids`. ``supports_eagle = True`` tells the
    weight loader to share the target model's embeddings / LM head when they
    are missing from the draft checkpoint (the standard EAGLE convention).
    """

    supports_eagle = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = GumihoLlamaModel(
            vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
        )

        hidden_size = self.config.hidden_size
        dtype = vllm_config.model_config.dtype
        # One MLP head per extra speculative token beyond the first two
        # transformer-generated ones. Each head matches the released Gumiho
        # checkpoint architecture (one NoResBlock + five ResBlocks).
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    GumihoNoResBlock(hidden_size * 2, hidden_size, dtype=dtype),
                    GumihoResBlock(hidden_size, hidden_size, dtype=dtype),
                    GumihoResBlock(hidden_size, hidden_size, dtype=dtype),
                    GumihoResBlock(hidden_size, hidden_size, dtype=dtype),
                    GumihoResBlock(hidden_size, hidden_size, dtype=dtype),
                    GumihoResBlock(hidden_size, hidden_size, dtype=dtype),
                )
                for _ in range(
                    max(0, vllm_config.speculative_config.num_speculative_tokens - 2)
                )
            ]
        )

        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            hidden_size,
            params_dtype=dtype,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multimodal inputs yet."
            )
        return self.model(input_ids, positions, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        assert logits is not None
        return logits

    def generate_mlp_draft_token_ids(
        self,
        draft_token_ids: list[torch.Tensor],
        draft_hidden_states: list[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor | None:
        """Generate ``num_tokens`` draft tokens with the parallel MLP heads.

        Inputs are the first two draft token id tensors and their hidden
        states, as produced by the transformer draft head. The two
        ``(input_ids, hidden_states)`` pairs are fused with
        :meth:`GumihoLlamaModel.fuse_inputs` and concatenated along the last
        dim, then fed into each MLP head independently. Returns a tensor of
        shape ``[batch_size, num_tokens]`` of argmax token ids, or ``None``
        when there is nothing to generate (no MLP heads configured, or
        insufficient draft history).
        """
        if num_tokens <= 0 or not self.mlp:
            return None
        if len(draft_token_ids) < 2 or len(draft_hidden_states) < 2:
            return None

        mlp_inputs = []
        for token_ids, hidden_states in zip(
            draft_token_ids[:2], draft_hidden_states[:2]
        ):
            mlp_inputs.append(self.model.fuse_inputs(token_ids, hidden_states))
        mlp_input = torch.cat(mlp_inputs, dim=-1)

        next_token_ids = []
        for block in self.mlp[:num_tokens]:
            hidden_states = block(mlp_input)
            logits = self.compute_logits(hidden_states)
            next_token_ids.append(logits.argmax(dim=-1))
        if not next_token_ids:
            return None
        return torch.stack(next_token_ids, dim=1)

    def _has_configured_mlp_head(self, name: str) -> bool:
        """Return True if ``name`` (e.g. ``mlp.3.0.linear.weight``) targets
        an MLP head index that is within the configured ``num_speculative_tokens``.

        Released Gumiho checkpoints contain weights for the maximum number of
        MLP heads supported by training. When the user requests fewer
        speculative tokens, the extra heads simply don't exist in this module,
        so we skip their weights instead of erroring out.
        """
        parts = name.split(".", 2)
        if len(parts) < 2:
            return False
        try:
            head_index = int(parts[1])
        except ValueError:
            return False
        return head_index < len(self.mlp)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Split checkpoint weights into two buckets:
        #   * ``model_weights`` -> the inner transformer draft head + fc.
        #   * ``outer_weights`` -> mlp heads + lm_head, owned by this class.
        # We support a few legacy prefix layouts that show up in published
        # Gumiho checkpoints (``model.model.*``, ``model.lm_head.*`` etc).
        model_weights: list[tuple[str, torch.Tensor]] = []
        outer_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in weights:
            if name == "token_map":
                logger.warning_once("Ignoring unsupported Gumiho token_map weight.")
                continue
            if name.startswith("model.lm_head."):
                name = name.removeprefix("model.")
                outer_weights.append((name, loaded_weight))
                process_eagle_weight(self, name)
            elif name.startswith("model.model."):
                inner_name = name.removeprefix("model.model.")
                model_weights.append((inner_name, loaded_weight))
                process_eagle_weight(self, f"model.{inner_name}")
            elif name.startswith("model."):
                inner_name = name.removeprefix("model.")
                model_weights.append((inner_name, loaded_weight))
                process_eagle_weight(self, name)
            elif name.startswith("fc."):
                model_weights.append((name, loaded_weight))
            elif name.startswith("mlp."):
                if not self._has_configured_mlp_head(name):
                    logger.warning_once(
                        "Skipping Gumiho MLP weights beyond configured "
                        "num_speculative_tokens."
                    )
                    continue
                outer_weights.append((name, loaded_weight))
                process_eagle_weight(self, name)
            elif name.startswith("lm_head."):
                outer_weights.append((name, loaded_weight))
                process_eagle_weight(self, name)
            else:
                model_weights.append((name, loaded_weight))

        if model_weights:
            self.model.load_weights(model_weights)
        if outer_weights:
            loader = AutoWeightsLoader(self, skip_prefixes=None)
            loader.load_weights(outer_weights)


# Back-compat alias: older Gumiho checkpoints set ``architectures=["GumihoModel"]``
# or ``architectures=["Gumiho"]``, which we map to the same class via the
# speculative model registry below.
GumihoModel = GumihoLlamaForCausalLM
