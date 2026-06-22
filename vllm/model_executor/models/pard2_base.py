# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared PARD-2 draft-model logic (family-agnostic).

PARD-2 (amd/PARD2-*) fuses a projection of target hidden states into the draft's
input embeddings, then runs a stock decoder stack:

    inputs_embeds = embed(input_ids) + scale * target_proj(reorder(concat(feats)))

``target_proj`` ships separately in ``warp_model.bin``. Only the decoder layers
differ per family: ``llama_pard2.py``/``qwen3_pard2.py`` supply them via
``Pard2ModelBase.build_layers`` and bind the model via ``pard2_model_cls``.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)

# Shared torch.compile dynamic dims for every PARD-2 draft model body.
PARD2_COMPILE_DYNAMIC_ARG_DIMS = {
    "input_ids": 0,
    "positions": -1,
    "hidden_states": 0,
    "input_embeds": 0,
}


class Pard2ModelBase(nn.Module):
    """PARD-2 draft body: a stock decoder stack on fused (embed + projected
    target hidden state) embeddings. Subclasses implement ``build_layers`` for
    their model family and apply ``@support_torch_compile`` themselves."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Family-specific decoder stack. KV-cache layer ids are offset past the
        # target model's layers so the draft does not collide with the verifier.
        self.layers = self.build_layers(vllm_config, start_layer_id, prefix)

        self._init_target_projection(vllm_config, prefix)

        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def build_layers(
        self, vllm_config: VllmConfig, start_layer_id: int, prefix: str
    ) -> nn.ModuleList:
        raise NotImplementedError

    def _init_target_projection(self, vllm_config: VllmConfig, prefix: str) -> None:
        # target_proj: concat(target hidden states) -> draft hidden.
        num_aux = getattr(self.config, "num_aux_hidden_states", 1)
        target_hidden = getattr(
            self.config, "target_hidden_size", self.config.hidden_size
        )
        self.num_aux = num_aux
        self.target_hidden = target_hidden
        self.fc_input_size = target_hidden * num_aux

        # vLLM captures aux hidden states in ascending layer order, but target_proj
        # was trained on concat in pard2_target_layers order (e.g. [-1,-8,-16,-24]);
        # build a permutation to reorder the blocks to match (ref: pard2_infer.py).
        resolved = list(
            getattr(self.config, "eagle_aux_hidden_state_layer_ids", []) or []
        )
        perm = list(range(num_aux))
        if len(resolved) == num_aux and num_aux > 1:
            captured_order = sorted(resolved)  # vLLM ascending capture order
            perm = [captured_order.index(layer) for layer in resolved]
        self._needs_reorder = perm != list(range(num_aux))
        self.register_buffer(
            "target_layer_perm",
            torch.tensor(perm, dtype=torch.long),
            persistent=False,
        )
        self.scale = float(
            getattr(
                self.config,
                "pard2_scale",
                getattr(self.config, "pard_scale", 1.0),
            )
        )
        proj_bias = bool(getattr(self.config, "pard2_proj_bias", False))
        self.target_proj = ReplicatedLinear(
            input_size=self.fc_input_size,
            output_size=self.config.hidden_size,
            bias=proj_bias,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "target_proj"),
            return_bias=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_target_feat(self, target_feat: torch.Tensor) -> torch.Tensor:
        if self._needs_reorder:
            # unflatten/index_select/flatten keeps this shape-agnostic: handles
            # both [num_tokens, fc_in] and a bare [fc_in] warmup vector.
            perm = self.target_layer_perm.to(target_feat.device)
            target_feat = (
                target_feat.unflatten(-1, (self.num_aux, self.target_hidden))
                .index_select(-2, perm)
                .flatten(-2)
            )
        return self.target_proj(target_feat) * self.scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        # `hidden_states` arrives already projected+scaled (see
        # Pard2ForCausalLMMixin.combine_hidden_states): fuse by addition.
        hidden = input_embeds + hidden_states
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden, hidden

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
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Pard2ForCausalLMMixin:
    """Shared PARD-2 ``*ForCausalLM`` logic. Concrete classes set
    ``pard2_model_cls`` (the family ``Pard2ModelBase`` subclass) and inherit
    from the family ``*ForCausalLM`` for interface/isinstance purposes."""

    pard2_model_cls: type[Pard2ModelBase]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self._draft_model_name = vllm_config.speculative_config.draft_model_config.model
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = self.pard2_model_cls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

        # `mask_hidden`: unused zero fallback for the generic static-mask path; PARD-2
        # fills masked parallel slots with repeat-last-feat in the proposer.
        self.use_parallel_drafting = vllm_config.speculative_config.parallel_drafting
        if self.use_parallel_drafting:
            self.register_buffer(
                "mask_hidden",
                torch.zeros(1, self.model.fc_input_size),
                persistent=False,
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
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # Full-vocab head: no draft->target id remap.
        return self.logits_processor(self.lm_head, hidden_states)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.project_target_feat(hidden_states)

    def _load_warp_projection(self) -> bool:
        """Load `target_proj` from the separate ``warp_model.bin`` that ships
        alongside the PARD-2 checkpoint (not in the main safetensors)."""
        import os

        from huggingface_hub import hf_hub_download

        path = self._draft_model_name
        local = os.path.join(path, "warp_model.bin")
        if not os.path.exists(local):
            try:
                local = hf_hub_download(repo_id=path, filename="warp_model.bin")
            except Exception as e:
                logger.warning("PARD-2: could not fetch warp_model.bin: %s", e)
                return False
        warp = torch.load(local, map_location="cpu", weights_only=True)
        proj = self.model.target_proj
        with torch.no_grad():
            proj.weight.copy_(warp["target_proj.weight"].to(proj.weight.dtype))
            if proj.bias is not None and "target_proj.bias" in warp:
                proj.bias.copy_(warp["target_proj.bias"].to(proj.bias.dtype))
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        for name, loaded_weight in weights:
            if "target_proj" in name:
                # may also live in main checkpoint for some exports
                model_weights[
                    "model." + name if not name.startswith("model.") else name
                ] = loaded_weight
                continue
            if "lm_head" not in name:
                name = "model." + name if not name.startswith("model.") else name
            model_weights[name] = loaded_weight
            # Sets has_own_embed_tokens/has_own_lm_head from the checkpoint; PARD-2
            # ships both, so the proposer keeps the draft's copies (not the target's).
            process_eagle_weight(self, name)

        skip_substrs = ["mask_hidden"]
        loader = AutoWeightsLoader(self, skip_prefixes=None, skip_substrs=skip_substrs)
        loaded = loader.load_weights(model_weights.items())

        # target_proj ships separately in warp_model.bin.
        if not any("target_proj" in n for n in model_weights):
            self._load_warp_projection()
        return loaded
