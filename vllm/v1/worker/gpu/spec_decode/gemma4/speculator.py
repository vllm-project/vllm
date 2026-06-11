# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP (Multi-Token Prediction) speculator for speculative decoding.

The Gemma4 assistant model runs all decoder layers per draft step
(producing one token), and all its attention layers share KV cache
with the target model via cross-model KV sharing.
"""

from collections import defaultdict

import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)

logger = init_logger(__name__)


class Gemma4Speculator(AutoRegressiveSpeculator):
    @property
    def advance_draft_positions(self) -> bool:
        # Gemma4 MTP is Q-only and reads K/V from the target's existing cache.
        # No new KV slots are written, so positions and seq_lens stay fixed.
        return False

    @property
    def model_returns_tuple(self) -> bool:
        # forward() returns (draft_hidden_states, backbone_hidden_states).
        # The proposer uses draft_hidden_states for compute_logits and
        # backbone_hidden_states for the hidden-state feedback buffer.
        return True

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            draft_model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        self._setup_gemma4_kv_sharing(draft_model, target_attn_layer_names)
        self._share_embeddings(draft_model, target_model)
        return draft_model

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Preserve the target's forced TRITON_ATTN backend for draft layers.

        Gemma4 forces TRITON_ATTN due to heterogeneous head dimensions
        (head_dim=256 sliding, global_head_dim=512 full). The base class
        resets attention_config.backend to None for draft models, causing
        sliding layers to fall back to FLASH_ATTN which cannot handle
        KV-shared cache. Override to carry the target's backend through.
        """
        draft_model_config = self.speculative_config.draft_model_config
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=draft_model_config,
        )
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is not None:
            draft_vllm_config = replace(
                draft_vllm_config,
                attention_config=replace(
                    draft_vllm_config.attention_config,
                    backend=target_backend,
                ),
            )
        return draft_vllm_config

    def _setup_gemma4_kv_sharing(
        self,
        model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> None:
        """Wire draft layers to share KV with the target model.

        Each draft decoder layer is mapped to the last non-KV-shared
        target layer of the same attention type (sliding or full).
        """
        draft_config = self.speculative_config.draft_model_config.hf_config
        draft_text_config = draft_config.get_text_config()
        target_config = self.vllm_config.model_config.hf_config
        target_text_config = target_config.get_text_config()
        target_layer_types = getattr(target_text_config, "layer_types", [])

        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            return

        target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
        num_non_shared = len(target_layer_types) - target_num_kv_shared
        type_to_target_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lt in enumerate(target_layer_types[:num_non_shared]):
            type_to_target_indices[lt].append(idx)

        target_prefix = "model.layers"
        for name in target_attn_layer_names:
            if ".layers." in name:
                target_prefix = name.split(".layers.")[0] + ".layers"
                break

        draft_layer_types = getattr(draft_text_config, "layer_types", [])
        for draft_idx, layer in enumerate(model.model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            attn = getattr(layer.self_attn, "attn", None)
            if attn is None:
                continue

            draft_layer_type = (
                draft_layer_types[draft_idx]
                if draft_idx < len(draft_layer_types)
                else "full_attention"
            )
            candidates = type_to_target_indices.get(draft_layer_type, [])
            if not candidates:
                logger.warning(
                    "No target layer of type '%s' for draft layer %d",
                    draft_layer_type,
                    draft_idx,
                )
                continue

            target_idx = candidates[-1]
            target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"
            attn.kv_sharing_target_layer_name = target_layer_name
            logger.info(
                "Gemma4 MTP: draft layer %d (%s) -> %s",
                draft_idx,
                draft_layer_type,
                target_layer_name,
            )

    def _share_embeddings(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
    ) -> None:
        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        if get_pp_group().world_size == 1:
            target_embed = getattr(target_language_model.model, "embed_tokens", None)
            if target_embed is not None:
                del draft_model.model.embed_tokens
                draft_model.model.embed_tokens = target_embed
