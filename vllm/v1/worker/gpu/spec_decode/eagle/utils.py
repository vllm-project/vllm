# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=vllm_config, model_config=draft_model_config
        )

    # Share target embeddings when the draft checkpoint does not include
    # its own vocab embedding table.
    share_embeddings = True
    if hasattr(eagle_model, "has_own_embed_tokens"):
        share_embeddings = not eagle_model.has_own_embed_tokens
    if share_embeddings:
        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        inner_model = getattr(target_language_model, "model", None)
        target_embed_tokens = None
        if inner_model is not None:
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
        if target_embed_tokens is not None and hasattr(eagle_model, "model"):
            if hasattr(eagle_model.model, "embed_tokens"):
                del eagle_model.model.embed_tokens
            eagle_model.model.embed_tokens = target_embed_tokens

    # Only share target lm_head when the draft model does not own one.
    share_lm_head = True
    if hasattr(eagle_model, "has_own_lm_head"):
        share_lm_head = not eagle_model.has_own_lm_head
    if share_lm_head and hasattr(target_model, "lm_head"):
        if hasattr(eagle_model, "lm_head"):
            del eagle_model.lm_head
        eagle_model.lm_head = target_model.lm_head

    return eagle_model
