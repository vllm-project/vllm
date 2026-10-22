# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongSpark draft model loading and target-module sharing."""

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config


def load_longspark_model(
    target_model: nn.Module,
    vllm_config: VllmConfig,
) -> nn.Module:
    """Load DraftFreeKV weights and attach the target's shared modules."""
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_vllm_config = speculative_config.apply_draft_overrides(vllm_config)
    draft_load_config = get_pp_safe_draft_load_config(
        get_draft_load_config(draft_vllm_config)
    )
    if draft_load_config is not draft_vllm_config.load_config:
        draft_vllm_config = replace(draft_vllm_config, load_config=draft_load_config)

    with set_model_tag("longspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_model_config,
        )
    draft_model.optimize_for_inference()

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    target_embed = getattr(target_inner, "embed_tokens", None)
    target_norm = getattr(target_inner, "norm", None)
    target_lm_head = get_target_lm_head(target_model, target_language_model)
    target_logits_processor = getattr(target_model, "logits_processor", None)
    if any(
        module is None
        for module in (
            target_embed,
            target_norm,
            target_lm_head,
            target_logits_processor,
        )
    ):
        raise RuntimeError(
            "LongSpark requires target embedding, final norm, lm_head, and "
            "logits processor modules"
        )
    draft_model.set_target_modules(
        target_embed,
        target_norm,
        target_lm_head,
        target_logits_processor,
    )
    return draft_model
