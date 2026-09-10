# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    # The engine resolves the physical KV-cache layout after model construction.
    # Keep the shared CacheConfig so the draft attention implementations observe
    # that later update, while exposing the draft-only dtype during construction.
    cache_config = vllm_config.cache_config
    target_cache_dtype = cache_config.cache_dtype
    try:
        if speculative_config.kv_cache_dtype is not None:
            cache_config.cache_dtype = speculative_config.kv_cache_dtype
        draft_vllm_config = replace(
            vllm_config,
            attention_config=replace(
                vllm_config.attention_config,
                use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
                backend=speculative_config.attention_backend,
            ),
            cache_config=cache_config,
        )
        with set_model_tag("dflash_head"):
            dflash_model = get_model(
                vllm_config=draft_vllm_config, model_config=draft_model_config
            )
    finally:
        cache_config.cache_dtype = target_cache_dtype

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
