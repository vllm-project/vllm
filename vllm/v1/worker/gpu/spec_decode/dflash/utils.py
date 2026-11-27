# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)


def with_draft_dtype(vllm_config: VllmConfig, dtype: torch.dtype) -> VllmConfig:
    """Return `vllm_config` with its model_config reporting the draft's dtype.

    The draft is built, and its "auto" KV cache sized, from
    `vllm_config.model_config`, which is the target's. Swapping in the draft
    ModelConfig would also change the target facts the draft reads (layer
    count, vocab size), and replace() would revalidate the target config, so
    only the dtype is overridden, on shallow copies.
    """
    if vllm_config.model_config.dtype == dtype:
        return vllm_config
    model_config = copy.copy(vllm_config.model_config)
    model_config.dtype = dtype
    vllm_config = copy.copy(vllm_config)
    vllm_config.model_config = model_config
    return vllm_config


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
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_draft_load_config(vllm_config),
    )
    draft_vllm_config = speculative_config.apply_draft_eplb_overrides(draft_vllm_config)
    # Build the draft's weights and KV cache in its own dtype
    # (SpeculativeConfig.dtype); a no-op when it matches the target's.
    draft_vllm_config = with_draft_dtype(draft_vllm_config, draft_model_config.dtype)
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

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
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        )
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head
        # The shared head takes the target's activation and head dtypes, also
        # when the draft runs in its own dtype (SpeculativeConfig.dtype).
        dflash_model.lm_head_dtype = vllm_config.model_config.dtype
        for module in dflash_model.modules():
            if isinstance(module, LogitsProcessor):
                module.head_dtype = vllm_config.model_config.head_dtype

    return dflash_model
