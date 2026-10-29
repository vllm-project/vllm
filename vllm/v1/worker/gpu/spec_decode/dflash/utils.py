# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config


def get_dflash_draft_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    return replace(
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
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
    )


def draft_attention_supports_dcp(draft_vllm_config: VllmConfig, model_tag: str) -> bool:
    """Whether the MLA draft layers can select an attention backend with DCP.

    Runs the same selection as the draft layers, so a draft with no DCP-capable
    backend can fall back to DCP=1 and a replicated KV cache. Sliding-window
    drafts always do: their cache cannot shard by DCP, and NIXL pulls such a
    draft group from a single DCP prefill rank.
    """
    from vllm.compilation.backends import set_model_tag
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.selector import get_attn_backend

    speculative_config = draft_vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    if (
        not draft_model_config.use_mla
        or getattr(draft_model_config.hf_text_config, "sliding_window", None)
        is not None
    ):
        return False
    try:
        with set_current_vllm_config(draft_vllm_config), set_model_tag(model_tag):
            get_attn_backend(
                draft_model_config.get_head_size(),
                draft_model_config.dtype,
                draft_vllm_config.cache_config.cache_dtype,
                use_mla=True,
                num_heads=draft_model_config.get_num_attention_heads(
                    draft_vllm_config.parallel_config
                ),
            )
    except ValueError:
        return False
    return True


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_vllm_config = get_dflash_draft_vllm_config(vllm_config)
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

    return dflash_model
