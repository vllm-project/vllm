# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def _validate_qwen3_vl_weight_contract(
    draft_model: nn.Module,
    draft_model_config: ModelConfig,
    target_vocab_size: int,
) -> None:
    if "Qwen3VLDSparkModel" not in (draft_model_config.architectures or []):
        return

    draft_input_vocab_size = draft_model_config.get_vocab_size()
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_input_vocab_size
    )
    if draft_input_vocab_size > target_vocab_size and not getattr(
        draft_model, "has_own_embed_tokens", False
    ):
        raise ValueError(
            "Qwen3-VL DSpark checkpoints with an expanded input vocabulary "
            "must include embed_tokens weights."
        )
    if draft_output_vocab_size != target_vocab_size and not getattr(
        draft_model, "has_own_lm_head", False
    ):
        raise ValueError(
            "Qwen3-VL DSpark checkpoints with a reduced output vocabulary "
            "must include lm_head weights."
        )
    if draft_output_vocab_size != target_vocab_size and not getattr(
        draft_model, "has_own_draft_id_mapping", False
    ):
        raise ValueError(
            "Qwen3-VL DSpark checkpoints with a reduced output vocabulary "
            "must include a d2t token mapping."
        )


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

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
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()
    draft_input_vocab_size = draft_model_config.get_vocab_size()
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_input_vocab_size
    )
    _validate_qwen3_vl_weight_contract(
        draft_model, draft_model_config, target_vocab_size
    )

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if (
        target_embed is not None
        and draft_input_vocab_size <= target_vocab_size
        and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed
        )
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
