# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.utils import PPMissingLayer


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Share when the draft has no own copy, or its copy matches the target."""

    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    w = draft.weight
    if w.is_cuda and torch.accelerator.get_memory_info(w.device)[0] < w.numel() * 2:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)


def get_target_lm_head(target_model: nn.Module, target_language_model: nn.Module):
    """The target's lm_head — from get_language_model() for
    *ForConditionalGeneration targets, else the top-level module."""
    return getattr(target_language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )


def maybe_share_target_embed(
    draft_model: nn.Module, draft_inner: nn.Module, target_inner: nn.Module
) -> None:
    """Alias the target's input embedding into the drafter when it needs one.

    Under PP the drafter runs on the last stage, where the target's embedding
    exists only because spec_decode_needs_target_embed() asked for it.
    """
    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    if isinstance(target_embed, PPMissingLayer):
        target_embed = None
    # If the target's embedding is LoRA-wrapped, share the underlying base
    # layer. The draft is not part of the LoRA adapter; sharing the wrapper
    # would make the draft run the LoRA embedding kernel with the target's
    # punica metadata (sized for the target's token count), causing an
    # out-of-bounds GPU access during multi-step draft decode.
    if isinstance(target_embed, BaseLayerWithLoRA):
        target_embed = target_embed.base_layer
    draft_embed = getattr(draft_inner, "embed_tokens", None)

    if get_pp_group().world_size > 1 and not hasattr(
        draft_model, "has_own_embed_tokens"
    ):
        # MTP-style drafts load an embedding from the target checkpoint, and the
        # flag that would tell a loaded one from a missing one is EAGLE-only.
        return

    if target_embed is None:
        # hasattr, not draft_embed is not None: DSpark drafts declare
        # embed_tokens as None and wait for the alias, and they are the ones
        # that cannot run without it.
        if hasattr(draft_inner, "embed_tokens") and not getattr(
            draft_model, "has_own_embed_tokens", False
        ):
            raise RuntimeError(
                f"{type(draft_model).__name__} ships no input embedding of its "
                "own and the target's is absent on this pipeline stage, "
                "leaving the drafter nothing to embed its proposals with. "
                "spec_decode_needs_target_embed() must be true for "
                f"{type(target_inner).__name__} so the last stage instantiates "
                "the target's embed_tokens."
            )
        return

    if _should_share(draft_model, "has_own_embed_tokens", draft_embed, target_embed):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    if speculative_config.kv_cache_dtype is not None:
        vllm_config = replace(
            vllm_config,
            cache_config=replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            ),
        )
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = eagle_model.model

    maybe_share_target_embed(eagle_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head

        # MTP layers route logits through layer.shared_head.head, not
        # eagle_model.lm_head, so the per-layer copies need fixing up too.
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    del sh.head
                    sh.head = target_lm_head

    # MTP shares topk_indices_buffer with the target model. We update
    # every module in the draft that holds a buffer reference so that
    # the per-layer indexer and sparse-attention backends all point to
    # the target's buffer.
    if hasattr(target_inner, "topk_indices_buffer"):
        target_buffer = target_inner.topk_indices_buffer
        if target_buffer is not None:
            for _, module in draft_inner.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer

    return eagle_model
