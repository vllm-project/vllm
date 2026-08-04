# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import json
import os

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config

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
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    # Under PP the target's vocab embedding lives on the first stage, so on the
    # last stage -- where the drafter runs -- there is only a PPMissingLayer to
    # alias. PPMissingLayer.forward returns its input unchanged, which would
    # feed raw int64 token ids into the draft backbone's norm. Load the real
    # table from the target checkpoint instead.
    if isinstance(target_embed, PPMissingLayer):
        target_embed = _load_target_embed_tokens_for_pp(vllm_config)
    if target_embed is not None and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model


def _load_target_embed_tokens_for_pp(vllm_config: VllmConfig) -> nn.Module:
    """Build and load the target's vocab embedding on a non-first PP stage.

    Mirrors the target's own construction (a VocabParallelEmbedding sharded over
    the TP group) and loads the weight from the target checkpoint, which is
    present in full on every node's model directory. Costs
    vocab_size * hidden_size * dtype_size / tp_size bytes, e.g. 0.28 GiB per GPU
    for a 163840 x 7168 bf16 table at TP8.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.platforms import current_platform

    model_config = vllm_config.model_config
    text_config = model_config.hf_text_config
    model_dir = model_config.model

    # Locate the embedding tensor: prefer the shard index, else scan the shards.
    key = None
    shard_path = None
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for name, shard in weight_map.items():
            if name.endswith("embed_tokens.weight"):
                key = name
                shard_path = os.path.join(model_dir, shard)
                break
    else:
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt") as f:
                # safe_open is not a dict and is not directly iterable.
                for name in f.keys():  # noqa: SIM118
                    if name.endswith("embed_tokens.weight"):
                        key = name
                        shard_path = path
                        break
            if key is not None:
                break
    if key is None:
        raise RuntimeError(
            "DSpark under pipeline parallelism needs the target's embed_tokens "
            f"on the last stage, but no *embed_tokens.weight was found in "
            f"{model_dir}"
        )
    assert shard_path is not None

    with torch.device(current_platform.current_device()):
        embed = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            params_dtype=model_config.dtype,
            prefix="dspark_pp.embed_tokens",
        )
    with safe_open(shard_path, framework="pt") as f:
        embed.weight_loader(embed.weight, f.get_tensor(key))
    logger.info(
        "Loaded draft embed_tokens %s from %s (key %s) for PP",
        tuple(embed.weight.shape),
        os.path.basename(shard_path),
        key,
    )
    return embed
