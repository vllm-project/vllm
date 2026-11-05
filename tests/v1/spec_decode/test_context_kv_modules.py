# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With `prefill_context_kv_only`, a DFlash-based drafter frees the weights
outside its draft model's `context_kv_modules()`. The context K/V it stores
must not change."""

import pytest
import torch
from torch import nn

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import supports_context_kv_only
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode import init_speculator

NUM_CONTEXT_TOKENS = 37


def _load_speculator(target: str, draft: str, method: str, k: int):
    vllm_config = EngineArgs(
        model=target,
        speculative_config={
            "method": method,
            "model": draft,
            "num_speculative_tokens": k,
            "prefill_context_kv_only": True,
        },
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=1024,
        max_num_seqs=8,
        enforce_eager=True,
    ).create_engine_config()
    model_config = vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    hidden_size = model_config.get_hidden_size()
    device = torch.device("cuda")
    with set_current_vllm_config(vllm_config):
        speculator = init_speculator(vllm_config, device, req_states=None)
        # Drafts share only the embedding and the LM head with the target.
        target_model = nn.Module()
        target_model.model = nn.Module()
        with device:
            target_model.model.embed_tokens = VocabParallelEmbedding(
                vocab_size, hidden_size, params_dtype=model_config.dtype
            )
            target_model.lm_head = ParallelLMHead(
                vocab_size, hidden_size, params_dtype=model_config.dtype
            )
        speculator.load_model(target_model)
    return speculator, target_model


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    ("target", "draft", "method", "k"),
    [
        # DFlashQwen3Model's list, which DSpark, DFlash2 and XPress return too.
        ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", "dflash", 16),
        # A draft model with its own context K/V path and list.
        ("google/gemma-4-12B-it", "deepseek-ai/dspark_gemma4_12b_block7", "dspark", 7),
    ],
)
def test_freeing_keeps_context_kv(dist_init, monkeypatch, target, draft, method, k):
    speculator, target_model = _load_speculator(target, draft, method, k)
    model = speculator.model
    attn_layers = model.model._attn_layers

    def context_kv() -> list[torch.Tensor]:
        written: list[torch.Tensor] = []
        for attn in attn_layers:
            monkeypatch.setattr(
                attn.impl,
                "do_kv_cache_update",
                lambda layer, key, value, kv_cache, slots: written.extend(
                    (key.clone(), value.clone())
                ),
            )
        generator = torch.Generator("cuda").manual_seed(0)
        target_states = torch.randn(
            NUM_CONTEXT_TOKENS,
            model.model.fc.input_size,
            generator=generator,
            device="cuda",
            dtype=speculator.dtype,
        )
        positions = torch.arange(NUM_CONTEXT_TOKENS, device="cuda")
        model.precompute_and_store_context_kv(
            model.combine_hidden_states(target_states), positions, positions
        )
        return written

    with torch.inference_mode():
        before = context_kv()
        speculator.free_unused_weights(target_model)
        assert any(param.numel() == 0 for param in model.parameters())
        after = context_kv()

    assert len(before) == len(after) == 2 * len(attn_layers)
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_opt_in_is_per_class():
    # A subclass's context K/V path may read other weights.
    class Draft:
        def context_kv_modules(self) -> list[nn.Module]:
            return []

    class Subclass(Draft):
        pass

    assert supports_context_kv_only(Draft())
    assert not supports_context_kv_only(Subclass())
