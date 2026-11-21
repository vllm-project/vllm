# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform


@torch.inference_mode()
def _check_tied_embedding_prepack(model):
    embedding = next(
        module
        for module in model.modules()
        if isinstance(module, VocabParallelEmbedding)
        and not isinstance(module, ParallelLMHead)
    )
    head = next(
        module for module in model.modules() if isinstance(module, ParallelLMHead)
    )
    assert head.weight is embedding.weight
    assert getattr(embedding, "cpu_linear", None) is None
    assert head.cpu_linear is not None
    inputs = torch.zeros(1, head.embedding_dim, dtype=head.weight.dtype)
    inputs[0, 0] = 1
    actual = head.quant_method.apply(head, inputs)
    torch.testing.assert_close(actual, head.weight[:, 0].unsqueeze(0), rtol=0, atol=0)


def _reload_changed_embedding(worker):
    from vllm.model_executor.model_loader import get_model_loader

    loader = get_model_loader(worker.vllm_config.load_config)
    changed = []

    def weights():
        for name, weight in loader.get_all_weights(
            worker.vllm_config.model_config, worker.model_runner.get_model()
        ):
            if name.endswith("embed_tokens.weight"):
                changed.append(name)
                weight = -weight
            yield name, weight

    worker.reload_weights(weights_iterator=weights())
    assert len(changed) == 1


@pytest.mark.cpu_model
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU GEMM preprocessing")
def test_tied_embedding_prepack_after_reload(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model_name = "Qwen/Qwen2.5-0.5B"
    with vllm_runner(
        model_name,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        kv_cache_memory_bytes=128 * 1024**2,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    ) as llm:
        llm.apply_model(_check_tied_embedding_prepack)
        # One request per phase keeps the batch shape fixed across reloads.
        prompts = ["The capital of France is"]
        original = llm.generate_greedy_logprobs(prompts, max_tokens=8, num_logprobs=5)
        llm.collective_rpc(_reload_changed_embedding)
        llm.apply_model(_check_tied_embedding_prepack)
        changed = llm.generate_greedy_logprobs(prompts, max_tokens=8, num_logprobs=5)
        assert changed != original
        llm.collective_rpc("reload_weights", kwargs={"weights_path": model_name})
        llm.apply_model(_check_tied_embedding_prepack)
        reloaded = llm.generate_greedy_logprobs(prompts, max_tokens=8, num_logprobs=5)
        assert reloaded == original
