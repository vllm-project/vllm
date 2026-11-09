# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`VLLM_REPLICATE_EMBED` reaches the ROCm DeepSeek-V4.1 input embedding.

The embedding is built before the Engram layout and the decoder layers, so only
the head of `DeepseekV4Model.__init__` runs: the test stops it right after the
embedding and checks the table each of two TP ranks would hold.
"""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

VOCAB, HIDDEN, TP = 1024, 256, 2


class _StopAfterEmbedding(Exception):
    pass


def _build_embedding(monkeypatch, tp_rank: int):
    from vllm.model_executor.layers import vocab_parallel_embedding
    from vllm.models.deepseek_v41.amd import model as amd_model

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: TP
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(
        amd_model, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
    )
    monkeypatch.setattr(amd_model, "supports_mhc_overlap", lambda _: False)

    def stop(_config):
        raise _StopAfterEmbedding

    monkeypatch.setattr(amd_model.EngramLayout, "from_config", stop)

    hf_config = SimpleNamespace(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        hc_eps=1e-6,
        hc_mult=4,
        rms_norm_eps=1e-6,
        index_topk=8,
        tie_word_embeddings=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=None,
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )
    model = amd_model.DeepseekV4Model.__new__(amd_model.DeepseekV4Model)
    with pytest.raises(_StopAfterEmbedding):
        model.__init__(vllm_config=vllm_config, prefix="model")
    return model.embed_tokens


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("replicate", [False, True], ids=["sharded", "replicated"])
def test_embedding_follows_replicate_embed(
    replicate, tp_rank, monkeypatch, default_vllm_config
) -> None:
    monkeypatch.setenv("VLLM_REPLICATE_EMBED", "1" if replicate else "0")
    embed = _build_embedding(monkeypatch, tp_rank)

    assert embed.weight.shape[1] == HIDDEN
    if replicate:
        # The full table on every rank: a local gather, no all-reduce.
        assert embed.tp_size == 1
        assert embed.shard_indices.org_vocab_start_index == 0
        assert embed.shard_indices.org_vocab_end_index == VOCAB
    else:
        # Each rank holds its share of the vocab.
        assert embed.tp_size == TP
        shard = VOCAB // TP
        assert embed.shard_indices.org_vocab_start_index == tp_rank * shard
        assert embed.shard_indices.org_vocab_end_index == (tp_rank + 1) * shard
