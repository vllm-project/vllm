# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import lcm

from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    VocabParallelEmbedding,
    pad_vocab_size,
)


def test_vocab_padding_is_tp_aligned_for_qwopus_tp3():
    vocab_size = 248320
    tp_size = 3
    padded = pad_vocab_size(vocab_size, lcm(DEFAULT_VOCAB_PADDING_SIZE, tp_size))

    assert padded == 248448
    assert padded % tp_size == 0

    shard_sizes = []
    for rank in range(tp_size):
        shard = VocabParallelEmbedding._get_indices(
            vocab_size_padded=padded,
            org_vocab_size_padded=padded,
            vocab_size=vocab_size,
            org_vocab_size=vocab_size,
            tp_rank=rank,
            tp_size=tp_size,
        )
        shard_sizes.append(shard.num_elements_padded)

    assert shard_sizes == [82816, 82816, 82816]
