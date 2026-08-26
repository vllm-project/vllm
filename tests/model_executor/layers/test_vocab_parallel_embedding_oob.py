# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: VocabParallelEmbedding must tolerate out-of-range ids at
tp_size == 1.

Speculative decoding feeds the target model a padded verification query block
whose not-yet-filled / rejected draft slots carry the ``-1`` sentinel. The
``tp_size > 1`` branch has always masked out-of-range ids before the gather;
the single-GPU branch passed them straight into ``F.embedding``, tripping a
device-side index assert (``indexSelectSmallIndex: srcIndex <
srcSelectDimSize``) on the first spec step of every single-GPU spec-decode
config, independent of KV-cache dtype.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


@pytest.fixture(scope="module", autouse=True)
def _single_gpu_parallel_state():
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)
        yield


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tp1_out_of_range_ids_are_masked(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    vocab_size, hidden = 128, 16
    with set_current_vllm_config(VllmConfig()):
        layer = VocabParallelEmbedding(vocab_size, hidden).to(device)

    # Valid ids, the spec-decode -1 sentinel, and an over-range id.
    input_ = torch.tensor([0, 5, -1, vocab_size, vocab_size - 1], device=device)
    out = layer(input_)

    # No device-side assert, and the out-of-range rows are zeroed --
    # mirroring the tp > 1 semantics (those positions are discarded
    # downstream by rejection sampling).
    assert out.shape == (5, hidden)
    assert torch.all(out[2] == 0)
    assert torch.all(out[3] == 0)

    # Valid-id rows are the plain embedding lookup, unchanged by the mask.
    ref = layer.weight[input_[[0, 1, 4]]]
    torch.testing.assert_close(out[[0, 1, 4]], ref)
