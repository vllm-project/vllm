# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``get_top_tokens`` on the Qwen4Exp AMD MTP drafter.

The speculator opts into ``use_local_argmax_reduction`` by probing the draft
model for ``get_top_tokens``, and expects it to select the same greedy draft
token the full-vocab argmax (``compute_logits``) would.
"""

from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "get_top_tokens is implemented on the ROCm Qwen4Exp AMD MTP drafter only.",
        allow_module_level=True,
    )

from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP  # noqa: E402

HIDDEN_SIZE = 16
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"


def _make_mtp(vocab_size: int) -> Qwen4ExpMTP:
    # A single vocab shard, so the reduction returns the local argmax without
    # touching the process group.
    with (
        mock.patch(f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=0),
        mock.patch(
            f"{_EMBEDDING}.get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        lm_head = ParallelLMHead(vocab_size, HIDDEN_SIZE, params_dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        lm_head.weight.normal_(0.0, 0.02, generator=generator)

    mtp = object.__new__(Qwen4ExpMTP)
    mtp.lm_head = lm_head
    mtp.logits_processor = LogitsProcessor(vocab_size)
    return mtp


def _hidden_states(num_tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.float32, generator=generator
    )
    return hidden_states.to(dtype=torch.bfloat16)


def test_get_top_tokens_matches_compute_logits():
    """Same draft token as the path taken when the reduction is off.

    A typo delegating to the wrong head or tensor (e.g. ``self.model``
    instead of ``self.lm_head``) would silently return a different draft
    token instead of failing, which this comparison catches.
    """
    mtp = _make_mtp(vocab_size=256)

    for num_tokens in (1, 16, 128):
        hidden_states = _hidden_states(num_tokens, seed=11)

        logits = mtp.compute_logits(hidden_states)
        top = mtp.get_top_tokens(hidden_states)

        assert top.dtype == torch.int64
        assert torch.equal(top, logits.argmax(dim=-1))


def test_get_top_tokens_honors_padded_vocab():
    """Padding entries above ``org_vocab_size`` must never win the argmax."""
    vocab_size = 200  # not a multiple of the default vocab padding
    mtp = _make_mtp(vocab_size)
    assert mtp.lm_head.shard_indices.num_org_vocab_padding > 0

    hidden_states = _hidden_states(32, seed=13)
    logits = mtp.compute_logits(hidden_states)
    top = mtp.get_top_tokens(hidden_states)

    assert torch.equal(top, logits.argmax(dim=-1))
    assert (top < vocab_size).all()
