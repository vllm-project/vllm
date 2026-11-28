# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`get_top_tokens` on the AMD Qwen4Exp MTP drafter.

The speculator opts into `use_local_argmax_reduction` by probing the draft
model for `get_top_tokens`, and expects it to select the same greedy draft
token as `compute_logits` followed by a full-vocab argmax.
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

DEVICE_TYPE = current_platform.device_type
HIDDEN_SIZE = 16
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"
_LOGITS = "vllm.model_executor.layers.logits_processor"


def _make_head(vocab_size: int, tp_rank: int, tp_size: int) -> ParallelLMHead:
    with (
        mock.patch(
            f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=tp_rank
        ),
        mock.patch(
            f"{_EMBEDDING}.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
    ):
        head = ParallelLMHead(vocab_size, HIDDEN_SIZE, params_dtype=torch.bfloat16)
    return head.to(DEVICE_TYPE)


def _make_mtp(lm_head: ParallelLMHead, vocab_size: int) -> Qwen4ExpMTP:
    # Only the attributes the two methods read.
    mtp = object.__new__(Qwen4ExpMTP)
    torch.nn.Module.__init__(mtp)
    mtp.lm_head = lm_head
    mtp.logits_processor = LogitsProcessor(vocab_size)
    return mtp


def _hidden_states(num_tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, generator=generator
    ).to(DEVICE_TYPE)


def _assert_selects_max_logit(top: torch.Tensor, logits: torch.Tensor) -> None:
    """The selected token must carry the row's maximum logit (bf16 rows can
    tie, so compare the selected values rather than the indices)."""
    assert top.dtype == torch.int64
    assert top.shape == (logits.shape[0],)
    selected = logits.gather(1, top.unsqueeze(1)).squeeze(1)
    assert torch.equal(selected, logits.max(dim=-1).values)


def test_get_top_tokens_matches_compute_logits(default_vllm_config):
    """Delegates to the same head and tensor as `compute_logits`."""
    vocab_size = 256
    lm_head = _make_head(vocab_size, tp_rank=0, tp_size=1)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.randn(
                lm_head.weight.shape, generator=torch.Generator().manual_seed(7)
            )
            * 0.02
        )
    mtp = _make_mtp(lm_head, vocab_size)

    for num_tokens in (1, 16):
        hidden_states = _hidden_states(num_tokens, seed=11)
        logits = mtp.compute_logits(hidden_states)
        assert logits is not None
        _assert_selects_max_logit(mtp.get_top_tokens(hidden_states), logits)


def test_get_top_tokens_tp2_padded_vocab(default_vllm_config):
    """Two vocab shards through the rank reduction, the last one padded.

    500 tokens pad to two 256-row shards, so rank 1 holds 244 real rows and 12
    padding rows. The padding rows get the largest weights: if they were not
    masked, they would win every row's argmax.
    """
    vocab_size, tp_size = 500, 2
    generator = torch.Generator().manual_seed(7)
    weight = (torch.randn(vocab_size, HIDDEN_SIZE, generator=generator) * 0.02).to(
        DEVICE_TYPE, torch.bfloat16
    )

    full_head = _make_head(vocab_size, tp_rank=0, tp_size=1)
    shards = [_make_head(vocab_size, tp_rank=r, tp_size=tp_size) for r in range(2)]
    assert shards[0].shard_indices.num_org_vocab_padding == 0
    assert shards[1].shard_indices.num_org_vocab_padding > 0
    with torch.no_grad():
        full_head.weight.zero_()
        full_head.weight[:vocab_size] = weight
        for head in shards:
            idx = head.shard_indices
            num_real = idx.org_vocab_end_index - idx.org_vocab_start_index
            head.weight.fill_(10.0)
            head.weight[:num_real] = weight[
                idx.org_vocab_start_index : idx.org_vocab_end_index
            ]

    # Positive, so the 10.0 padding rows score above every real row.
    hidden_states = _hidden_states(32, seed=13).abs()
    logits = _make_mtp(full_head, vocab_size).compute_logits(hidden_states)
    assert logits is not None
    logits = logits[:, :vocab_size]

    # Each rank's (value, index) pair, in rank order, as the all-gather
    # would return it.
    pairs: list[torch.Tensor] = []

    def record(local_pair: torch.Tensor, dim: int) -> torch.Tensor:
        pairs.append(local_pair)
        return local_pair.repeat(1, tp_size)

    with mock.patch(f"{_LOGITS}.tensor_model_parallel_all_gather", side_effect=record):
        for head in shards:
            _make_mtp(head, vocab_size).get_top_tokens(hidden_states)
    gathered = torch.cat(pairs, dim=-1)

    for head in shards:
        with mock.patch(
            f"{_LOGITS}.tensor_model_parallel_all_gather", return_value=gathered
        ):
            top = _make_mtp(head, vocab_size).get_top_tokens(hidden_states)
        assert (top < vocab_size).all()
        _assert_selects_max_logit(top, logits)
