# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's vocab-sharded DSpark Markov chain (ROCm) against vLLM's loop.

Each TP rank runs `MarkovVocabShard.sample_sequential` on its vocabulary
shard of the base logits and of markov_w2, exchanging the per-block keys
through the TP all-gather. vLLM's `DSparkSpeculator._sample_sequential` runs
on the full vocabulary on every rank. With a bias that does not depend on the
order of its sum (weights k / 16), both give the same draft tokens and, for
probabilistic drafts, the same draft logits cache bits.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tensor_model_parallel_rank,
)
from vllm.platforms import current_platform

from ..utils import multi_gpu_test
from .eplb_utils import distributed_run, set_env_vars_and_device

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark ROCm draft path"
)


class _ShardedLogits:
    """The draft logits processor on this rank's vocabulary shard."""

    def __init__(self, base: torch.Tensor, start: int, end: int) -> None:
        self.base, self.start, self.end = base, start, end

    def __call__(self, lm_head, hidden, skip_gather=False):
        assert skip_gather
        return self.base[: hidden.shape[0], self.start : self.end].contiguous()

    def _apply_head(self, head, x, bias):
        return head.quant_method.apply(head, x, bias=bias)


def _sharded(spec, rank: int, world_size: int, greedy: bool):
    """The speculator's model as the vocab-sharded chain sees it on `rank`."""
    from tests.kernels.models.amd.test_muse_spark_fused_markov import VOCAB

    model = spec.model
    shard = VOCAB // world_size
    start, end = rank * shard, (rank + 1) * shard
    model.config = SimpleNamespace(draft_vocab_size=VOCAB)
    model.lm_head = SimpleNamespace(
        weight=model.lm_head.weight,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=end
        ),
    )
    model.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_sample_method="greedy" if greedy else "probabilistic"
        )
    )
    model.logits_processor = _ShardedLogits(model.base, start, end)
    return model


def _hand_back(handed: list[int], spec, num_reqs: int, head_hidden) -> bool:
    handed.append(num_reqs)
    return True


def vocab_shard_matches_vllm_loop(env: dict[str, str], world_size: int) -> None:
    set_env_vars_and_device(env)
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(world_size, 1)
    from tests.kernels.models.amd.test_muse_spark_fused_markov import (
        MAX_REQS,
        N_SPEC,
        _speculator,
    )
    from vllm.models.muse_spark.amd.ops.markov_vocab_shard import MarkovVocabShard
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    rank = get_tensor_model_parallel_rank()
    hidden = torch.zeros(MAX_REQS * N_SPEC, 8, device="cuda")
    cases = [
        (True, False, 1),
        (True, False, 8),
        (False, False, 1),
        (False, False, 16),
        (False, True, 3),
    ]
    for greedy, fp64, num_reqs in cases:
        s = _speculator(1.0, seed=40 + num_reqs, greedy=greedy, fp64=fp64, exact=True)
        s.max_num_reqs, s.device = MAX_REQS, torch.device("cuda")
        DSparkSpeculator._sample_sequential(s, num_reqs, hidden)
        ref_tok = s.draft_tokens.clone()
        ref_cache = None if greedy else s.draft_logits.values.clone()
        s.draft_tokens.zero_()
        if not greedy:
            s.draft_logits.values.zero_()

        shard = MarkovVocabShard(_sharded(s, rank, world_size, greedy), None)
        assert shard.sample_sequential(s, num_reqs, hidden)
        torch.accelerator.synchronize()
        case = f"greedy={greedy} fp64={fp64} num_reqs={num_reqs} rank={rank}"
        assert torch.equal(s.draft_tokens, ref_tok), case
        if ref_cache is not None:
            got = s.draft_logits.values.view(torch.int16)
            assert torch.equal(got, ref_cache.view(torch.int16)), case

        # A step it declines writes nothing and goes to the hook it was
        # installed ahead of (the --monokernel rounds), else to vLLM's loop.
        s.draft_tokens.zero_()
        s.draft_watermarker = object()
        assert not shard.sample_sequential(s, num_reqs, hidden)
        assert not s.draft_tokens.any()
        handed: list[int] = []
        shard.fallback = functools.partial(_hand_back, handed)
        assert shard.sample_sequential(s, num_reqs, hidden)
        assert handed == [num_reqs] and not s.draft_tokens.any()


@multi_gpu_test(num_gpus=2)
def test_muse_spark_markov_vocab_shard() -> None:
    distributed_run(vocab_shard_matches_vllm_loop, 2)
