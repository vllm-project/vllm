# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.triton_utils import HAS_TRITON


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_topk_topp_warmups_expand_for_cuda_alike_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.sample.ops import topk_topp_triton as topk_topp

    class RocmPlatform:
        @staticmethod
        def is_cuda() -> bool:
            return False

        @staticmethod
        def is_cuda_alike() -> bool:
            return True

    class Config:
        scheduler_config = SimpleNamespace(max_num_seqs=32)
        model_config = SimpleNamespace(get_vocab_size=lambda: 8192)
        num_speculative_tokens = 3

    monkeypatch.setattr(topk_topp, "current_platform", RocmPlatform())
    config = Config()
    monolithic_cases = list(
        topk_topp._topk_topp._provider_cases(
            topk_topp._topk_topp._warmup_inputs_fn, config
        )
    )
    assert monolithic_cases
    assert max(case["logits"].shape[0] for case in monolithic_cases) == (
        config.scheduler_config.max_num_seqs * config.num_speculative_tokens
    )
    assert not any(
        case["k"] is None
        and case["p"] is not None
        and case["logits"].shape[0] <= topk_topp._SPLIT_MAX_BATCH
        for case in monolithic_cases
    )

    for launcher in (
        topk_topp._topp_split_stats,
        topk_topp._topp_split_step,
        topk_topp._topp_split_mask,
    ):
        cases = list(launcher._provider_cases(launcher._warmup_inputs_fn, config))
        assert max(case["logits"].shape[0] for case in cases) == min(
            config.scheduler_config.max_num_seqs * config.num_speculative_tokens,
            topk_topp._SPLIT_MAX_BATCH,
        )

    logits = SimpleNamespace(shape=(1, 8192), device=SimpleNamespace(type="cuda"))
    _, launch_kwargs = topk_topp._topk_topp._dispatch_fn(
        logits,
        object(),
        object(),
        object(),
        object(),
        object(),
        float("-inf"),
        132,
    )
    assert launch_kwargs["SPLIT_COVERS_PONLY"]
