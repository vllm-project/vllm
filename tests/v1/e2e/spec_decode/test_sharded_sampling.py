# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-sharded sampling must be a drop-in for replicated sampling under
speculative decoding.

The non-spec arm of this claim lives in tests/v1/e2e/general/
test_sharded_sampling.py. Spec decode is the harder case: draft tokens give
each request a different number of logits rows, so the shard plan's
per-request arithmetic and the rejection sampler are both exercised.
"""

import pytest
import torch

from tests.utils import wait_for_memory_to_settle
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from .utils import _skip_if_insufficient_gpus_for_tp, get_test_prompts, greedy_sampling


def _is_sharded_sampling_active(worker) -> bool:
    return worker.model_runner.batch_sharder is not None


def test_mtp_sharded_sampling_equivalence(monkeypatch: pytest.MonkeyPatch):
    """Batch-sharded sampling must be a bit-exact drop-in for replicated
    sampling under MTP spec decoding: the collectives move the same logits
    bytes, Gumbel keys derive from (request slot, position, seed), and slot
    assignment is rank-deterministic. Both runs here are spec decode with
    identical math, so outputs must match exactly."""
    tp_size = 2
    _skip_if_insufficient_gpus_for_tp(tp_size)
    model_name = "Qwen/Qwen3.5-0.8B-Base"
    test_prompts = get_test_prompts(mm_enabled=False, num_prompts=20)

    with monkeypatch.context() as m:
        # Batch-sharded sampling only exists in the V2 model runner.
        m.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        # Required for the collective_rpc mode check below.
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        def run(disable_sharding: bool):
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                enable_batch_sharded_sampling=not disable_sharding,
                speculative_config={
                    "method": "mtp",
                    "num_speculative_tokens": 1,
                    "max_model_len": 2048,
                },
                max_model_len=2048,
                limit_mm_per_prompt={"image": 0, "video": 0},
                # More requests than ranks, so round-robin slot ownership
                # splits the batch across both ranks.
                max_num_seqs=len(test_prompts),
                # Timing-based kernel selection is the dominant source of
                # cross-boot numeric noise; disable for a tight comparison.
                kernel_config={"enable_flashinfer_autotune": False},
            )
            try:
                # Guard against a vacuous comparison: verify every worker is
                # in the intended sampling mode.
                modes = llm.llm_engine.collective_rpc(_is_sharded_sampling_active)
                assert all(mode == (not disable_sharding) for mode in modes)
                greedy = llm.chat(test_prompts, greedy_sampling())
                seeded = llm.chat(
                    test_prompts,
                    SamplingParams(temperature=1.0, seed=33, max_tokens=10),
                )
                return greedy, seeded
            finally:
                del llm
                torch.accelerator.empty_cache()
                cleanup_dist_env_and_memory()
                wait_for_memory_to_settle()

        ref_greedy, ref_seeded = run(disable_sharding=True)
        shard_greedy, shard_seeded = run(disable_sharding=False)

        # Engine boots are not bitwise deterministic (kernel/collective
        # selection shifts logits by ~1 ulp, flipping near-tie tokens), so
        # allow a small number of divergent prompts. A sharding bug produces
        # wholesale divergence, not isolated near-tie flips.
        for name, ref_outputs, shard_outputs in (
            ("greedy", ref_greedy, shard_greedy),
            ("seeded", ref_seeded, shard_seeded),
        ):
            num_divergent = 0
            for i, (ref, out) in enumerate(zip(ref_outputs, shard_outputs)):
                if list(ref.outputs[0].token_ids) != list(out.outputs[0].token_ids):
                    num_divergent += 1
                    print(
                        f"{name} prompt {i} diverged:\n"
                        f"  replicated: {ref.outputs[0].text!r}\n"
                        f"  sharded:    {out.outputs[0].text!r}"
                    )
            assert num_divergent <= 2, (
                f"{name}: {num_divergent}/{len(ref_outputs)} prompts diverged, "
                "beyond near-tie boot noise"
            )
