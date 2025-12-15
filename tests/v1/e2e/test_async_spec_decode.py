# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that verifies no implicit GPU-CPU synchronization occurs during
speculative decoding generation under expected conditions.
"""

import multiprocessing
import sys
import traceback

import pytest
import torch


@pytest.fixture
def sync_tracker():
    """
    Fixture that patches CommonAttentionMetadata.seq_lens_cpu to detect
    lazy init syncs. Prints stack traces immediately when syncs occur.
    """
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    # Shared counter for cross-process communication (inherited by fork)
    sync_count = multiprocessing.Value("i", 0)

    # Save original property
    original_prop = CommonAttentionMetadata.seq_lens_cpu
    original_fget = original_prop.fget

    # Create tracking wrapper
    def tracking_seq_lens_cpu(self):
        if self._seq_lens_cpu is None:
            # Increment counter
            with sync_count.get_lock():
                sync_count.value += 1
                count = sync_count.value
            # Print stack trace immediately (shows in subprocess output)
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"SYNC #{count}: seq_lens_cpu lazy init triggered!", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
            sys.stderr.flush()
        return original_fget(self)

    # Apply patch
    CommonAttentionMetadata.seq_lens_cpu = property(tracking_seq_lens_cpu)

    class SyncTracker:
        @property
        def count(self) -> int:
            return sync_count.value

        def assert_no_sync(self, msg: str = ""):
            count = sync_count.value
            assert count == 0, (
                f"Unexpected GPU-CPU sync: seq_lens_cpu lazy init triggered "
                f"{count} times. See stack traces above. {msg}"
            )

    yield SyncTracker()

    # Restore original property
    CommonAttentionMetadata.seq_lens_cpu = original_prop
    torch._dynamo.reset()


# Test configurations: (model, spec_model, method, num_spec_tokens, backend_env)
SPEC_DECODE_CONFIGS = [
    pytest.param(
        "meta-llama/Llama-3.2-1B-Instruct",
        "nm-testing/Llama3_2_1B_speculator.eagle3",
        "eagle3",
        2,
        id="eagle3-llama",
    ),
    pytest.param(
        "eagle618/deepseek-v3-random",
        "eagle618/eagle-deepseek-v3-random",
        "eagle",
        2,
        id="eagle-mla-deepseek",
    ),
]


@pytest.mark.parametrize(
    "model,spec_model,method,num_spec_tokens",
    SPEC_DECODE_CONFIGS,
)
def test_no_sync_with_spec_decode(
    sync_tracker,
    model: str,
    spec_model: str,
    method: str,
    num_spec_tokens: int,
):
    """
    Test that no implicit GPU-CPU sync occurs during speculative decoding
    generation.
    """
    # Import vLLM AFTER sync_tracker fixture has applied the patch
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = LLM(
        model=model,
        max_model_len=256,
        speculative_config={
            "method": method,
            "num_speculative_tokens": num_spec_tokens,
            "model": spec_model,
        },
        enforce_eager=True,
        async_scheduling=True,
    )

    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0, max_tokens=10),
    )

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0

    del llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    sync_tracker.assert_no_sync()
