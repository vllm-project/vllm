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
    Fixture that patches CommonAttentionMetadata.seq_lens to detect .cpu() calls.
    This tracks when code accesses seq_lens and converts it to CPU, which causes
    a GPU-CPU sync that breaks async scheduling.
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    # Shared counter for cross-process communication (inherited by fork)
    sync_count = multiprocessing.Value("i", 0)

    original_cpu = torch.Tensor.cpu

    # Create a wrapper that tracks .cpu() calls on seq_lens tensors
    tracked_tensors: set = set()

    original_getattribute = CommonAttentionMetadata.__getattribute__

    def tracking_getattribute(self, name):
        value = original_getattribute(self, name)
        if name == "seq_lens" and isinstance(value, torch.Tensor):
            # Mark this tensor as one we want to track
            tracked_tensors.add(id(value))
        return value

    # Backends that intentionally call .cpu() for their operations
    ALLOWED_BACKENDS = ["flashinfer.py", "mla/indexer.py", "mla/flashmla_sparse.py"]

    def tracking_cpu(tensor_self, *args, **kwargs):
        if tensor_self.is_cuda and id(tensor_self) in tracked_tensors:
            # Check if this is from an allowed backend
            stack = traceback.format_stack()
            stack_str = "".join(stack)
            is_allowed = any(backend in stack_str for backend in ALLOWED_BACKENDS)
            if not is_allowed:
                with sync_count.get_lock():
                    sync_count.value += 1
                    count = sync_count.value
                print(f"\n{'=' * 60}", file=sys.stderr)
                print(
                    f"SYNC #{count}: .cpu() called on CommonAttentionMetadata.seq_lens",
                    file=sys.stderr,
                )
                print(
                    f"Shape: {tensor_self.shape}, dtype: {tensor_self.dtype}",
                    file=sys.stderr,
                )
                print(f"{'=' * 60}", file=sys.stderr)
                traceback.print_stack(file=sys.stderr)
                print(f"{'=' * 60}\n", file=sys.stderr)
                sys.stderr.flush()
        return original_cpu(tensor_self, *args, **kwargs)

    # Apply patches
    CommonAttentionMetadata.__getattribute__ = tracking_getattribute
    torch.Tensor.cpu = tracking_cpu

    class SyncTracker:
        @property
        def count(self) -> int:
            return sync_count.value

        def start_tracking(self):
            """Start tracking syncs from this point. Call after model loading."""
            with sync_count.get_lock():
                sync_count.value = 0
            tracked_tensors.clear()

        def assert_no_sync(self, msg: str = ""):
            count = sync_count.value
            assert count == 0, (
                f"Unexpected GPU-CPU sync: .cpu() called on "
                f"CommonAttentionMetadata.seq_lens {count} times. "
                f"See stack traces above. {msg}"
            )

    yield SyncTracker()

    # Restore original methods
    CommonAttentionMetadata.__getattribute__ = original_getattribute
    torch.Tensor.cpu = original_cpu
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

    # Start tracking after model loading - we only care about syncs during generation
    sync_tracker.start_tracking()

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
