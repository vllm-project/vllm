# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that batched_count_greater_than does not trigger 0/1 specialization
recompiles when batch_size varies."""

import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.sampler import Sampler

DEVICE = current_platform.device_type


def test_batched_count_greater_than_correctness():
    """Basic correctness: counts elements >= the corresponding value."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=DEVICE)
    values = torch.tensor([[2.0], [5.0]], device=DEVICE)
    result = batched_count_greater_than(x, values)
    expected = torch.tensor([2, 2], device=DEVICE)
    torch.testing.assert_close(result, expected)


def test_gather_logprobs_no_recompile():
    """Sampler.gather_logprobs with batch_size=1 then 2 must not recompile.

    This guards against 0/1 specialization: dynamo normally specializes on
    tensor sizes 0 and 1, causing a recompile when the size first exceeds 1.
    The mark_unbacked calls in gather_logprobs prevent this.
    """
    torch._dynamo.reset()

    compile_count = 0
    orig_backend = current_platform.simple_compile_backend

    def counting_backend(gm, example_inputs):
        nonlocal compile_count
        compile_count += 1
        if orig_backend == "inductor":
            return torch._inductor.compile(gm, example_inputs)
        return gm

    # Monkey-patch batched_count_greater_than with our counting backend
    # so we can detect recompiles through the production code path.
    import vllm.v1.sample.ops.logprobs as logprobs_module
    import vllm.v1.sample.sampler as sampler_module

    unwrapped = batched_count_greater_than._torchdynamo_orig_callable
    patched = torch.compile(unwrapped, backend=counting_backend)
    orig_fn = logprobs_module.batched_count_greater_than

    logprobs_module.batched_count_greater_than = patched
    sampler_module.batched_count_greater_than = patched

    try:
        vocab_size = 32
        num_logprobs = 3

        # Call 1: batch_size=1
        logprobs1 = torch.randn(1, vocab_size, device=DEVICE)
        token_ids1 = torch.randint(
            0, vocab_size, (1,), device=DEVICE, dtype=torch.int64
        )
        Sampler.gather_logprobs(logprobs1, num_logprobs, token_ids1)
        assert compile_count == 1, f"Expected 1 compile, got {compile_count}"

        # Call 2: batch_size=2 — should NOT recompile
        logprobs2 = torch.randn(2, vocab_size, device=DEVICE)
        token_ids2 = torch.randint(
            0, vocab_size, (2,), device=DEVICE, dtype=torch.int64
        )
        Sampler.gather_logprobs(logprobs2, num_logprobs, token_ids2)
        assert compile_count == 1, (
            f"Recompiled on batch_size 1->2 (0/1 specialization). "
            f"Expected 1 compile, got {compile_count}"
        )

        # Call 3: batch_size=8 — should NOT recompile
        logprobs3 = torch.randn(8, vocab_size, device=DEVICE)
        token_ids3 = torch.randint(
            0, vocab_size, (8,), device=DEVICE, dtype=torch.int64
        )
        Sampler.gather_logprobs(logprobs3, num_logprobs, token_ids3)
        assert compile_count == 1, (
            f"Recompiled on batch_size change. Expected 1 compile, got {compile_count}"
        )
    finally:
        # Restore original function
        logprobs_module.batched_count_greater_than = orig_fn
        sampler_module.batched_count_greater_than = orig_fn
        torch._dynamo.reset()
