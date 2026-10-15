# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
import textwrap


def test_compute_token_logprobs_with_triton_interpreter() -> None:
    """Exercise both logprob kernels on CPU without requiring a GPU driver."""
    script = textwrap.dedent(
        """
        import torch

        from vllm import envs
        from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

        torch.manual_seed(7)
        dtypes = (torch.float32, torch.float16, torch.bfloat16)
        vocab_sizes = (2053, 5003)
        paths = ((3, False), (3, True), (6, False))

        for dtype in dtypes:
            for vocab_size in vocab_sizes:
                token_ids = (
                    torch.arange(1105, dtype=torch.int64).remainder(vocab_size)
                    .unsqueeze(0)
                )
                for batch_size, batch_invariant in paths:
                    envs.VLLM_BATCH_INVARIANT = batch_invariant
                    logits = torch.randn((batch_size, vocab_size), dtype=dtype)
                    logits[0].fill_(float("-inf"))
                    logits[1, : vocab_size // 2] = float("-inf")
                    row_ids = token_ids.expand(batch_size, -1).contiguous()

                    actual = compute_token_logprobs(logits, row_ids)
                    expected = torch.log_softmax(logits.float(), dim=-1).gather(
                        -1, row_ids
                    )
                    torch.testing.assert_close(
                        actual, expected, rtol=5e-4, atol=5e-4, equal_nan=True
                    )
        """
    )
    env = os.environ.copy()
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
