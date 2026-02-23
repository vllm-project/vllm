# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

from tests.tt.utils import RequestConfig, run_concurrent_batch


class TestBatchIsolation:
    """
    Verify each request gets its own parameters applied.
    """

    def test_mixed_params_batch(self, tt_server, tt_model_name, max_batch_size):
        """
        Batch where each request has completely different parameters.
        """
        configs = [
            # Greedy
            RequestConfig(prompt="Count: ", max_tokens=5, temperature=0),
            # High temp with seed
            RequestConfig(
                prompt="Random: ", max_tokens=5, temperature=2.0, top_k=100, seed=42
            ),
            # With repetition penalty
            RequestConfig(
                prompt="test test. Say: ",
                max_tokens=5,
                temperature=0.5,
                repetition_penalty=3.0,
                seed=42,
            ),
            # With presence penalty
            RequestConfig(
                prompt="List: ",
                max_tokens=10,
                temperature=0.5,
                presence_penalty=2.0,
                seed=42,
            ),
            # top_k=1 (should be greedy-like)
            RequestConfig(
                prompt="Word: ", max_tokens=3, temperature=2.0, top_k=1, seed=99
            ),
            # Low temp
            RequestConfig(prompt="Letter: ", max_tokens=3, temperature=0.01, seed=42),
            # High temp low top_k
            RequestConfig(
                prompt="Number: ", max_tokens=3, temperature=2.0, top_k=5, seed=42
            ),
            # With frequency penalty
            RequestConfig(
                prompt="a a a. Next: ",
                max_tokens=5,
                temperature=0.5,
                frequency_penalty=2.0,
                seed=42,
            ),
            # Combined penalties
            RequestConfig(
                prompt="All: ",
                max_tokens=8,
                temperature=0.5,
                repetition_penalty=1.5,
                presence_penalty=1.0,
                frequency_penalty=1.0,
                seed=6,
            ),
            # Top-p variation
            RequestConfig(
                prompt="TopP: ", max_tokens=8, temperature=1.0, top_p=0.5, seed=7
            ),
        ][:max_batch_size]

        # Run twice, with random batch order on second run
        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # Shuffle configs and results1 together for the second run
        indices = list(range(len(configs)))
        random.shuffle(indices)
        configs = [configs[i] for i in indices]
        results1 = [results1[i] for i in indices]

        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # Each deterministic config should reproduce
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            if configs[i].temperature == 0 or configs[i].seed is not None:
                assert r1 == r2, (
                    f"Request {i} should be deterministic/reproducible.\n"
                    f"Config:"
                    f"temp={configs[i].temperature},"
                    f"seed={configs[i].seed}\n"
                    f"Run 1: {r1!r}\n"
                    f"Run 2: {r2!r}"
                )
