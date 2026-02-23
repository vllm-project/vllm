# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.tt.utils import RequestConfig, run_concurrent_batch


class TestLogprobs:
    @pytest.mark.parametrize("batch_fraction", [0, 0.5, 1, 1.5])
    @pytest.mark.parametrize("num_logprobs", [1, 3, 5, 10])
    def test_logprobs(
        self, tt_server, tt_model_name, max_batch_size, batch_fraction, num_logprobs
    ):
        """Test logprobs parameter returns actual logprobs data.

        batch_fraction: 1.5 = one and a half batches, 1 = full batch,
            0.5 = half batch, 0 = single request
        num_logprobs: number of top logprobs alternatives to return
        """
        if batch_fraction == 0:
            num_requests = 1
        else:
            num_requests = max(1, int(max_batch_size * batch_fraction))

        # Use return_tokens_as_token_ids to ensure unique keys in top_logprobs
        # dict. Without this, different token IDs that decode to the same
        # string would collide, reducing the count below num_logprobs.
        configs = [
            RequestConfig(
                prompt=f"Count from {i}: ",
                max_tokens=10,
                logprobs=num_logprobs,
                return_tokens_as_token_ids=True,
            )
            for i in range(num_requests)
        ]
        results = run_concurrent_batch(
            tt_server, tt_model_name, configs, return_full_response=True
        )
        assert len(results) == len(configs)

        # Verify ALL responses have valid logprobs (catches DP rank issues)
        for i, response in enumerate(results):
            choice = response.choices[0]

            # Verify logprobs are returned for every request
            assert choice.logprobs is not None, (
                f"logprobs should be returned for request {i}"
            )
            assert choice.logprobs.tokens is not None, (
                f"logprobs.tokens should exist for request {i}"
            )
            assert len(choice.logprobs.tokens) > 0, (
                f"should have at least one token for request {i}"
            )

            # Verify top_logprobs contains the requested number of alternatives
            assert choice.logprobs.top_logprobs is not None, (
                f"top_logprobs should be returned for request {i}"
            )
            assert len(choice.logprobs.top_logprobs) > 0, (
                f"should have top_logprobs for tokens for request {i}"
            )
            # Each position should have up to num_logprobs+1 alternatives
            for j, top_lp in enumerate(choice.logprobs.top_logprobs):
                assert len(top_lp) >= num_logprobs, (
                    f"request {i}, token {j}: should have at least "
                    f"{num_logprobs} alternatives per token"
                )
                assert len(top_lp) <= num_logprobs + 1, (
                    f"request {i}, token {j}: should have at most "
                    f"{num_logprobs + 1} alternatives per token"
                )

            # Verify actual logprob values exist (not just structure)
            assert choice.logprobs.token_logprobs is not None, (
                f"request {i}: token_logprobs is None"
            )
            for j, lp in enumerate(choice.logprobs.token_logprobs):
                assert lp is not None, f"request {i}, token {j}: logprob value is None"
