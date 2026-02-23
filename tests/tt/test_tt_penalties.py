# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from tests.tt.utils import (
    RequestConfig,
    assert_deterministic,
    assert_varied,
    run_concurrent_batch,
)


class TestRepetitionPenalty:
    """
    Different repetition penalties per request in same batch.
    """

    def test_different_repetition_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Each request has different repetition penalty.
        """
        prompt = "a a a a a a a a a"
        penalties = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0][:max_batch_size]

        # Requests otherwise reproducible
        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=10,
                temperature=0,
                repetition_penalty=penalty,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(
            results, 2, "Varying penalty should at least influence the output somewhat."
        )

    # Caught https://github.com/tenstorrent/vllm/issues/286
    def test_repetition_penalty_mixed_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        prompt = "a a a a a a a a a"

        configs = []
        for i in range(max_batch_size):
            if i % 2 == 0:
                # No penalty
                configs.append(
                    RequestConfig(
                        prompt=prompt,
                        max_tokens=10,
                        temperature=0,
                        repetition_penalty=1.0,
                    )
                )
            else:
                # High penalty
                configs.append(
                    RequestConfig(
                        prompt=prompt,
                        max_tokens=10,
                        temperature=0,
                        repetition_penalty=2.0,
                    )
                )

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        assert_deterministic(no_penalty, "No penalty requests should be identical.")
        assert_deterministic(with_penalty, "With penalty requests should be identical.")
        assert_varied(
            [no_penalty[0], with_penalty[0]], 2, "Penalty should change output."
        )


class TestPresencePenalty:
    """
    Different presence penalties per request.
    """

    def test_different_presence_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        prompt = "a b c a b c a b c"
        penalties = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0][:max_batch_size]

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=40,
                temperature=0,
                presence_penalty=penalty,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(
            results, 2, "Different presence penalties should produce different outputs."
        )

    def test_presence_penalty_mixed_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        # Presence penalty is only applied once regardless of the repetition,
        # so we use a weaker prompt for more even logits
        prompt = "a b c a b c a b c"
        configs = []
        for i in range(max_batch_size):
            configs.append(
                RequestConfig(
                    prompt=prompt,
                    max_tokens=40,
                    temperature=0,
                    presence_penalty=0.0 if i % 2 == 0 else 2.0,
                )
            )

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        assert_deterministic(no_penalty, "No penalty requests should be identical.")
        assert_deterministic(with_penalty, "With penalty requests should be identical.")
        assert_varied(
            [no_penalty[0], with_penalty[0]], 2, "Penalty should change output."
        )


class TestFrequencyPenalty:
    """
    Different frequency penalties per request.
    """

    def test_different_frequency_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        prompt = "5 5 5 5 5 5 5 5"

        penalties = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0][:max_batch_size]

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=20,
                temperature=0,
                frequency_penalty=penalty,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(
            results,
            2,
            "Different frequency penalties should produce different outputs.",
        )

    def test_frequency_penalty_mixed_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        prompt = "a a a a a a a a a"

        configs = []
        for i in range(max_batch_size):
            configs.append(
                RequestConfig(
                    prompt=prompt,
                    max_tokens=15,
                    temperature=0,
                    frequency_penalty=0.0 if i % 2 == 0 else 2.0,
                )
            )

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        # Count "a"s in each output
        no_penalty_a_count = no_penalty[0].count("a")
        with_penalty_a_count = with_penalty[0].count("a")

        assert no_penalty_a_count > with_penalty_a_count, (
            f"Frequency penalty should reduce 'a' repetitions: "
            f"no_penalty={no_penalty_a_count},"
            f"with_penalty={with_penalty_a_count}"
        )
        assert_deterministic(no_penalty, "No penalty requests should be identical.")
        assert_deterministic(with_penalty, "With penalty requests should be identical.")
