# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest

from tests.tt.utils import (
    RequestConfig,
    assert_deterministic,
    assert_pairwise_varied,
    assert_varied,
    run_concurrent_batch,
)


class TestSeedingAndVariety:
    def test_seeding(self, tt_server, tt_model_name, max_batch_size):
        """
        Batch with mix of greedy (temp=0) and sampling requests.
        Greedy should be deterministic, sampling should vary.
        """
        prompt = "Random word: "

        # greedy, seeds, seeds
        seeds = max_batch_size // 3
        greedy_count = max_batch_size - seeds - seeds
        configs = []
        # Greedy requests
        for i in range(greedy_count):
            configs.append(RequestConfig(prompt=prompt, max_tokens=10, temperature=0))

        # Sampling requests with different seeds
        for _ in range(2):
            for i in range(seeds):
                configs.append(
                    RequestConfig(
                        prompt=prompt,
                        max_tokens=10,
                        temperature=1.5,
                        top_k=50,
                        seed=i * 100,
                    )
                )

        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # This tests both prefill and decode
        all_greedy = results1[:greedy_count] + results2[:greedy_count]
        assert_deterministic(
            all_greedy,
            "Greedy requests should produce the same outputacross positions and runs.",
        )

        # Check answers overall
        different_seeds = []
        for i in range(seeds):
            results_for_seed = [
                results1[greedy_count + i],
                results2[greedy_count + i],
                results1[greedy_count + seeds + i],
                results2[greedy_count + seeds + i],
            ]
            assert_deterministic(
                results_for_seed,
                "Seeded requests should produce the same output"
                "across positions and runs.",
            )
            different_seeds.append(results_for_seed[0])
        expected_variety = seeds // 3
        assert_varied(
            different_seeds,
            expected_variety,
            "Seeded requests should produce different outputsfor different seeds.",
        )

        # Check first tokens to make sure prefill is varied
        prefill_different_seeds = [x[:1] for x in different_seeds]
        assert_varied(
            prefill_different_seeds,
            expected_variety,
            "First token should be varied for different seeds.",
        )

    def test_different_seeds_produce_different_outputs(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Same prompt, different seeds per request.
        """
        prompt = "Random story: "

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=20,
                temperature=1.0,
                top_k=50,
                seed=seed,
            )
            for seed in range(max_batch_size)
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(
            results,
            max_batch_size // 3,
            "Different seeds should produce different outputs.",
        )

    def test_same_seeds_reproduce_across_batches(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Same seeds should reproduce even when in different batch positions.
        """
        prompt = "Generate: "
        seeds = list(range(100, 100 + max_batch_size))

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=15,
                temperature=1.0,
                top_k=50,
                seed=seed,
            )
            for seed in seeds
        ]

        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)

        shuffled_indices = list(range(max_batch_size))
        random.Random(42).shuffle(shuffled_indices)
        shuffled_configs = [configs[i] for i in shuffled_indices]

        results2 = run_concurrent_batch(tt_server, tt_model_name, shuffled_configs)

        results_per_seed: list[list[str]] = [[] for _ in range(max_batch_size)]
        for i in range(max_batch_size):
            results_per_seed[i].append(results1[i])
        for i, result in enumerate(results2):
            results_per_seed[shuffled_indices[i]].append(result)

        for i in range(max_batch_size):
            assert_deterministic(
                results_per_seed[i],
                "Seed should produce the same output in different batches.",
            )

    @pytest.mark.parametrize("seed", [42, 123, 999, 0])
    def test_specific_seed_reproducible(
        self, tt_server, tt_model_name, max_batch_size, seed
    ):
        """
        A specific seed should produce same result across batch runs.
        """
        # Create batch where one request has our test seed
        configs = [
            RequestConfig(
                prompt=f"Text {i}:",
                max_tokens=10,
                temperature=1.0,
                top_k=50,
                seed=seed if i == 0 else i * 1000,
            )
            for i in range(max_batch_size)
        ]

        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # The test seed request should reproduce
        assert results1[0] == results2[0], (
            f"Seed {seed} should reproduce.\n"
            f"Run 1: {results1[0]!r}\n"
            f"Run 2: {results2[0]!r}"
        )

    @pytest.mark.parametrize("seed", [1, 0])
    def test_batch1_seed_reproducible(self, tt_server, tt_model_name, seed):
        """
        Batch with 1 user with seed should produce reproducible outputs.
        """
        prompt = "Random story: "
        configs = [
            RequestConfig(
                prompt=prompt, max_tokens=20, temperature=10.0, top_k=50, seed=seed
            )
        ]
        results = []
        for _ in range(10):
            results.extend(run_concurrent_batch(tt_server, tt_model_name, configs))
        assert_deterministic(
            results, "Batch with 1 user with seed should produce reproducible outputs."
        )

    def test_batch1_no_seed_varied(self, tt_server, tt_model_name):
        """
        Batch with 1 user without seed should produce varied outputs.
        """
        prompt = "Random story: "
        configs = [
            RequestConfig(prompt=prompt, max_tokens=20, temperature=10.0, top_k=50)
        ]
        results = []
        for _ in range(10):
            results.extend(run_concurrent_batch(tt_server, tt_model_name, configs))
        assert_varied(
            results, 2, "Batch with 1 user with no seed should produce varied outputs."
        )

    @pytest.mark.parametrize("seed", [1, 0])
    @pytest.mark.parametrize("batch_size", [32, 10, 1])
    def test_uniform_seed_deterministic(
        self, tt_server, tt_model_name, batch_size, seed
    ):
        """
        Batches with uniform seed should produce deterministic outputs.
        """
        prompt = "Random story: "
        configs = [
            RequestConfig(
                prompt=prompt, max_tokens=20, temperature=1.0, top_k=50, seed=seed
            )
            for _ in range(batch_size)
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_deterministic(results, "Seed should produce deterministic outputs.")

    def test_uniform_noseed_varied(self, tt_server, tt_model_name, max_batch_size):
        """
        Full batch without seed should produce varied outputs.
        """
        prompt = "Random story: "
        configs = [
            RequestConfig(prompt=prompt, max_tokens=20, temperature=10.0, top_k=50)
            for _ in range(max_batch_size)
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(results, 5, "No seed should produce varied outputs.")

    def test_negative_seed_does_not_crash(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Negative seed should not crash.
        """
        prompt = "Random story: "
        configs = [
            RequestConfig(
                prompt=prompt, max_tokens=20, temperature=1.0, top_k=50, seed=-1
            )
        ]
        _ = run_concurrent_batch(tt_server, tt_model_name, configs)

    @pytest.mark.parametrize("batch_size", [7, 10, 19, 32, 37])
    def test_temperature_varied_in_batch(self, tt_server, tt_model_name, batch_size):
        prompt = "Random letter: "

        configs = [
            RequestConfig(prompt=prompt, max_tokens=10, temperature=5)
            for _ in range(batch_size)
        ]

        # Overall
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(results, 2, "With temperature=5, outputs should vary in batch.")

        # First token (prefill)
        prefill_results = [x[:1] for x in results]
        assert_varied(
            prefill_results, 2, "With temperature=5, first tokens should vary in batch."
        )

    def test_temperature_varied_between_batches(self, tt_server, tt_model_name):
        prompt = "Random letter: "

        configs = [RequestConfig(prompt=prompt, max_tokens=10, temperature=5)]

        tries = 10

        # Overall
        results = [
            run_concurrent_batch(tt_server, tt_model_name, configs)[0]
            for _ in range(tries)
        ]
        assert_varied(results, 2, "With temperature=5, outputs should vary on re-runs.")

        # First token (prefill)
        prefill_results = [x[:1] for x in results]
        assert_varied(
            prefill_results,
            2,
            "With temperature=5, first tokens should vary on re-runs.",
        )

    @pytest.mark.parametrize("batch_size", [15, 19, 32])
    def test_topk(self, tt_server, tt_model_name, batch_size):
        prompt = "<question>Random letter a-z:</question><answer>"
        num_greedy = batch_size // 2
        greedy_config = [
            RequestConfig(prompt=prompt, max_tokens=10, temperature=0)
            for _ in range(num_greedy)
        ]

        topk_configs = [
            RequestConfig(prompt=prompt, max_tokens=10, top_k=10, temperature=50)
            for _ in range(batch_size - num_greedy)
        ]
        joint_configs = greedy_config + topk_configs

        results_1 = run_concurrent_batch(tt_server, tt_model_name, joint_configs)
        results_2 = run_concurrent_batch(tt_server, tt_model_name, joint_configs)

        # Within each batch:
        for results in [results_1, results_2]:
            # Overall
            greedy_results = results[:num_greedy]
            non_greedy_results = results[num_greedy:]

            assert_deterministic(
                greedy_results, "Greedy requests should produce the same output."
            )
            assert_varied(
                non_greedy_results, 2, "With top_k=10, outputs should vary in batch."
            )

            # First token (prefill)
            prefill_results = [x[:1] for x in non_greedy_results]
            assert_varied(
                prefill_results, 2, "With top_k=10, first tokens should vary in batch."
            )

        # Between batches:
        greedy_results_1 = results_1[:num_greedy]
        greedy_results_2 = results_2[:num_greedy]
        assert_deterministic(
            greedy_results_1 + greedy_results_2,
            "Greedy requests should produce the same output when re-ran.",
        )

        # Overall
        non_greedy_results_1 = results_1[num_greedy:]
        non_greedy_results_2 = results_2[num_greedy:]
        assert_pairwise_varied(
            non_greedy_results_1,
            non_greedy_results_2,
            len(non_greedy_results_1) // 2,
            "Non-greedy requests should produce different outputs when re-ran.",
        )

        # First token (prefill)
        prefill_results_1 = [x[:1] for x in non_greedy_results_1]
        prefill_results_2 = [x[:1] for x in non_greedy_results_2]
        assert_pairwise_varied(
            prefill_results_1,
            prefill_results_2,
            2,
            "With top_k=10, first tokens should vary when re-ran.",
        )

    def test_top1_is_greedy(self, tt_server, tt_model_name):
        prompt = "Number: "
        greedy_config = RequestConfig(prompt=prompt, max_tokens=10, temperature=0)
        top1_config = RequestConfig(
            prompt=prompt, max_tokens=10, temperature=2.0, top_k=1
        )
        batch = [greedy_config] + [top1_config] * 3
        result_1 = run_concurrent_batch(tt_server, tt_model_name, batch)
        result_2 = run_concurrent_batch(tt_server, tt_model_name, batch)

        all_results = result_1 + result_2
        assert_deterministic(
            all_results,
            "top_k=1 requests should produce the same outputacross positions and runs.",
        )
