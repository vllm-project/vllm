# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.metrics.loggers import LoggingStatLogger, PerEngineStatLoggerAdapter


def test_spec_decode_padding_tight_budget_e2e(vllm_runner, monkeypatch, caplog_vllm):
    """A real prefill leaving one slot must not defer a cached prompt tail."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if current_platform.is_cuda():
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    budget = 64
    prefix_len = 128
    max_tokens = 16
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=budget,
        block_size=16,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        disable_log_stats=False,
        cudagraph_metrics=True,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_min": 1,
            "prompt_lookup_max": 4,
        },
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [4, 8, 16, 32, 64],
        },
    ) as runner:
        llm = runner.llm
        engine = llm.llm_engine
        content = llm.get_tokenizer().encode(
            "This shared document describes a software system and its components. "
            "The system processes requests, caches common context, and produces "
            "a concise response. ",
            add_special_tokens=False,
        )
        cached = (content * ((prefix_len + len(content)) // len(content)))[
            : prefix_len + 1
        ]
        cold = [50000] + [1000 + (37 * i) % 10000 for i in range(budget - 2)]
        prompts = {
            "cold": {"prompt_token_ids": cold},
            "cached": {"prompt_token_ids": cached},
        }

        def sampling_params():
            return SamplingParams(
                temperature=0,
                max_tokens=max_tokens,
                ignore_eos=True,
                logprobs=5,
                output_kind=RequestOutputKind.CUMULATIVE,
            )

        warm = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)

        # Sequential reference with the same cache state and speculation config.
        llm.generate([prompts["cached"]], warm, use_tqdm=False)
        reference = {
            name: llm.generate([prompt], sampling_params(), use_tqdm=False)[0]
            for name, prompt in prompts.items()
        }
        assert reference["cached"].num_cached_tokens == prefix_len
        assert llm.reset_prefix_cache()
        llm.generate([prompts["cached"]], warm, use_tqdm=False)

        def accepted_tokens():
            return next(
                metric.value
                for metric in llm.get_metrics()
                if metric.name == "vllm:spec_decode_num_accepted_tokens"
            )

        accepted_before = accepted_tokens()
        # Queue both before executing the first synchronous engine iteration.
        # LLM.generate sets its params to FINAL_ONLY; use fresh cumulative params.
        for name, prompt in prompts.items():
            engine.add_request(name, prompt, sampling_params())
        first = {output.request_id: output for output in engine.step()}
        assert set(first) == set(prompts)
        assert first["cold"].num_cached_tokens == 0
        assert first["cached"].num_cached_tokens == prefix_len
        for output in first.values():
            completion = output.outputs[0]
            assert len(completion.token_ids) == 1
            assert completion.logprobs is not None
            assert len(completion.logprobs) == 1
            assert completion.token_ids[0] in completion.logprobs[0]

        # Observe actual model-runner dispatch, not an inferred graph mode.
        assert engine.logger_manager is not None
        graph = next(
            logger.last_scheduler_stats.cudagraph_stats
            for adapter in engine.logger_manager.stat_loggers
            if isinstance(adapter, PerEngineStatLoggerAdapter)
            for logger in adapter.per_engine_stat_loggers.values()
            if isinstance(logger, LoggingStatLogger)
        )
        assert graph is not None
        assert graph.num_unpadded_tokens == budget
        assert graph.runtime_mode != "FULL"
        if current_platform.is_cuda():
            assert graph.runtime_mode == "PIECEWISE"

        results = {}
        # Every subsequent iteration must make progress on a decode or finish.
        for _ in range(2 * max_tokens):
            for output in engine.step():
                if output.finished:
                    results[output.request_id] = output
            if not engine.has_unfinished_requests():
                break
        assert set(results) == set(prompts)
        assert accepted_tokens() > accepted_before

        def generated(outputs):
            completions = [outputs[name].outputs[0] for name in prompts]
            for completion in completions:
                assert len(completion.token_ids) == max_tokens
                assert completion.logprobs is not None
                assert len(completion.logprobs) == max_tokens
            return [
                (list(completion.token_ids), completion.text, completion.logprobs)
                for completion in completions
            ]

        reference_outputs = generated(reference)
        mixed_outputs = generated(results)
        # Compare complete greedy sequences, and check logprobs even when
        # the first sampled token matches the sequential reference.
        for always_check in (False, True):
            check_logprobs_close(
                outputs_0_lst=reference_outputs,
                outputs_1_lst=mixed_outputs,
                name_0="sequential",
                name_1="mixed",
                near_tie_tol=0.1,
                always_check_logprobs=always_check,
            )
