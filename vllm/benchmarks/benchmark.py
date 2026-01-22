# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import importlib.util
import random
import shutil
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)
from vllm.benchmarks.lib.ready_checker import wait_for_endpoint
from vllm.tokenizers import TokenizerLike

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

TERM_PLOTLIB_AVAILABLE = (importlib.util.find_spec("termplotlib") is not None) and (
    shutil.which("gnuplot") is not None
)


class TaskType(Enum):
    GENERATION = "generation"
    POOLING = "pooling"


@dataclass
class BenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    # Max output tokens per second and concurrent requests at that peak
    max_output_tokens_per_s: float
    max_concurrent_requests: int


@dataclass
class EmbedBenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    request_throughput: float
    total_token_throughput: float
    mean_e2el_ms: float
    std_e2el_ms: float
    median_e2el_ms: float
    percentiles_e2el_ms: float


@dataclass
class SpecDecodeMetrics:
    """Speculative decoding metrics from the server's Prometheus endpoint."""

    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    accepted_per_pos: dict[int, int]


@dataclass
class ServingBenchmarkConfig:
    task_type: TaskType
    endpoint_type: str
    api_url: str
    base_url: str
    model_id: str
    model_name: str
    tokenizer: TokenizerLike
    logprobs: int | None
    request_rate: float
    burstiness: float
    disable_tqdm: bool
    num_warmups: int
    profile: bool
    selected_percentile_metrics: list[str]
    selected_percentiles: list[float]
    ignore_eos: bool
    goodput_config_dict: dict[str, float]
    max_concurrency: int | None
    lora_modules: Iterable[str] | None
    extra_headers: dict | None
    extra_body: dict | None
    ramp_up_strategy: Literal["linear", "exponential"] | None = None
    ramp_up_start_rps: int | None = None
    ramp_up_end_rps: int | None = None
    ready_check_timeout_sec: int = 600


class ServingBenchmark:
    def __init__(
        self, config: ServingBenchmarkConfig, input_requests: list[SampleRequest]
    ):
        self.config = config
        self.input_requests = input_requests
        self.spec_decode_stats = None

    async def run(self):
        request_func = self._get_request_func()
        session = self._get_session()
        test_input = self._get_test_input()
        # test run
        await self._ready_check(request_func, session, test_input)
        # warmup
        await self._run_warmup(request_func, session, test_input)
        print("Starting main benchmark run...")
        with self.benchmark_context(request_func, session, test_input):
            outputs, benchmark_duration, rps_change_events = await self.run_benchmark(
                request_func, session
            )

        if self.config.task_type == TaskType.GENERATION:
            metrics, actual_output_lens = self.calculate_metrics(
                input_requests=self.input_requests,
                outputs=outputs,
                dur_s=benchmark_duration,
                tokenizer=self.config.tokenizer,
                selected_percentiles=self.config.selected_percentiles,
                goodput_config_dict=self.config.goodput_config_dict,
            )
        else:
            metrics = self.calculate_metrics_for_embeddings(
                outputs=outputs,
                dur_s=benchmark_duration,
                selected_percentiles=self.config.selected_percentiles,
            )
            actual_output_lens = 0

        result = self.init_result(
            metrics, benchmark_duration, outputs, actual_output_lens, rps_change_events
        )

        result |= self.process_metrics(metrics, benchmark_duration)

        return result

    async def _get_spec_decode_stats(self, spec_decode_metrics_before, session):
        spec_decode_metrics_after = await self.fetch_spec_decode_metrics(session)
        spec_decode_stats: dict[str, Any] | None = None
        if (
            spec_decode_metrics_before is not None
            and spec_decode_metrics_after is not None
        ):
            delta_drafts = (
                spec_decode_metrics_after.num_drafts
                - spec_decode_metrics_before.num_drafts
            )
            delta_draft_tokens = (
                spec_decode_metrics_after.num_draft_tokens
                - spec_decode_metrics_before.num_draft_tokens
            )
            delta_accepted = (
                spec_decode_metrics_after.num_accepted_tokens
                - spec_decode_metrics_before.num_accepted_tokens
            )
            per_pos_rates: list[float] = []
            if delta_drafts > 0:
                positions = sorted(
                    set(spec_decode_metrics_before.accepted_per_pos.keys())
                    | set(spec_decode_metrics_after.accepted_per_pos.keys())
                )
                for pos in positions:
                    before_val = spec_decode_metrics_before.accepted_per_pos.get(pos, 0)
                    after_val = spec_decode_metrics_after.accepted_per_pos.get(
                        pos, before_val
                    )
                    delta_pos = after_val - before_val
                    per_pos_rates.append(delta_pos / delta_drafts)

            if delta_draft_tokens > 0:
                acceptance_rate = (delta_accepted / delta_draft_tokens) * 100
                acceptance_length = (
                    1 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0
                )
                spec_decode_stats = {
                    "num_drafts": delta_drafts,
                    "draft_tokens": delta_draft_tokens,
                    "accepted_tokens": delta_accepted,
                    "acceptance_rate": acceptance_rate,
                    "acceptance_length": acceptance_length,
                    "per_position_acceptance_rates": per_pos_rates,
                }
        return spec_decode_stats

    async def run_benchmark(self, request_func, session):
        if self.config.lora_modules:
            # For each input request, choose a LoRA module at random.
            self.config.lora_modules = iter(
                [
                    random.choice(self.config.lora_modules)
                    for _ in range(len(self.input_requests))
                ]
            )
        distribution = (
            "Poisson process" if self.config.burstiness == 1.0 else "Gamma distribution"
        )

        if self.config.ramp_up_strategy is not None:
            print(f"Traffic ramp-up strategy: {self.config.ramp_up_strategy}.")
            print(
                f"Will increase RPS from {self.config.ramp_up_start_rps} to "
                f"{self.config.ramp_up_end_rps} RPS over the duration of the benchmark."
            )
        else:
            print(f"Traffic request rate: {self.config.request_rate}")

        print(f"Burstiness factor: {self.config.burstiness} ({distribution})")
        print(f"Maximum request concurrency: {self.config.max_concurrency}")

        pbar = (
            None if self.config.disable_tqdm else tqdm(total=len(self.input_requests))
        )

        semaphore = (
            asyncio.Semaphore(self.config.max_concurrency)
            if self.config.max_concurrency
            else contextlib.nullcontext()
        )

        async def limited_request_func(request_func_input, session, pbar):
            async with semaphore:
                return await request_func(
                    request_func_input=request_func_input, session=session, pbar=pbar
                )

        benchmark_start_time = time.perf_counter()
        tasks: list[asyncio.Task] = []

        rps_change_events = []
        last_int_rps = -1
        if (
            self.config.ramp_up_strategy is not None
            and self.config.ramp_up_start_rps is not None
        ):
            last_int_rps = self.config.ramp_up_start_rps
            rps_change_events.append(
                {
                    "rps": last_int_rps,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        async for request, current_request_rate in self.get_request(
            self.input_requests,
            self.config.request_rate,
            self.config.burstiness,
            self.config.ramp_up_strategy,
            self.config.ramp_up_start_rps,
            self.config.ramp_up_end_rps,
        ):
            if self.config.ramp_up_strategy is not None:
                current_int_rps = int(current_request_rate)
                if current_int_rps > last_int_rps:
                    timestamp = datetime.now().isoformat()
                    for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                        rps_change_events.append(
                            {"rps": rps_val, "timestamp": timestamp}
                        )
                    last_int_rps = current_int_rps
            prompt, prompt_len, output_len, mm_content, request_id = (
                request.prompt,
                request.prompt_len,
                request.expected_output_len,
                request.multi_modal_data,
                request.request_id,
            )
            req_model_id, req_model_name = self.config.model_id, self.config.model_name
            if self.config.lora_modules:
                req_lora_module = next(self.config.lora_modules)
                req_model_id, req_model_name = req_lora_module, req_lora_module

            request_func_input = RequestFuncInput(
                model=req_model_id,
                model_name=req_model_name,
                prompt=prompt,
                api_url=self.config.api_url,
                prompt_len=prompt_len,
                output_len=output_len,
                logprobs=self.config.logprobs,
                multi_modal_content=mm_content,
                ignore_eos=self.config.ignore_eos,
                extra_headers=self.config.extra_headers,
                extra_body=self.config.extra_body,
                request_id=request_id,
            )
            tasks.append(
                asyncio.create_task(
                    limited_request_func(
                        request_func_input=request_func_input,
                        session=session,
                        pbar=pbar,
                    )
                )
            )
            outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

            if pbar is not None:
                pbar.close()

            benchmark_duration = time.perf_counter() - benchmark_start_time

        return outputs, benchmark_duration, rps_change_events

    def _get_request_func(self):
        try:
            request_func = self.__class__.get_async_request_funcs_map()[
                self.config.endpoint_type
            ]
            return request_func
        except KeyError:
            raise ValueError(f"Unknown backend: {self.config.endpoint_type}") from None

    def _get_session(self):
        # Reuses connections across requests to reduce TLS handshake overhead.
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrency or 0,
            limit_per_host=self.config.max_concurrency or 0,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ssl=("https://" in self.config.api_url),
        )

        return aiohttp.ClientSession(
            connector=connector,
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
        )

    def _get_test_input(self):
        test_prompt, test_prompt_len, test_output_len, test_mm_content = (
            self.input_requests[0].prompt,
            self.input_requests[0].prompt_len,
            self.input_requests[0].expected_output_len,
            self.input_requests[0].multi_modal_data,
        )
        assert (
            test_mm_content is None
            or isinstance(test_mm_content, dict)
            or (
                isinstance(test_mm_content, list)
                and all(isinstance(item, dict) for item in test_mm_content)
            )
        ), "multi_modal_data must be a dict or list[dict]"
        return RequestFuncInput(
            model=self.config.model_id,
            model_name=self.config.model_name,
            prompt=test_prompt,
            api_url=self.config.api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=self.config.logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=self.config.ignore_eos,
            extra_headers=self.config.extra_headers,
            extra_body=self.config.extra_body,
        )

    async def _ready_check(self, request_func, session, test_input):
        print("Starting initial single prompt test run...")
        if self.config.ready_check_timeout_sec > 0:
            test_output = await wait_for_endpoint(
                request_func,
                test_input,
                session,
                timeout_seconds=self.config.ready_check_timeout_sec,
            )
            if not test_output.success:
                raise ValueError(
                    "Initial test run failed - Please make sure benchmark "
                    "arguments are correctly specified. "
                    f"Error: {test_output.error}"
                )
            else:
                print("Initial test run completed.")
        else:
            print("Skipping endpoint ready check.")

    async def _run_warmup(self, request_func, session, test_input):
        if self.config.num_warmups > 0:
            print(f"Warming up with {self.config.num_warmups} requests...")
            warmup_pbar = (
                None
                if self.config.disable_tqdm
                else tqdm(total=self.config.num_warmups)
            )
            warmup_semaphore = (
                asyncio.Semaphore(self.config.max_concurrency)
                if self.config.max_concurrency
                else contextlib.nullcontext()
            )
            warmup_tasks = []

            async def warmup_limited_request_func():
                async with warmup_semaphore:
                    return await request_func(
                        request_func_input=test_input, session=session, pbar=warmup_pbar
                    )

            for _ in range(self.config.num_warmups):
                request_task = asyncio.create_task(warmup_limited_request_func())
                warmup_tasks.append(request_task)
            _ = await asyncio.gather(*warmup_tasks)

            if warmup_pbar is not None:
                warmup_pbar.close()
            print("Warmup run completed.")

    async def get_request(
        self,
        input_requests: list[SampleRequest],
        request_rate: float,
        burstiness: float = 1.0,
        ramp_up_strategy: Literal["linear", "exponential"] | None = None,
        ramp_up_start_rps: int | None = None,
        ramp_up_end_rps: int | None = None,
    ) -> AsyncGenerator[tuple[SampleRequest, float], None]:
        """
        Asynchronously generates requests at a specified rate
        with OPTIONAL burstiness and OPTIONAL ramp-up strategy.

        Args:
            input_requests:
                A list of input requests, each represented as a SampleRequest.
            request_rate:
                The rate at which requests are generated (requests/s).
            burstiness (optional):
                The burstiness factor of the request generation.
                Only takes effect when request_rate is not inf.
                Default value is 1, which follows a Poisson process.
                Otherwise, the request intervals follow a gamma distribution.
                A lower burstiness value (0 < burstiness < 1) results
                in more bursty requests, while a higher burstiness value
                (burstiness > 1) results in a more uniform arrival of requests.
            ramp_up_strategy (optional):
                The ramp-up strategy. Can be "linear" or "exponential".
                If None, uses constant request rate (specified by request_rate).
            ramp_up_start_rps (optional):
                The starting request rate for ramp-up.
            ramp_up_end_rps (optional):
                The ending request rate for ramp-up.
        """
        assert burstiness > 0, (
            f"A positive burstiness factor is expected, but given {burstiness}."
        )
        # Convert to list to get length for ramp-up calculations
        if isinstance(input_requests, Iterable) and not isinstance(
            input_requests, list
        ):
            input_requests = list(input_requests)

        total_requests = len(input_requests)
        assert total_requests > 0, "No requests provided."

        # Precompute delays among requests to minimize request send laggings
        request_rates = []
        delay_ts = []
        for request_index, request in enumerate(input_requests):
            current_request_rate = self._get_current_request_rate(
                ramp_up_strategy,
                ramp_up_start_rps,
                ramp_up_end_rps,
                request_index,
                total_requests,
                request_rate,
            )
            assert current_request_rate > 0.0, (
                f"Obtained non-positive request rate {current_request_rate}."
            )
            request_rates.append(current_request_rate)
            if current_request_rate == float("inf"):
                delay_ts.append(0)
            elif burstiness == float("inf"):
                # when burstiness tends to infinity, the delay time becomes constant
                # and tends to the inverse of the request rate
                delay_ts.append(1.0 / current_request_rate)
            else:
                theta = 1.0 / (current_request_rate * burstiness)

                # Sample the request interval from the gamma distribution.
                # If burstiness is 1, it follows exponential distribution.
                delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

        # Calculate the cumulative delay time from the first sent out requests.
        for i in range(1, len(delay_ts)):
            delay_ts[i] += delay_ts[i - 1]
        if ramp_up_strategy is None and delay_ts[-1] != 0:
            # When ramp_up_strategy is not set, we assume the request rate is fixed
            # and all requests should be sent in target_total_delay_s, the following
            # logic would re-scale delay time to ensure the final delay_ts
            # align with target_total_delay_s.
            #
            # NOTE: If we simply accumulate the random delta values
            # from the gamma distribution, their sum would have 1-2% gap
            # from target_total_delay_s. The purpose of the following logic is to
            # close the gap for stabilizing the throughput data
            # from different random seeds.
            target_total_delay_s = total_requests / request_rate
            normalize_factor = target_total_delay_s / delay_ts[-1]
            delay_ts = [delay * normalize_factor for delay in delay_ts]

        start_ts = time.time()
        for request_index, request in enumerate(input_requests):
            if delay_ts[request_index] > 0:
                current_ts = time.time()
                sleep_interval_s = start_ts + delay_ts[request_index] - current_ts
                if sleep_interval_s > 0:
                    await asyncio.sleep(sleep_interval_s)
            yield request, request_rates[request_index]

    def _get_current_request_rate(
        ramp_up_strategy: Literal["linear", "exponential"] | None,
        ramp_up_start_rps: int | None,
        ramp_up_end_rps: int | None,
        request_index: int,
        total_requests: int,
        request_rate: float,
    ) -> float:
        if (
            ramp_up_strategy
            and ramp_up_start_rps is not None
            and ramp_up_end_rps is not None
        ):
            progress = request_index / max(total_requests - 1, 1)
            if ramp_up_strategy == "linear":
                increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
                return ramp_up_start_rps + increase
            elif ramp_up_strategy == "exponential":
                ratio = ramp_up_end_rps / ramp_up_start_rps
                return ramp_up_start_rps * (ratio**progress)
            else:
                raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
        return request_rate

    @asynccontextmanager
    async def benchmark_context(self, request_func, session, test_input):
        spec_decode_metrics_before = await self.fetch_spec_decode_metrics(session)
        if self.config.profile:
            print("Starting profiler...")
            profile_input = RequestFuncInput(
                model=self.config.model_id,
                model_name=self.config.model_name,
                prompt=test_input.prompt,
                api_url=self.config.base_url + "/start_profile",
                prompt_len=test_input.prompt_len,
                output_len=test_input.output_len,
                logprobs=test_input.logprobs,
                multi_modal_content=test_input.multi_modal_content,
                ignore_eos=self.config.ignore_eos,
                extra_headers=self.config.extra_headers,
                extra_body=self.config.extra_body,
            )
            profile_output = await request_func(
                request_func_input=profile_input, session=session
            )
            if profile_output.success:
                print("Profiler started")
        try:
            yield
        finally:
            self.spec_decode_stats = await self._get_spec_decode_stats(
                spec_decode_metrics_before, session
            )
            if self.config.profile:
                print("Stopping profiler...")
                profile_input = RequestFuncInput(
                    model=self.config.model_id,
                    prompt=test_input.prompt,
                    api_url=self.config.base_url + "/stop_profile",
                    prompt_len=test_input.prompt_len,
                    output_len=test_input.output_len,
                    logprobs=test_input.logprobs,
                )
                profile_output = await request_func(
                    request_func_input=profile_input, session=session
                )
                if profile_output.success:
                    print("Profiler stopped")

    def calculate_metrics(
        self,
        input_requests: list[SampleRequest],
        outputs: list[RequestFuncOutput],
        dur_s: float,
        tokenizer: TokenizerLike,
        selected_percentiles: list[float],
        goodput_config_dict: dict[str, float],
    ) -> tuple[BenchmarkMetrics, list[int]]:
        """Calculate the metrics for the benchmark.

        Args:
            input_requests: The input requests.
            outputs: The outputs of the requests.
            dur_s: The duration of the benchmark.
            tokenizer: The tokenizer to use.
            selected_percentiles: The percentiles to select.
            goodput_config_dict: The goodput configuration.

        Returns:
            A tuple of the benchmark metrics and the actual output lengths.
        """
        actual_output_lens: list[int] = []
        total_input = 0
        completed = 0
        good_completed = 0
        itls: list[float] = []
        tpots: list[float] = []
        all_tpots: list[float] = []
        ttfts: list[float] = []
        e2els: list[float] = []
        for i in range(len(outputs)):
            if outputs[i].success:
                output_len = outputs[i].output_tokens

                if not output_len:
                    # We use the tokenizer to count the number of output tokens
                    # for some serving backends instead of looking at
                    # len(outputs[i].itl) since multiple output tokens may be
                    # bundled together
                    # Note : this may inflate the output token count slightly
                    output_len = len(
                        tokenizer(
                            outputs[i].generated_text, add_special_tokens=False
                        ).input_ids
                    )
                actual_output_lens.append(output_len)
                total_input += input_requests[i].prompt_len
                tpot = 0
                if output_len > 1:
                    latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                    tpot = latency_minus_ttft / (output_len - 1)
                    tpots.append(tpot)
                # Note: if output_len <= 1, we regard tpot as 0 for goodput
                all_tpots.append(tpot)
                itls += outputs[i].itl
                ttfts.append(outputs[i].ttft)
                e2els.append(outputs[i].latency)
                completed += 1
            else:
                actual_output_lens.append(0)

        if goodput_config_dict:
            valid_metrics = []
            slo_values = []

            if "ttft" in goodput_config_dict:
                valid_metrics.append(ttfts)
                slo_values.append(
                    goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
                )
            if "tpot" in goodput_config_dict:
                valid_metrics.append(all_tpots)
                slo_values.append(
                    goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
                )
            if "e2el" in goodput_config_dict:
                valid_metrics.append(e2els)
                slo_values.append(
                    goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
                )

            for req_metric in zip(*valid_metrics):
                is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
                if is_good_req:
                    good_completed += 1

        if completed == 0:
            warnings.warn(
                "All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments.",
                stacklevel=2,
            )

        # Calculate max output tokens per second metric
        max_output_tokens_per_s = 0.0
        max_concurrent_requests = 0

        # Find the time range across all successful requests
        successful_outputs = [output for output in outputs if output.success]
        failed_outputs = [output for output in outputs if not output.success]
        if successful_outputs:
            min_start_time = min(output.start_time for output in successful_outputs)
            max_end_time = max(
                output.start_time + output.latency for output in successful_outputs
            )

            # Create second buckets (ceiling to ensure we capture all time)
            duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
            tokens_per_second = np.zeros(duration_seconds)
            concurrent_requests_per_second = np.zeros(duration_seconds)

            for i, output in enumerate(successful_outputs):
                # Calculate token generation timestamp using
                # start_time, ttft, and itl
                token_times = [output.start_time + output.ttft]
                current_time = token_times[0]
                for itl_value in output.itl:
                    current_time += itl_value
                    token_times.append(current_time)

                # Add tokens to second buckets
                for token_time in token_times:
                    second_bucket = int(token_time - min_start_time)
                    if 0 <= second_bucket < duration_seconds:
                        tokens_per_second[second_bucket] += 1

                # Track concurrent requests for each second this request was active
                request_start_second = int(output.start_time - min_start_time)
                request_end_second = int(
                    (output.start_time + output.latency) - min_start_time
                )

                for second in range(request_start_second, request_end_second + 1):
                    concurrent_requests_per_second[second] += 1

            # Find the maximum tokens per second and corresponding
            # concurrent requests
            if len(tokens_per_second) > 0:
                max_output_tokens_per_s = float(np.max(tokens_per_second))
                max_concurrent_requests = int(np.max(concurrent_requests_per_second))

            if TERM_PLOTLIB_AVAILABLE:
                import termplotlib as tpl

                fig = tpl.figure()
                fig.plot(
                    np.arange(len(tokens_per_second)),
                    tokens_per_second,
                    title="Output tokens per second",
                )
                fig.plot(
                    np.arange(len(concurrent_requests_per_second)),
                    concurrent_requests_per_second,
                    title="Concurrent requests per second",
                )
                fig.show()
            else:
                print("tip: install termplotlib and gnuplot to plot the metrics")

        metrics = BenchmarkMetrics(
            completed=completed,
            failed=len(failed_outputs),
            total_input=total_input,
            total_output=sum(actual_output_lens),
            request_throughput=completed / dur_s,
            request_goodput=good_completed / dur_s,
            output_throughput=sum(actual_output_lens) / dur_s,
            total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
            mean_ttft_ms=np.mean(ttfts or 0)
            * 1000,  # ttfts is empty if streaming is not supported by the endpoint
            std_ttft_ms=np.std(ttfts or 0) * 1000,
            median_ttft_ms=np.median(ttfts or 0) * 1000,
            percentiles_ttft_ms=[
                (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
            ],
            mean_tpot_ms=np.mean(tpots or 0) * 1000,
            std_tpot_ms=np.std(tpots or 0) * 1000,
            median_tpot_ms=np.median(tpots or 0) * 1000,
            percentiles_tpot_ms=[
                (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
            ],
            mean_itl_ms=np.mean(itls or 0) * 1000,
            std_itl_ms=np.std(itls or 0) * 1000,
            median_itl_ms=np.median(itls or 0) * 1000,
            percentiles_itl_ms=[
                (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
            ],
            mean_e2el_ms=np.mean(e2els or 0) * 1000,
            std_e2el_ms=np.std(e2els or 0) * 1000,
            median_e2el_ms=np.median(e2els or 0) * 1000,
            percentiles_e2el_ms=[
                (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
            ],
            max_output_tokens_per_s=max_output_tokens_per_s,
            max_concurrent_requests=max_concurrent_requests,
        )

        return metrics, actual_output_lens

    def calculate_metrics_for_embeddings(
        outputs: list[RequestFuncOutput],
        dur_s: float,
        selected_percentiles: list[float],
    ) -> EmbedBenchmarkMetrics:
        """Calculate the metrics for the embedding requests.

        Args:
            outputs: The outputs of the requests.
            dur_s: The duration of the benchmark.
            selected_percentiles: The percentiles to select.

        Returns:
            The calculated benchmark metrics.
        """
        total_input = 0
        completed = 0
        failed = 0
        e2els: list[float] = []
        for i in range(len(outputs)):
            if outputs[i].success:
                e2els.append(outputs[i].latency)
                completed += 1
                total_input += outputs[i].prompt_len
            else:
                failed += 1

        if completed == 0:
            warnings.warn(
                "All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments.",
                stacklevel=2,
            )
        metrics = EmbedBenchmarkMetrics(
            completed=completed,
            failed=failed,
            total_input=total_input,
            request_throughput=completed / dur_s,
            total_token_throughput=total_input / dur_s,
            mean_e2el_ms=np.mean(e2els or 0) * 1000,
            std_e2el_ms=np.std(e2els or 0) * 1000,
            median_e2el_ms=np.median(e2els or 0) * 1000,
            percentiles_e2el_ms=[
                (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
            ],
        )
        return metrics

    async def fetch_spec_decode_metrics(
        self, session: aiohttp.ClientSession
    ) -> SpecDecodeMetrics | None:
        """Fetch speculative decoding metrics from the server's Prometheus endpoint.
        Returns None if speculative decoding is not enabled or
        metrics are not available.
        """
        metrics_url = f"{self.config.base_url}/metrics"
        try:
            async with session.get(metrics_url) as response:
                if response.status != 200:
                    return None
                text = await response.text()

                num_drafts = 0
                num_draft_tokens = 0
                num_accepted_tokens = 0
                accepted_per_pos: dict[int, int] = {}
                found_spec_decode = False

                for line in text.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if line.startswith("vllm:spec_decode"):
                        found_spec_decode = True
                        parts = line.split()
                        if parts:
                            with contextlib.suppress(ValueError):
                                if "num_drafts" in line:
                                    num_drafts += int(float(parts[-1]))
                                elif "num_draft_tokens" in line:
                                    num_draft_tokens += int(float(parts[-1]))
                                elif "num_accepted_tokens_per_pos" in line:
                                    pos_label = 'position="'
                                    if pos_label in line:
                                        start = line.index(pos_label) + len(pos_label)
                                        end = line.index('"', start)
                                        pos = int(line[start:end])
                                        val = int(float(parts[-1]))
                                        accepted_per_pos[pos] = (
                                            accepted_per_pos.get(pos, 0) + val
                                        )
                                elif "num_accepted_tokens" in line:
                                    num_accepted_tokens += int(float(parts[-1]))

                if not found_spec_decode:
                    return None

                return SpecDecodeMetrics(
                    num_drafts=num_drafts,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted_tokens,
                    accepted_per_pos=accepted_per_pos,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

    def init_result(
        self,
        metrics,
        benchmark_duration,
        outputs,
        actual_output_lens,
        rps_change_events,
    ):
        if isinstance(metrics, BenchmarkMetrics):
            result = {
                "duration": benchmark_duration,
                "completed": metrics.completed,
                "failed": metrics.failed,
                "total_input_tokens": metrics.total_input,
                "total_output_tokens": metrics.total_output,
                "request_throughput": metrics.request_throughput,
                "request_goodput": metrics.request_goodput
                if self.config.goodput_config_dict
                else None,
                "output_throughput": metrics.output_throughput,
                "total_token_throughput": metrics.total_token_throughput,
                "input_lens": [output.prompt_len for output in outputs],
                "output_lens": actual_output_lens,
                "ttfts": [output.ttft for output in outputs],
                "itls": [output.itl for output in outputs],
                "generated_texts": [output.generated_text for output in outputs],
                "errors": [output.error for output in outputs],
                "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
                "max_concurrent_requests": metrics.max_concurrent_requests,
            }
        else:
            result = {
                "duration": benchmark_duration,
                "completed": metrics.completed,
                "total_input_tokens": metrics.total_input,
                "request_throughput": metrics.request_throughput,
                "total_token_throughput": metrics.total_token_throughput,
                "input_lens": [output.prompt_len for output in outputs],
                "errors": [output.error for output in outputs],
            }

        if rps_change_events:
            result["rps_change_events"] = rps_change_events

        if self.spec_decode_stats is not None:
            result["spec_decode_acceptance_rate"] = self.spec_decode_stats[
                "acceptance_rate"
            ]
            result["spec_decode_acceptance_length"] = self.spec_decode_stats[
                "acceptance_length"
            ]
            result["spec_decode_num_drafts"] = int(self.spec_decode_stats["num_drafts"])
            result["spec_decode_draft_tokens"] = int(
                self.spec_decode_stats["draft_tokens"]
            )
            result["spec_decode_accepted_tokens"] = int(
                self.spec_decode_stats["accepted_tokens"]
            )
            result["spec_decode_per_position_acceptance_rates"] = (
                self.spec_decode_stats.get("per_position_acceptance_rates", [])
            )

    def process_metrics(self, metrics, benchmark_duration):
        print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
        if self.config.max_concurrency is not None:
            print(
                "{:<40} {:<10}".format(
                    "Maximum request concurrency:", self.config.max_concurrency
                )
            )
        if self.config.request_rate != float("inf"):
            print(
                "{:<40} {:<10.2f}".format(
                    "Request rate configured (RPS):", self.config.request_rate
                )
            )
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        if isinstance(metrics, BenchmarkMetrics):
            print(
                "{:<40} {:<10}".format("Total generated tokens:", metrics.total_output)
            )
        print(
            "{:<40} {:<10.2f}".format(
                "Request throughput (req/s):", metrics.request_throughput
            )
        )
        if self.config.goodput_config_dict:
            print(
                "{:<40} {:<10.2f}".format(
                    "Request goodput (req/s):", metrics.request_goodput
                )
            )
        if isinstance(metrics, BenchmarkMetrics):
            print(
                "{:<40} {:<10.2f}".format(
                    "Output token throughput (tok/s):", metrics.output_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Peak output token throughput (tok/s):",
                    metrics.max_output_tokens_per_s,
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Peak concurrent requests:", metrics.max_concurrent_requests
                )
            )
        print(
            "{:<40} {:<10.2f}".format(
                "Total token throughput (tok/s):", metrics.total_token_throughput
            )
        )
        result = {}

        if self.config.task_type == TaskType.GENERATION:
            result |= self.process_one_metric(
                metrics, "ttft", "TTFT", "Time to First Token"
            )
            result |= self.process_one_metric(
                metrics,
                "tpot",
                "TPOT",
                "Time per Output Token (excl. 1st token)",
            )
            result |= self.process_one_metric(
                metrics, "itl", "ITL", "Inter-token Latency"
            )
        result |= self.process_one_metric(metrics, "e2el", "E2EL", "End-to-end Latency")

        print("=" * 50)

        return result

    def _process_one_metric(
        self,
        metrics,
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in self.config.selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result = {}
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value
        return result

    @classmethod
    def get_samples(cls, args, tokenizer: TokenizerLike) -> list[SampleRequest]:
        from vllm.benchmarks.datasets import get_samples

        return get_samples(args, tokenizer)

    @classmethod
    def get_async_request_funcs_map():
        return ASYNC_REQUEST_FUNCS

    @classmethod
    def get_openai_compatible_backends():
        return OPENAI_COMPATIBLE_BACKENDS
