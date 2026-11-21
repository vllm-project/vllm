import typing
from dataclasses import fields
from typing import Union

import numpy as np
from vllm_simulator.simulation.types import RequestStats, SchedulerConfig
from vllm_simulator.spec.accelerator import AcceleratorInfo
from vllm_simulator.time_predictor.aisimulate import get_perf_model

if typing.TYPE_CHECKING:
    from vllm.config import ModelConfig


def profile_device_available_bytes(
    model: "ModelConfig", device: AcceleratorInfo, scheduler_config: SchedulerConfig
) -> int:

    perf_model = get_perf_model(scheduler_config, model)
    weights = 0
    for op in perf_model.context_ops:
        weights += op.get_weights()
    # Count weights on a single GPU
    weights /= perf_model.config.pp_size
    framework_reserved_mem_gb = 1.4
    rest_memory = (
        scheduler_config.mem_fraction_static * device.hbm_capacity_gb
        - framework_reserved_mem_gb
    ) * (1 << 30) - weights
    return int(rest_memory)


def calc_metrics(
    requests: list[Union[RequestStats, dict]], duration_start_time: float = 0.0
) -> dict:
    stat_fields = {f.name for f in fields(RequestStats)}
    requests = [
        RequestStats(**{k: v for k, v in req.items() if k in stat_fields})
        if isinstance(req, dict)
        else req
        for req in requests
    ]

    ttfts = []
    tpots = []
    itls = []
    e2e_latencies = []
    total_dur_s = 1e-9
    total_input = 0
    total_output = 0
    completed = 0
    total_local_cache_hit = 0
    total_external_kv_transfer = 0
    queue_durs = []
    for req in requests:
        completed += 1
        ttfts.append(req.gen_token_latencies[0])
        queue_durs.append(req.queue_end - req.queue_start)
        if len(req.gen_token_latencies) > 1:
            # output length > 1
            tpots.append(np.mean(req.gen_token_latencies[1:]))
        itls.extend(req.gen_token_latencies[1:])
        e2e_latencies.append(sum(req.gen_token_latencies))
        total_dur_s = max(total_dur_s, req.last_event_time - duration_start_time)
        total_input += req.input_length
        total_output += req.output_length
        total_local_cache_hit += req.local_kv_hit_len
        total_external_kv_transfer += req.ext_kv_hit_len
    total_cached_tokens = total_local_cache_hit + total_external_kv_transfer
    return {
        "num_requests": len(requests),
        "completed": completed,
        "total_input": total_input,
        "total_output": total_output,
        "duration": total_dur_s,
        "request_throughput": len(requests) / total_dur_s,
        "input_throughput": total_input / total_dur_s,
        "output_throughput": total_output / total_dur_s,
        "total_throughput": (total_input + total_output) / total_dur_s,
        # Prompt-token breakdown by source, named after vLLM PromptTokenStats
        # (vllm:prompt_tokens_cached / vllm:prompt_tokens_by_source).
        "prompt_tokens_cached_ratio": (
            0 if total_input == 0 else total_cached_tokens / total_input
        ),
        "prompt_tokens_local_cache_hit_ratio": (
            0 if total_input == 0 else total_local_cache_hit / total_input
        ),
        "prompt_tokens_external_kv_transfer_ratio": (
            0 if total_input == 0 else total_external_kv_transfer / total_input
        ),
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "p90_ttft_ms": np.percentile(ttfts or 0, 90) * 1000,
        "p95_ttft_ms": np.percentile(ttfts or 0, 95) * 1000,
        "p99_ttft_ms": np.percentile(ttfts or 0, 99) * 1000,
        "mean_queue_ms": np.mean(queue_durs or 0) * 1000,
        "mean_tpot_ms": np.mean(tpots or 0) * 1000,
        "median_tpot_ms": np.median(tpots or 0) * 1000,
        "std_tpot_ms": np.std(tpots or 0) * 1000,
        "p90_tpot_ms": np.percentile(tpots or 0, 90) * 1000,
        "p95_tpot_ms": np.percentile(tpots or 0, 95) * 1000,
        "p99_tpot_ms": np.percentile(tpots or 0, 99) * 1000,
        "mean_itl_ms": np.mean(itls or 0) * 1000,
        "median_itl_ms": np.median(itls or 0) * 1000,
        "std_itl_ms": np.std(itls or 0) * 1000,
        "p90_itl_ms": np.percentile(itls or 0, 90) * 1000,
        "p95_itl_ms": np.percentile(itls or 0, 95) * 1000,
        "p99_itl_ms": np.percentile(itls or 0, 99) * 1000,
        "max_itl_ms": np.max(itls or 0) * 1000,
        "mean_e2e_latency_ms": np.mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": np.median(e2e_latencies) * 1000,
        "std_e2e_latency_ms": np.std(e2e_latencies) * 1000,
        "p90_e2e_latency_ms": np.percentile(e2e_latencies or 0, 90) * 1000,
        "p95_e2e_latency_ms": np.percentile(e2e_latencies or 0, 95) * 1000,
        "p99_e2e_latency_ms": np.percentile(e2e_latencies or 0, 99) * 1000,
        "time_cost": -1,  # Updated by external benchmark caller
    }
