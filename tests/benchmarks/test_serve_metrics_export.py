# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.benchmarks import serve as benchmark_serve


def test_parse_benchmark_load_metrics_and_build_result():
    before_text = """
# HELP vllm:num_requests_running Running requests.
vllm:num_requests_running{engine="0",model_name="qwen"} 2
vllm:num_requests_waiting{engine="0",model_name="qwen"} 1
vllm:kv_cache_usage_perc{engine="0",model_name="qwen"} 0.35
vllm:available_kv_cache_memory_bytes{engine="0",model_name="qwen"} 4294967296
vllm:prefix_cache_queries{engine="0",model_name="qwen"} 100
vllm:prefix_cache_hits{engine="0",model_name="qwen"} 45
vllm:external_prefix_cache_queries{engine="0",model_name="qwen"} 10
vllm:external_prefix_cache_hits{engine="0",model_name="qwen"} 2
"""
    after_text = """
vllm:num_requests_running{engine="0",model_name="qwen"} 5
vllm:num_requests_running{engine="1",model_name="qwen"} 3
vllm:num_requests_waiting{engine="0",model_name="qwen"} 4
vllm:num_requests_waiting{engine="1",model_name="qwen"} 2
vllm:kv_cache_usage_perc{engine="0",model_name="qwen"} 0.72
vllm:kv_cache_usage_perc{engine="1",model_name="qwen"} 0.64
vllm:available_kv_cache_memory_bytes{engine="0",model_name="qwen"} 8589934592
vllm:available_kv_cache_memory_bytes{engine="1",model_name="qwen"} 7516192768
vllm:prefix_cache_queries{engine="0",model_name="qwen"} 180
vllm:prefix_cache_queries{engine="1",model_name="qwen"} 40
vllm:prefix_cache_hits{engine="0",model_name="qwen"} 105
vllm:prefix_cache_hits{engine="1",model_name="qwen"} 20
vllm:external_prefix_cache_queries{engine="0",model_name="qwen"} 25
vllm:external_prefix_cache_hits{engine="0",model_name="qwen"} 8
"""

    before = benchmark_serve.parse_benchmark_load_metrics(before_text)
    after = benchmark_serve.parse_benchmark_load_metrics(after_text)

    assert before is not None
    assert after is not None
    assert after.num_requests_running == 8.0
    assert after.num_requests_waiting == 6.0
    assert after.kv_cache_usage_perc == 0.72
    assert after.available_kv_cache_memory_bytes == 8589934592.0

    exported = benchmark_serve.build_benchmark_load_result(before, after)

    assert exported == {
        "num_requests_running": 8.0,
        "num_requests_waiting": 6.0,
        "kv_cache_usage_perc": 0.72,
        "available_kv_cache_memory_bytes": 8589934592.0,
        "prefix_cache_hit_rate": 80.0 / 120.0,
        "external_prefix_cache_hit_rate": 6.0 / 15.0,
    }
