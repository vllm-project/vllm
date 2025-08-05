# SPDX-License-Identifier: Apache-2.0
import math

import server.vllm_autocalc_rules as rules


def test_calc_TENSOR_PARALLEL_SIZE():
    ctx = {'TENSOR_PARALLEL_SIZE': 4}
    assert rules.calc_TENSOR_PARALLEL_SIZE(ctx) == 4


def test_calc_MAX_MODEL_LEN():
    ctx = {'MAX_MODEL_LEN': 1024}
    assert rules.calc_MAX_MODEL_LEN(ctx) == 1024


def test_calc_PT_HPU_ENABLE_LAZY_COLLECTIVES():
    ctx = {'TENSOR_PARALLEL_SIZE': 2}
    assert rules.calc_PT_HPU_ENABLE_LAZY_COLLECTIVES(ctx) is True


def test_calc_MODEL_MEM_FROM_CONFIG():
    ctx = {'MODEL_MEM_FROM_CONFIG': "123.5"}
    assert rules.calc_MODEL_MEM_FROM_CONFIG(ctx) == 123.5


def test_calc_DEVICE_HPU_MEM():
    ctx = {'HPU_MEM': {'GAUDI2': 96}, 'DEVICE_NAME': 'GAUDI2'}
    assert rules.calc_DEVICE_HPU_MEM(ctx) == 96


def test_calc_TOTAL_GPU_MEM():
    ctx = {'DEVICE_HPU_MEM': 96, 'TENSOR_PARALLEL_SIZE': 4}
    assert rules.calc_TOTAL_GPU_MEM(ctx) == 384


def test_calc_MODEL_MEM_IN_GB():
    ctx = {
        'MODEL_MEM_FROM_CONFIG': 2 * 1024**3,
        'QUANT_DTYPE': 1,
        'MODEL_DTYPE': 1
    }
    assert rules.calc_MODEL_MEM_IN_GB(ctx) == 2.0


def test_calc_USABLE_MEM():
    ctx = {
        'TOTAL_GPU_MEM': 384,
        'TENSOR_PARALLEL_SIZE': 4,
        'UNAVAILABLE_MEM_ABS': 10,
        'MODEL_MEM_IN_GB': 8,
        'PROFILER_MEM_OVERHEAD': 2
    }
    expected = ((384 / 4) - 10 - (8 / 4) - 2)
    assert rules.calc_USABLE_MEM(ctx) == expected


def test_calc_GPU_MEMORY_UTIL_TEMP():
    ctx = {'GPU_FREE_MEM_TARGET': 10, 'USABLE_MEM': 100}
    assert rules.calc_GPU_MEMORY_UTIL_TEMP(ctx) == 0.9


def test_calc_GPU_MEM_UTILIZATION():
    ctx = {'GPU_MEMORY_UTIL_TEMP': 0.987}
    assert rules.calc_GPU_MEM_UTILIZATION(ctx) == math.floor(0.987 * 100) / 100


def test_calc_KV_CACHE_PER_SEQ():
    ctx = {
        'MAX_MODEL_LEN': 128,
        'NUM_HIDDEN_LAYERS': 2,
        'HIDDEN_SIZE': 4,
        'NUM_KEY_VALUE_HEADS': 2,
        'CACHE_DTYPE_BYTES': 2,
        'NUM_ATTENTION_HEADS': 2
    }
    expected = ((2 * 128 * 2 * 4 * 2 * 2) / 2) / (1024 * 1024 * 1024)
    assert rules.calc_KV_CACHE_PER_SEQ(ctx) == expected


def test_calc_EST_MAX_NUM_SEQS():
    ctx = {
        'TENSOR_PARALLEL_SIZE': 4,
        'USABLE_MEM': 100,
        'GPU_MEM_UTILIZATION': 0.9,
        'KV_CACHE_PER_SEQ': 0.5
    }
    expected = (4 * 100 * 0.9) / 0.5
    assert rules.calc_EST_MAX_NUM_SEQS(ctx) == expected


def test_calc_EST_HPU_BLOCKS():
    ctx = {'MAX_MODEL_LEN': 128, 'EST_MAX_NUM_SEQS': 32, 'BLOCK_SIZE': 16}
    expected = (128 * 32) / 16
    assert rules.calc_EST_HPU_BLOCKS(ctx) == expected


def test_calc_DECODE_BS_RAMP_GRAPHS():
    ctx = {'VLLM_DECODE_BS_BUCKET_STEP': 16, 'VLLM_DECODE_BS_BUCKET_MIN': 2}
    expected = 1 + int(math.log(16 / 2, 2))
    assert rules.calc_DECODE_BS_RAMP_GRAPHS(ctx) == expected


def test_calc_DECODE_BS_STEP_GRAPHS():
    ctx = {'EST_MAX_NUM_SEQS': 64, 'VLLM_DECODE_BS_BUCKET_STEP': 8}
    expected = max(0, int(1 + (64 - 8) / 8))
    assert rules.calc_DECODE_BS_STEP_GRAPHS(ctx) == expected


def test_calc_DECODE_BLOCK_RAMP_GRAPHS():
    ctx = {
        'VLLM_DECODE_BLOCK_BUCKET_STEP': 16,
        'VLLM_DECODE_BLOCK_BUCKET_MIN': 2
    }
    expected = 1 + int(math.log(16 / 2, 2))
    assert rules.calc_DECODE_BLOCK_RAMP_GRAPHS(ctx) == expected


def test_calc_DECODE_BLOCK_STEP_GRAPHS():
    ctx = {'EST_HPU_BLOCKS': 64, 'VLLM_DECODE_BLOCK_BUCKET_STEP': 8}
    expected = max(0, int(1 + (64 - 8) / 8))
    assert rules.calc_DECODE_BLOCK_STEP_GRAPHS(ctx) == expected


def test_calc_NUM_DECODE_GRAPHS():
    ctx = {
        'DECODE_BS_RAMP_GRAPHS': 2,
        'DECODE_BS_STEP_GRAPHS': 3,
        'DECODE_BLOCK_RAMP_GRAPHS': 4,
        'DECODE_BLOCK_STEP_GRAPHS': 5
    }
    expected = (2 + 3) * (4 + 5)
    assert rules.calc_NUM_DECODE_GRAPHS(ctx) == expected


def test_calc_PROMPT_BS_RAMP_GRAPHS():
    ctx = {
        'MAX_NUM_PREFILL_SEQS': 16,
        'VLLM_PROMPT_BS_BUCKET_STEP': 8,
        'VLLM_PROMPT_BS_BUCKET_MIN': 2
    }
    expected = 1 + int(math.log(min(16, 8) / 2, 2))
    assert rules.calc_PROMPT_BS_RAMP_GRAPHS(ctx) == expected


def test_calc_PROMPT_BS_STEP_GRAPHS():
    ctx = {'MAX_NUM_PREFILL_SEQS': 32, 'VLLM_PROMPT_BS_BUCKET_STEP': 8}
    expected = max(0, int(1 + (32 - 8) / 8))
    assert rules.calc_PROMPT_BS_STEP_GRAPHS(ctx) == expected


def test_calc_PROMPT_SEQ_RAMP_GRAPHS():
    ctx = {'VLLM_PROMPT_SEQ_BUCKET_STEP': 16, 'VLLM_PROMPT_SEQ_BUCKET_MIN': 2}
    expected = 1 + int(math.log(16 / 2, 2))
    assert rules.calc_PROMPT_SEQ_RAMP_GRAPHS(ctx) == expected


def test_calc_PROMPT_SEQ_STEP_GRAPHS():
    ctx = {'MAX_MODEL_LEN': 64, 'VLLM_PROMPT_SEQ_BUCKET_STEP': 8}
    expected = int(1 + (64 - 8) / 8)
    assert rules.calc_PROMPT_SEQ_STEP_GRAPHS(ctx) == expected


def test_calc_EST_NUM_PROMPT_GRAPHS():
    ctx = {
        'PROMPT_BS_RAMP_GRAPHS': 2,
        'PROMPT_BS_STEP_GRAPHS': 3,
        'PROMPT_SEQ_RAMP_GRAPHS': 4,
        'PROMPT_SEQ_STEP_GRAPHS': 5
    }
    expected = ((2 + 3) * (4 + 5)) / 2
    assert rules.calc_EST_NUM_PROMPT_GRAPHS(ctx) == expected


def test_calc_EST_GRAPH_PROMPT_RATIO():
    ctx = {'EST_NUM_PROMPT_GRAPHS': 10, 'NUM_DECODE_GRAPHS': 30}
    expected = math.ceil(10 / (10 + 30) * 100) / 100
    assert rules.calc_EST_GRAPH_PROMPT_RATIO(ctx) == expected


def test_calc_VLLM_GRAPH_PROMPT_RATIO():
    ctx = {'EST_GRAPH_PROMPT_RATIO': 0.5}
    expected = math.ceil(min(max(0.5, 0.1), 0.9) * 10) / 10
    assert rules.calc_VLLM_GRAPH_PROMPT_RATIO(ctx) == expected


def test_calc_DECODE_GRAPH_TARGET_GB():
    ctx = {'NUM_DECODE_GRAPHS': 10, 'APPROX_MEM_PER_GRAPH_MB': 512}
    expected = math.ceil(10 * 512 / 1024 * 10) / 10
    assert rules.calc_DECODE_GRAPH_TARGET_GB(ctx) == expected


def test_calc_EST_GRAPH_RESERVE_MEM():
    ctx = {
        'DECODE_GRAPH_TARGET_GB': 5,
        'USABLE_MEM': 10,
        'GPU_MEM_UTILIZATION': 0.8,
        'VLLM_GRAPH_PROMPT_RATIO': 0.2
    }
    expected = math.ceil(5 / (10 * 0.8 * (1 - 0.2)) * 100) / 100
    assert rules.calc_EST_GRAPH_RESERVE_MEM(ctx) == expected


def test_calc_VLLM_GRAPH_RESERVED_MEM():
    ctx = {'EST_GRAPH_RESERVE_MEM': 0.3}
    expected = min(max(0.3, 0.01), 0.5)
    assert rules.calc_VLLM_GRAPH_RESERVED_MEM(ctx) == expected


def test_calc_KV_CACHE_MEM():
    ctx = {
        'USABLE_MEM': 10,
        'GPU_MEM_UTILIZATION': 0.8,
        'VLLM_GRAPH_RESERVED_MEM': 0.2
    }
    expected = 10 * 0.8 * (1 - 0.2)
    assert rules.calc_KV_CACHE_MEM(ctx) == expected


def test_calc_VLLM_DECODE_BLOCK_BUCKET_MAX():
    ctx = {'MAX_NUM_SEQS': 16, 'MAX_MODEL_LEN': 128}
    expected = max(128, math.ceil((16 * 128) / 128))
    assert rules.calc_VLLM_DECODE_BLOCK_BUCKET_MAX(ctx) == expected


def test_calc_VLLM_PROMPT_SEQ_BUCKET_MAX():
    ctx = {'MAX_MODEL_LEN': 4096}
    assert rules.calc_VLLM_PROMPT_SEQ_BUCKET_MAX(ctx) == 4096
