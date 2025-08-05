#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import math


def calc_TENSOR_PARALLEL_SIZE(ctx):
    return max(1, min(8, ctx['TENSOR_PARALLEL_SIZE']))


def calc_MAX_MODEL_LEN(ctx):
    return max(1, ctx['MAX_MODEL_LEN'])


def calc_PT_HPU_ENABLE_LAZY_COLLECTIVES(ctx):
    return ctx['TENSOR_PARALLEL_SIZE'] > 1


def calc_MODEL_MEM_FROM_CONFIG(ctx):
    return float(ctx.get('MODEL_MEM_FROM_CONFIG'))


def calc_DEVICE_HPU_MEM(ctx):
    return ctx['HPU_MEM'][ctx['DEVICE_NAME']]


def calc_TOTAL_GPU_MEM(ctx):
    return ctx['DEVICE_HPU_MEM'] * ctx['TENSOR_PARALLEL_SIZE']


def calc_MODEL_MEM_IN_GB(ctx):
    return (ctx['MODEL_MEM_FROM_CONFIG'] * ctx['QUANT_DTYPE'] /
            ctx['MODEL_DTYPE']) / (1024 * 1024 * 1024)


def calc_USABLE_MEM(ctx):
    return ((ctx['TOTAL_GPU_MEM'] / ctx['TENSOR_PARALLEL_SIZE']) -
            ctx['UNAVAILABLE_MEM_ABS'] -
            (ctx['MODEL_MEM_IN_GB'] / ctx['TENSOR_PARALLEL_SIZE']) -
            ctx['PROFILER_MEM_OVERHEAD'])


def calc_GPU_MEMORY_UTIL_TEMP(ctx):
    return (1 - ctx['GPU_FREE_MEM_TARGET'] / ctx['USABLE_MEM'])


def calc_GPU_MEM_UTILIZATION(ctx):
    return math.floor(ctx['GPU_MEMORY_UTIL_TEMP'] * 100) / 100


def calc_KV_CACHE_PER_SEQ(ctx):
    return ((2 * ctx['MAX_MODEL_LEN'] * ctx['NUM_HIDDEN_LAYERS'] *
             ctx['HIDDEN_SIZE'] * ctx['NUM_KEY_VALUE_HEADS'] *
             ctx['CACHE_DTYPE_BYTES']) /
            ctx['NUM_ATTENTION_HEADS']) / (1024 * 1024 * 1024)


def calc_EST_MAX_NUM_SEQS(ctx):
    return (ctx['TENSOR_PARALLEL_SIZE'] * ctx['USABLE_MEM'] *
            ctx['GPU_MEM_UTILIZATION'] / ctx['KV_CACHE_PER_SEQ'])


def calc_EST_HPU_BLOCKS(ctx):
    return (ctx['MAX_MODEL_LEN'] * ctx['EST_MAX_NUM_SEQS'] / ctx['BLOCK_SIZE'])


def calc_DECODE_BS_RAMP_GRAPHS(ctx):
    return 1 + int(
        math.log(
            ctx['VLLM_DECODE_BS_BUCKET_STEP'] /
            ctx['VLLM_DECODE_BS_BUCKET_MIN'], 2))


def calc_DECODE_BS_STEP_GRAPHS(ctx):
    return max(
        0,
        int(1 + (ctx['EST_MAX_NUM_SEQS'] - ctx['VLLM_DECODE_BS_BUCKET_STEP']) /
            ctx['VLLM_DECODE_BS_BUCKET_STEP']))


def calc_DECODE_BLOCK_RAMP_GRAPHS(ctx):
    return 1 + int(
        math.log(
            ctx['VLLM_DECODE_BLOCK_BUCKET_STEP'] /
            ctx['VLLM_DECODE_BLOCK_BUCKET_MIN'], 2))


def calc_DECODE_BLOCK_STEP_GRAPHS(ctx):
    return max(
        0,
        int(1 +
            (ctx['EST_HPU_BLOCKS'] - ctx['VLLM_DECODE_BLOCK_BUCKET_STEP']) /
            ctx['VLLM_DECODE_BLOCK_BUCKET_STEP']))


def calc_NUM_DECODE_GRAPHS(ctx):
    return (
        (ctx['DECODE_BS_RAMP_GRAPHS'] + ctx['DECODE_BS_STEP_GRAPHS']) *
        (ctx['DECODE_BLOCK_RAMP_GRAPHS'] + ctx['DECODE_BLOCK_STEP_GRAPHS']))


def calc_PROMPT_BS_RAMP_GRAPHS(ctx):
    return 1 + int(
        math.log(
            min(ctx['MAX_NUM_PREFILL_SEQS'], ctx['VLLM_PROMPT_BS_BUCKET_STEP'])
            / ctx['VLLM_PROMPT_BS_BUCKET_MIN'], 2))


def calc_PROMPT_BS_STEP_GRAPHS(ctx):
    return max(
        0,
        int(1 +
            (ctx['MAX_NUM_PREFILL_SEQS'] - ctx['VLLM_PROMPT_BS_BUCKET_STEP']) /
            ctx['VLLM_PROMPT_BS_BUCKET_STEP']))


def calc_PROMPT_SEQ_RAMP_GRAPHS(ctx):
    return 1 + int(
        math.log(
            ctx['VLLM_PROMPT_SEQ_BUCKET_STEP'] /
            ctx['VLLM_PROMPT_SEQ_BUCKET_MIN'], 2))


def calc_PROMPT_SEQ_STEP_GRAPHS(ctx):
    return int(1 +
               (ctx['MAX_MODEL_LEN'] - ctx['VLLM_PROMPT_SEQ_BUCKET_STEP']) /
               ctx['VLLM_PROMPT_SEQ_BUCKET_STEP'])


def calc_EST_NUM_PROMPT_GRAPHS(ctx):
    return ((ctx['PROMPT_BS_RAMP_GRAPHS'] + ctx['PROMPT_BS_STEP_GRAPHS']) *
            (ctx['PROMPT_SEQ_RAMP_GRAPHS'] + ctx['PROMPT_SEQ_STEP_GRAPHS']) /
            2)


def calc_EST_GRAPH_PROMPT_RATIO(ctx):
    return math.ceil(
        ctx['EST_NUM_PROMPT_GRAPHS'] /
        (ctx['EST_NUM_PROMPT_GRAPHS'] + ctx['NUM_DECODE_GRAPHS']) * 100) / 100


def calc_VLLM_GRAPH_PROMPT_RATIO(ctx):
    return math.ceil(
        min(max(ctx['EST_GRAPH_PROMPT_RATIO'], 0.1), 0.9) * 10) / 10


def calc_DECODE_GRAPH_TARGET_GB(ctx):
    return math.ceil(ctx['NUM_DECODE_GRAPHS'] *
                     ctx['APPROX_MEM_PER_GRAPH_MB'] / 1024 * 10) / 10


def calc_EST_GRAPH_RESERVE_MEM(ctx):
    return math.ceil(ctx['DECODE_GRAPH_TARGET_GB'] /
                     (ctx['USABLE_MEM'] * ctx['GPU_MEM_UTILIZATION'] *
                      (1 - ctx['VLLM_GRAPH_PROMPT_RATIO'])) * 100) / 100


def calc_VLLM_GRAPH_RESERVED_MEM(ctx):
    return min(max(ctx['EST_GRAPH_RESERVE_MEM'], 0.01), 0.5)


def calc_KV_CACHE_MEM(ctx):
    return (ctx['USABLE_MEM'] * ctx['GPU_MEM_UTILIZATION'] *
            (1 - ctx['VLLM_GRAPH_RESERVED_MEM']))


def calc_MAX_NUM_SEQS(ctx):
    # If user provided, just clamp to min 1
    if ctx.get('MAX_NUM_SEQS') is not None:
        return max(1, ctx['MAX_NUM_SEQS'])
    # Otherwise, calculate
    val = (ctx['TENSOR_PARALLEL_SIZE'] * ctx['KV_CACHE_MEM'] /
           ctx['KV_CACHE_PER_SEQ'])
    if ctx['DTYPE'] == 'fp8':
        val = (max(1, math.floor(val / ctx['VLLM_DECODE_BS_BUCKET_STEP'])) *
               ctx['VLLM_DECODE_BS_BUCKET_STEP'])
    else:
        val = (math.ceil(val / ctx['VLLM_DECODE_BS_BUCKET_STEP']) *
               ctx['VLLM_DECODE_BS_BUCKET_STEP'])
    # Special limit for Vision-Instruct models
    if ctx['MODEL'] in [
            'meta-llama/Llama-3.2-11B-Vision-Instruct',
            'meta-llama/Llama-3.2-90B-Vision-Instruct'
    ] and val > 128:
        print(f"{ctx['MODEL']} currently does not support max-num-seqs > 128. "
              "Limiting max-num-seqs to 128")
        val = 128
    if val < 1:
        raise ValueError(
            "Not enough memory for kv cache. Increase TENSOR_PARALLEL_SIZE or "
            "reduce MAX_MODEL_LEN or increase bucket step")
    return val


def calc_VLLM_DECODE_BLOCK_BUCKET_MAX(ctx):
    return max(128,
               math.ceil((ctx['MAX_NUM_SEQS'] * ctx['MAX_MODEL_LEN']) / 128))


def calc_VLLM_PROMPT_SEQ_BUCKET_MAX(ctx):
    return ctx['MAX_MODEL_LEN']


# Map parameter names to calculation functions
PARAM_CALC_FUNCS = {
    "TENSOR_PARALLEL_SIZE": calc_TENSOR_PARALLEL_SIZE,
    "MAX_MODEL_LEN": calc_MAX_MODEL_LEN,
    "PT_HPU_ENABLE_LAZY_COLLECTIVES": calc_PT_HPU_ENABLE_LAZY_COLLECTIVES,
    "MODEL_MEM_FROM_CONFIG": calc_MODEL_MEM_FROM_CONFIG,
    "DEVICE_HPU_MEM": calc_DEVICE_HPU_MEM,
    "TOTAL_GPU_MEM": calc_TOTAL_GPU_MEM,
    "MODEL_MEM_IN_GB": calc_MODEL_MEM_IN_GB,
    "USABLE_MEM": calc_USABLE_MEM,
    "GPU_MEMORY_UTIL_TEMP": calc_GPU_MEMORY_UTIL_TEMP,
    "GPU_MEM_UTILIZATION": calc_GPU_MEM_UTILIZATION,
    "KV_CACHE_PER_SEQ": calc_KV_CACHE_PER_SEQ,
    "EST_MAX_NUM_SEQS": calc_EST_MAX_NUM_SEQS,
    "EST_HPU_BLOCKS": calc_EST_HPU_BLOCKS,
    "DECODE_BS_RAMP_GRAPHS": calc_DECODE_BS_RAMP_GRAPHS,
    "DECODE_BS_STEP_GRAPHS": calc_DECODE_BS_STEP_GRAPHS,
    "DECODE_BLOCK_RAMP_GRAPHS": calc_DECODE_BLOCK_RAMP_GRAPHS,
    "DECODE_BLOCK_STEP_GRAPHS": calc_DECODE_BLOCK_STEP_GRAPHS,
    "NUM_DECODE_GRAPHS": calc_NUM_DECODE_GRAPHS,
    "PROMPT_BS_RAMP_GRAPHS": calc_PROMPT_BS_RAMP_GRAPHS,
    "PROMPT_BS_STEP_GRAPHS": calc_PROMPT_BS_STEP_GRAPHS,
    "PROMPT_SEQ_RAMP_GRAPHS": calc_PROMPT_SEQ_RAMP_GRAPHS,
    "PROMPT_SEQ_STEP_GRAPHS": calc_PROMPT_SEQ_STEP_GRAPHS,
    "EST_NUM_PROMPT_GRAPHS": calc_EST_NUM_PROMPT_GRAPHS,
    "EST_GRAPH_PROMPT_RATIO": calc_EST_GRAPH_PROMPT_RATIO,
    "VLLM_GRAPH_PROMPT_RATIO": calc_VLLM_GRAPH_PROMPT_RATIO,
    "DECODE_GRAPH_TARGET_GB": calc_DECODE_GRAPH_TARGET_GB,
    "EST_GRAPH_RESERVE_MEM": calc_EST_GRAPH_RESERVE_MEM,
    "VLLM_GRAPH_RESERVED_MEM": calc_VLLM_GRAPH_RESERVED_MEM,
    "KV_CACHE_MEM": calc_KV_CACHE_MEM,
    "MAX_NUM_SEQS": calc_MAX_NUM_SEQS,
    "VLLM_DECODE_BLOCK_BUCKET_MAX": calc_VLLM_DECODE_BLOCK_BUCKET_MAX,
    "VLLM_PROMPT_SEQ_BUCKET_MAX": calc_VLLM_PROMPT_SEQ_BUCKET_MAX,
}
