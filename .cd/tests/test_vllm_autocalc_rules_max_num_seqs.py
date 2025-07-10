# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import server.vllm_autocalc_rules as rules


def test_calc_MAX_NUM_SEQS_user_provided():
    ctx = {'MAX_NUM_SEQS': 5}
    assert rules.calc_MAX_NUM_SEQS(ctx) == 5

    ctx = {'MAX_NUM_SEQS': 0}
    assert rules.calc_MAX_NUM_SEQS(ctx) == 1


def test_calc_MAX_NUM_SEQS_fp8():
    ctx = {
        'MAX_NUM_SEQS': None,
        'TENSOR_PARALLEL_SIZE': 2,
        'KV_CACHE_MEM': 64,
        'KV_CACHE_PER_SEQ': 2,
        'DTYPE': 'fp8',
        'VLLM_DECODE_BS_BUCKET_STEP': 8,
        'MODEL': 'test'
    }
    val = (2 * 64 / 2)
    expected = max(1, math.floor(val / 8)) * 8
    assert rules.calc_MAX_NUM_SEQS(ctx) == expected


def test_calc_MAX_NUM_SEQS_non_fp8():
    ctx = {
        'MAX_NUM_SEQS': None,
        'TENSOR_PARALLEL_SIZE': 2,
        'KV_CACHE_MEM': 64,
        'KV_CACHE_PER_SEQ': 2,
        'DTYPE': 'bfloat16',
        'VLLM_DECODE_BS_BUCKET_STEP': 8,
        'MODEL': 'test'
    }
    val = (2 * 64 / 2)
    expected = math.ceil(val / 8) * 8
    assert rules.calc_MAX_NUM_SEQS(ctx) == expected


def test_calc_MAX_NUM_SEQS_vision_instruct_limit():
    ctx = {
        'MAX_NUM_SEQS': None,
        'TENSOR_PARALLEL_SIZE': 2,
        'KV_CACHE_MEM': 2048,
        'KV_CACHE_PER_SEQ': 2,
        'DTYPE': 'bfloat16',
        'VLLM_DECODE_BS_BUCKET_STEP': 8,
        'MODEL': 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    }
    assert rules.calc_MAX_NUM_SEQS(ctx) == 128


def test_calc_MAX_NUM_SEQS_not_enough_memory():
    ctx = {
        'MAX_NUM_SEQS': None,
        'TENSOR_PARALLEL_SIZE': 2,
        'KV_CACHE_MEM': 0,
        'KV_CACHE_PER_SEQ': 2,
        'DTYPE': 'bfloat16',
        'VLLM_DECODE_BS_BUCKET_STEP': 8,
        'MODEL': 'test'
    }
    with pytest.raises(ValueError):
        rules.calc_MAX_NUM_SEQS(ctx)
