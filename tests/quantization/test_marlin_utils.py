# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    verify_marlin_supports_shape)


def test_verify_marlin_supports_shape_rejects_bad_output_partition():
    # output_size_per_partition must be a multiple of min_thread_n (64).
    with pytest.raises(ValueError) as exc:
        verify_marlin_supports_shape(output_size_per_partition=63,
                                     input_size_per_partition=128,
                                     input_size=128,
                                     group_size=128)
    assert "--quantization gptq" not in str(exc.value)
    assert "tensor_parallel_size" in str(exc.value)


def test_verify_marlin_supports_shape_rejects_bad_input_partition():
    # input_size_per_partition must be a multiple of min_thread_k (128).
    with pytest.raises(ValueError) as exc:
        verify_marlin_supports_shape(output_size_per_partition=64,
                                     input_size_per_partition=127,
                                     input_size=128,
                                     group_size=128)
    assert "--quantization gptq" not in str(exc.value)
    assert "tensor_parallel_size" in str(exc.value)


def test_verify_marlin_supports_shape_accepts_valid_shape():
    verify_marlin_supports_shape(output_size_per_partition=64,
                                 input_size_per_partition=128,
                                 input_size=128,
                                 group_size=128)
