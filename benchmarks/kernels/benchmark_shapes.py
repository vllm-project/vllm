# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

WEIGHT_SHAPES = {
    "ideal": [[4 * 256 * 32, 256 * 32]],
    "mistralai/Mistral-7B-v0.1/TP1": [
        [4096, 6144],
        [4096, 4096],
        [4096, 28672],
        [14336, 4096],
    ],
    "mistralai/Mistral-7B-v0.1/TP2": [
        [4096, 3072],
        [2048, 4096],
        [4096, 14336],
        [7168, 4096],
    ],
    "mistralai/Mistral-7B-v0.1/TP4": [
        [4096, 1536],
        [1024, 4096],
        [4096, 7168],
        [3584, 4096],
    ],
    "meta-llama/Llama-2-7b-hf/TP1": [
        [4096, 12288],
        [4096, 4096],
        [4096, 22016],
        [11008, 4096],
    ],
    "meta-llama/Llama-2-7b-hf/TP2": [
        [4096, 6144],
        [2048, 4096],
        [4096, 11008],
        [5504, 4096],
    ],
    "meta-llama/Llama-2-7b-hf/TP4": [
        [4096, 3072],
        [1024, 4096],
        [4096, 5504],
        [2752, 4096],
    ],
    "meta-llama/Llama-2-13b-hf/TP1": [
        [5120, 15360],
        [5120, 5120],
        [5120, 27648],
        [13824, 5120],
    ],
    "meta-llama/Llama-2-13b-hf/TP2": [
        [5120, 7680],
        [2560, 5120],
        [5120, 13824],
        [6912, 5120],
    ],
    "meta-llama/Llama-2-13b-hf/TP4": [
        [5120, 3840],
        [1280, 5120],
        [5120, 6912],
        [3456, 5120],
    ],
    "meta-llama/Llama-2-70b-hf/TP1": [
        [8192, 10240],
        [8192, 8192],
        [8192, 57344],
        [28672, 8192],
    ],
    "meta-llama/Llama-2-70b-hf/TP2": [
        [8192, 5120],
        [4096, 8192],
        [8192, 28672],
        [14336, 8192],
    ],
    "meta-llama/Llama-2-70b-hf/TP4": [
        [8192, 2560],
        [2048, 8192],
        [8192, 14336],
        [7168, 8192],
    ],
}

WEIGHT_SHAPES_MOE = {
    "nm-testing/Mixtral-8x7B-Instruct-v0.1": [
        [8, 2, 4096, 28672],
        [8, 2, 14336, 4096],
    ],
    "nm-testing/deepseekv2-lite": [
        [64, 6, 2048, 1408],
    ],
    "ibm-granite/granite-3.0-1b-a400m": [
        [32, 8, 1024, 1024],
    ],
    "ibm-granite/granite-3.0-3b-a800m": [
        [40, 8, 1024, 1536],
    ],
}
