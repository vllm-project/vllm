# SPDX-License-Identifier: Apache-2.0

# Weight Shapes are in the format
# ([K, N], TP_SPLIT_DIM)
# Example:
#  A shape of ([14336, 4096], 0) indicates the following GEMM shape,
#   - TP1 : K = 14336, N = 4096
#   - TP2 : K = 7168, N = 4096
#  A shape of ([4096, 6144], 1) indicates the following GEMM shape,
#   - TP1 : K = 4096, N = 6144
#   - TP4 : K = 4096, N = 1536

# TP1 shapes
WEIGHT_SHAPES = {
    "mistralai/Mistral-7B-v0.1": [
        ([4096, 6144], 1),
        ([4096, 4096], 0),
        ([4096, 28672], 1),
        ([14336, 4096], 0),
    ],
    "meta-llama/Llama-2-7b-hf": [
        ([4096, 12288], 1),
        ([4096, 4096], 0),
        ([4096, 22016], 1),
        ([11008, 4096], 0),
    ],
    "meta-llama/Llama-3-8b": [
        ([4096, 6144], 1),
        ([4096, 4096], 0),
        ([4096, 28672], 1),
        ([14336, 4096], 0),
    ],
    "meta-llama/Llama-2-13b-hf": [
        ([5120, 15360], 1),
        ([5120, 5120], 0),
        ([5120, 27648], 1),
        ([13824, 5120], 0),
    ],
    "meta-llama/Llama-2-70b-hf": [
        ([8192, 10240], 1),
        ([8192, 8192], 0),
        ([8192, 57344], 1),
        ([28672, 8192], 0),
    ],
}
