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
    "meta-llama/Llama-2-70b-tp4-hf": [([8192, 2560], None), ([2048,
                                                              8192], None),
                                      ([8192, 14336], None),
                                      ([7168, 8192], None)],
    # The shape space is very big when benchmarking a large set of kernels.
    # For example: Let,
    #  - #kernels to benchmark be 1700
    #  - #models to benchmark be 4 (each model has 4 shapes)
    #  - #batch sizes be 6 (16, 32, 64, 128, 256, 512)
    # For 1 kernel, 1 shape and 1 batch-size, H100 takes 1 second (approx.)
    # to run, then the benchmark suite would take,
    # 1700 * (4 * 4) * 6 = 163200 seconds => 46 hrs.
    # Below, we exploit some observation on the benchmark shapes to create a
    # representative set.
    #
    # From previous benchmarking runs, we observe that perf if stratified as,
    # N - small, medium, large and K - small and large. We also observe that
    # in the model shapes, when K is small, we have small, medium and large Ns.
    # when K is large, we only have small Ns.
    #
    # models : ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-3-8b',
    #  'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-70b-tp4-hf']
    # Ks : [2048, 4096, 5120, 7168, 8192, 11008, 13824, 14336]
    # Ns : [2560, 4096, 5120, 6144, 8192, 12288, 14336, 15360,
    #         22016, 27648, 28672]
    "llama-representative-set": [
        ([4096, 4096], None),  # small K, small N
        ([4096, 8192], None),  # small K, medium N
        ([4096, 22016], None),  # small K, large N
        ([14336, 4096], None),  # large K, small N
        ([8192, 14336], None),  # medium K, large N (from llama-2-70b-tp4-hf
    ],
}
