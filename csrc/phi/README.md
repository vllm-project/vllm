# PHI Kernel Extensions
## MOE Kernels
To make phimoe be served in one A100(80G) gpu, we proposed this fp8 grouped gemm kernel to make phimoe been served in low latency and high accuracy.

1. *phi/moe/tensorrt_llm_moe_grouped_gemm*, is adopted from tensorrt-llm project. With modification on the grouped gemm kernels, we currently support: Activation as float16/bfloat16, and Weight as fp8 - which is stored as int8 in this kernel    .
2. *phi/moe/tensorrt_llm_moe_grouped_gemm/utils/moe_align_block_size_kernels.cu*, is adopted from vllm moe kernel with modification on interface to get result for #1 grouped gemm kernels.

Phi Ampere FP8 quant and kernel can also leveraged by other MoE models.