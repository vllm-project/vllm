# Overview of the optional performance features uinque to https://github.com/ROCm/vllm

## Triton attention
The default attention function on ROCm is using triton attention kernel. To fallback to the https://github.com/ROCm/flash-attention implementation set up the following environment symbol:  
`VLLM_USE_TRITON_FLASH_ATTN=0`

## Tunable ops
Pytorch tunable ops are supported.  
Define the following environment symbol: `PYTORCH_TUNABLEOP_ENABLED=1` in order to enable both the runtime tuning and the subsequent use of tuned results. To only use the tuned results without tuning any newly encountered shapes, set `PYTORCH_TUNABLEOP_TUNING=0`

## Custom PagedAttention

On ROCm, to have better performance, a custom paged attention is available by switching on the env variable: `VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1`.
Currently, this env variable is enabled by default. To fallback to PagedAttention v2 kernel assign the env variable to 0.
The custom PagedAttention kernel is enabled for dtype: bf16, fp16, block-size=16, head-size=128, and max context length <= 16k, with GQA ratio (num_heads//num_kv_heads) between 1 to 16. On all the other cases, we fallback to PagedAttention v2 kernel.

## NCCL Performance environment variable

For MI300x, setting environment variable NCCL_MIN_NCHANNELS=112 is expected to improve performance.

