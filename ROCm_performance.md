# Overview of the optional performance features uinque to https://github.com/ROCm/vllm
## Multi-GPU torchrun
On ROCm the default multi GPU executor is `torchrun` as opposed to `ray` on NVIDIA  
This can be overridden by the `--worker-use-ray` flag to vllm or its benchmarks  
To utilize torchran parallelism, the run command should be modified from  
`python <command>`  
to  
`torchrun --standalone --nnodes=1 --nproc-per-node=<world-size> <command>`
## Triton attention
The default attention function on ROCm is using triton attention kernel. To fallback to the https://github.com/ROCm/flash-attention implementation set up the following environment symbol:  
`VLLM_USE_FLASH_ATTN_TRITON=False`
## Tunable ops
Pytorch tunable ops are supported.  
Define the following environment symbol: `PYTORCH_TUNABLEOP_ENABLED=1` in order to enable both the runtime tuning and the subsequent use of tuned results. To only use the tuned results without tuning any newly encountered shapes, also define `PYTORCH_TUNABLEOP_TUNING=1`

## Custom PagedAttention

On ROCm, to have better performance, a custom paged attention is available by switching on the env variable: `VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1`.
Currently, this env variable is enabled by default. To fallback to PagedAttention v2 kernel assign the env variable to 0.
The custom PagedAttention kernel is enabled for dtype: fp16, block-size=16, head-size=128, and max context length <= 16k, with GQA ratio (num_heads//num_kv_heads) between 1 to 16. On all the other cases, we fallback to PagedAttention v2 kernel.
