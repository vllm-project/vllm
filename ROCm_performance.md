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

## Fp8 Quantization

To use fp8 quantization, first step is to use Nvidia ammo to quantize your model to fp8 format, following this [instruction](https://github.com/vllm-project/vllm/blob/main/examples/fp8/quantizer/README.md). This will give a safetensor file that contains the quantized weights and the corresponding scaling factors of your model. We will need to put the safetensor file under your model folder, and add file called `serenity_config.json`, which contains a json object with a key: `"quantized_weights": "quantized/osf/rank0.safetensors"`, the value should be the releative path of your safetensor file containing the quantized weights.

Then we can run a model with fp8 quantization using vllm, just add a parameter `quantization="fp8"` when creating the vllm.LLM object.

## Gemm Tunning for Fp8

To get better performance of fp8 quantization, we will need to tune the gemm with the information of all the shapes used in the execution of the model. 

To obtain all the shapes of gemms during the execution of the model, set the env value TUNE_FP8=1 and the run the model as usual. We will get the a file called `/fp8_shapes.csv`.

Next, run gradlib to obtain the best solutions of these shapes:

```
cd gradlib_fp8
python3 -m pip uninstall gradlib
python3 setup.py install
python3 gemm_tunner.py --input_file /fp8_shapes.csv --tuned_file /tuned_fp8_16.csv
cd ../gradlib
python3 -m pip uninstall gradlib
python3 setup.py install
cd ..
```

Now, when running inference with fp8, we are using the tunned gemm for best performance.