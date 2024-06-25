## AMD Quark
### Quantizer Utilities
`quantize_quark.py`: AMD quantization utilities using Quark.

### Prerequisite
Quark Installation:
`pip install --no-cache-dir quark`

### Usage
```
python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
      --output_dir output_dir \
      --quant_scheme w_fp8_a_fp8_o_fp8 \
      --num_calib_data 128 \
      --model_export vllm_adopted_safetensors \
      --no_weight_matrix_merge
```

Output: model structure, quantized model & parameters (with scaling factors) are in JSON and Safetensors.
```
ll output_dir/
total 6837032
drwxr-xr-x 2 root root       4096 Jun 25 18:42 ./
drwxrwxr-x 1 root root       4096 Jun 25 18:42 ../
-rw-r--r-- 1 root root     115917 Jun 25 18:42 llama.json
-rw-r--r-- 1 root root 7000987528 Jun 25 18:42 llama.safetensors
```

## NVIDIA Ammo
### Quantizer Utilities
`quantize.py`: NVIDIA Quantization utilities using AMMO, ported from TensorRT-LLM:
`https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/quantize.py`

### Prerequisite

#### AMMO (AlgorithMic Model Optimization) Installation: nvidia-ammo 0.7.1 or later
`pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo` 

#### AMMO Download (code and docs)
`https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.5.0.tar.gz`
`https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.7.1.tar.gz`

### Usage

#### Run on H100 system for speed if FP8; number of GPUs depends on the model size

#### Example: quantize Llama2-7b model from HF to FP8 with FP8 KV Cache:
`python quantize.py --model_dir ./ll2-7b --dtype float16 --qformat fp8 --kv_cache_dtype fp8 --output_dir ./ll2_7b_fp8 --calib_size 512 --tp_size 1`

Outputs: model structure, quantized model & parameters (with scaling factors) are in JSON and Safetensors (npz is generated only for the reference)
```
# ll ./ll2_7b_fp8/
total 19998244
drwxr-xr-x 2 root root        4096 Feb  7 01:08 ./
drwxrwxr-x 8 1060 1061        4096 Feb  7 01:08 ../
-rw-r--r-- 1 root root      176411 Feb  7 01:08 llama_tp1.json
-rw-r--r-- 1 root root 13477087480 Feb  7 01:09 llama_tp1_rank0.npz
-rw-r--r-- 1 root root  7000893272 Feb  7 01:08 rank0.safetensors
#
```

