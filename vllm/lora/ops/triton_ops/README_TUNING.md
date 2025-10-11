# Multi-LoRA Tuning

**Note**: The lora config folder should be passed in by export VLLM_TUNED_CONFIG_FOLDER=/path/to/configs. Without it, the kernel would use default configs

## Tuning Process
Multi-lora shrink/expand Triton kernel tuning follows a similar methodology from [Triton MoE tuning](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py). An example searching space:

```
    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [32, 64, 128, 256]
    block_k_range = [32, 64, 128, 256]
    num_warps_range = [4, 8]
    num_stage_range = [2, 3, 4, 5]
    num_ctas_range = [1]
    split_k_range = [4, 8, 16, 32, 64]
```
Specifically for multi-lora, `num_slices = [1,2,3]` requires to be tuned sperately for different `MNK` shapes for both shrink and expand kernels.

## Config Files
### File Name

For `shrink`, the config file is named as `{gpu_name}_SHRINK.json`, e.g. `NVIDIA_H200_SHRINK.json`. 

For `expand`, the config fileis named as `{gpu_name}_EXPAND_{add_input}.json`, e.g. `NVIDIA_H200_EXPAND_TRUE.json`.

`gpu_name` can be automatically detected by calling `torch.cuda.get_device_name()` 

### Json Structure
Optimial kernel config files are saved in Json file with a structure as `config_data[max_loras][num_slices][m][k][n]`
