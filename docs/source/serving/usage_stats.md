# Usage Stats Collection

vLLM collects usage data by default. This data is used to help engineers working on the project to better understand which hardware and model configuration is widely used, so we can prioritize our attention to the workload that matters. The data is collected is anonymous, transparent, and does not contain any sensitive information. The collected data is also going to be publically released so that the community can benefit from the insights.

## What data is collected?

You can see the up to date list of data collected by vLLM in the [usage_lib.py](https://github.com/vllm-project/vllm/blob/main/vllm/usage/usage_lib.py).

Here is an example as of v0.4.0:

```json
{
  "uuid": "fbe880e9-084d-4cab-a395-8984c50f1109",
  "provider": "GCP",
  "num_cpu": 24,
  "cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
  "cpu_family_model_stepping": "6,85,7",
  "total_memory": 101261135872,
  "architecture": "x86_64",
  "platform": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
  "gpu_count": 2,
  "gpu_type": "NVIDIA L4",
  "gpu_memory_per_device": 23580639232,
  "model_architecture": "OPTForCausalLM",
  "vllm_version": "0.3.2+cu123",
  "context": "LLM_CLASS",
  "log_time": 1711663373492490000,
  "source": "production",
  "dtype": "torch.float16",
  "tensor_parallel_size": 1,
  "block_size": 16,
  "gpu_memory_utilization": 0.9,
  "quantization": null,
  "kv_cache_dtype": "auto",
  "enable_lora": false,
  "enable_prefix_caching": false,
  "enforce_eager": false,
  "disable_custom_all_reduce": true
}
```

You can preview the data being collected by running the following command:

```bash
tail ~/.config/vllm/usage_stats.json
```

## Opt-out of Usage Stats Collection

You can opt-out the collection through either the existence of environment variable (`VLLM_NO_USAGE_STATS` or `DO_NOT_TRACK`)
or the existence of the file `~/.config/vllm/do_not_track`.

```bash
# any of the following way can disable the usage stats collection
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track
```
