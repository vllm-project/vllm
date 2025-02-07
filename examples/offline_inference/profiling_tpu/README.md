# vLLM TPU Profiling

This script is used to profile the TPU performance of vLLM for specific prefill or decode token shapes.

Note: an actual running server is a mix of both prefill of many shapes and decode of many shapes.

We assume you are on a TPU already (this was tested on TPU v6e) and have installed vLLM according to the [installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/ai_accelerator/index.html).

> In all examples below, we run several warmups before (so `--enforce-eager` is okay)

## Profile Examples

### Generate Prefill Trace

This example runs Qwen/Qwen2.5-7B-Instruct with a single request of 1024 input tokens. This is set up in attempt to profile just the prefill time and operations.

```bash
export XLA_HLO_DEBUG=1
export MODEL=Qwen/Qwen2.5-7B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=3000
export VLLM_TPU_PROFILE_DELAY_MS=0

python3 profiling.py \
    --model $MODEL \
    --input-len 1024 --output-len 1 \
    --batch-size 1 --enforce-eager \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \
    --profile-result-dir profiles
```


### Generate Decode Trace

This example runs Llama 3.1 70B with a batch of 32 requests where each has 1 input token and 128 output tokens. This is set up in attempt to profile just the 32 decodes running in parallel by having an extremely small prefill of 1 token and setting `VLLM_TPU_PROFILE_DELAY_MS=1000` to skip the first second of inference (hopefully prefill).

```bash
export XLA_HLO_DEBUG=1
export MODEL=meta-llama/Llama-3.1-70B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=2000
export VLLM_TPU_PROFILE_DELAY_MS=1000

rm -rf ~/.cache/vllm/xla_cache
python3 profiling.py \
    --model $MODEL \
    --input-len 1 \
    --output-len 128 \
    --batch-size 32 \
    --enforce-eager \
    --profile-result-dir profiles \
    --max-model-len 2048 --tensor-parallel-size 8
```


## Visualizing the profiles

Once you have collected your profiles with this script, you can visualize them using [TensorBoard](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm).

Here are most likely the dependencies you need to install:
```bash
pip install tensorflow-cpu tensorboard-plugin-profile etils importlib_resources
```

Then you just need to point TensorBoard to the directory where you saved the profiles and visit `http://localhost:6006/` in your browser:
```bash
tensorboard --logdir profiles/ --port 6006
```