Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

## Use fastsafetensors in vLLM

On CUDA and ROCm builds, `fastsafetensors` is installed by default and
`--load-format auto` will prefer it automatically for safetensors checkpoints.

To force this loader explicitly, use the `--load-format fastsafetensors`
command-line argument.

For example:

```bash
vllm serve Qwen/Qwen2.5-0.5B --load-format fastsafetensors
```
