Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

## Installation

`fastsafetensors` is an optional dependency and is not installed with the default `vllm` package.

Install it with the vLLM extra:

```bash
pip install "vllm[fastsafetensors]"
```

If you are installing vLLM from source in editable mode, include the extra during installation:

```bash
pip install -e ".[fastsafetensors]"
```

## Use fastsafetensors in vLLM

To enable this feature, use the `--load-format fastsafetensors` command-line argument.

For example:

```bash
vllm serve Qwen/Qwen2.5-0.5B --load-format fastsafetensors
```
