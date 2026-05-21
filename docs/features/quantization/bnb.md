# BitsAndBytes

BitsAndBytes support is provided by the out-of-tree
[`vllm-bnb-plugin`](https://github.com/vllm-project/vllm-bnb-plugin).

Install the plugin first:

```bash
uv pip install vllm-bnb-plugin
```

The plugin registers the `bitsandbytes` quantization method and
`bitsandbytes` load format through vLLM's general plugin system, so existing
usage stays the same after installation.

It supports both in-flight 4-bit quantization and pre-quantized 4-bit / 8-bit
checkpoints. Refer to the plugin README for the current installation matrix and
examples.
