Loading Model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

To enable this feature, use the `--load-format fastsafetensors` command-line argument
