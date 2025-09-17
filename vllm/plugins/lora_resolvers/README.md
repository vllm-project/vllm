# LoRA Resolver Plugins

This directory contains vLLM general plugins for dynamically discovering and loading LoRA adapters
via the LoRAResolver plugin framework.

Note that `VLLM_ALLOW_RUNTIME_LORA_UPDATING` must be set to true to allow LoRA resolver plugins
to work, and `VLLM_PLUGINS` must be set to include the desired resolver plugins.

## lora_filesystem_resolver

This LoRA Resolver is installed with vLLM by default.
To use, set `VLLM_PLUGIN_LORA_CACHE_DIR` to a local directory. When vLLM receives a request
for a LoRA adapter `foobar` it doesn't currently recognize, it will look in that local directory
for a subdirectory `foobar` containing a LoRA adapter. If such an adapter exists, it will
load that adapter, and then service the request as normal. That adapter will then be available
for future requests as normal.
