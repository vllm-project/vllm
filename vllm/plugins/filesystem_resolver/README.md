# Filesystem Resolver

This is a vLLM general plugin for dynamically discovered LoRA adapters from a local directory.
To use, set `VLLM_PLUGIN_LORA_CACHE_DIR` to a local directory. When vLLM receives a request
for a LoRA adapter `foobar` it doesn't currently recognize, it will look in that local directory
for a subdirectory `foobar` containing a LoRA adapter. If such an adapter exists, it will
load that adapter, and then service the request as normal. That adapter will then be available
for future requests as normal.

Note that `VLLM_ALLOW_RUNTIME_LORA_UPDATING` must be set to true to allow LoRA resolver plugins
to work.
