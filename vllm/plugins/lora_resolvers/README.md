# LoRA Resolver Plugins

This directory contains vLLM general plugins for dynamically discovering and loading LoRA adapters
via the LoRAResolver plugin framework.

Note that `VLLM_ALLOW_RUNTIME_LORA_UPDATING` must be set to true to allow LoRA resolver plugins
to work, and `VLLM_PLUGINS` must be set to include the desired resolver plugins.

The LoRA Resolvers listed below are installed with vLLM by default.

# lora_filesystem_resolver
To use, set `VLLM_LORA_RESOLVER_CACHE_DIR` to a local directory. When vLLM receives a request
for a LoRA adapter `foobar` it doesn't currently recognize, it will look in that local directory
for a subdirectory `foobar` containing a LoRA adapter. If such an adapter exists and matches the
model's `base_model_name_or_path`, it will load that adapter, and then service the request
as normal. That adapter will then be available for future requests as normal.

# hf_hub_resolver
To use, set `VLLM_LORA_RESOLVER_HF_REPO` to a repository ID on Huggingface Hub. When vLLM receives
a request for a LoRA adapter `foobar` it doesn't currently recognize that is a directory in the repo
containing an adapter config, it will download that suibdirectory from the repository, then proceed
in an identical manner to the `lora_filesystem_resolver` using the cached directory.
