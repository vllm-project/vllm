# SPDX-License-Identifier: Apache-2.0
import os

from vllm.lora.resolver import LoRAResolverRegistry
#from vllm.plugins.filesystem_resolver import FilesystemResolver


def register():
    """Register the filesytem LoRA Resolver with vLLM"""

    lora_cache_dir = os.environ["VLLM_PLUGIN_LORA_CACHE_DIR"]
    if lora_cache_dir:
        if not os.path.exists(lora_cache_dir) or not os.path.isdir(
                lora_cache_dir):
            raise ValueError(
                "VLLM_PLUGIN_LORA_CACHE_DIR must be set to a valid directory for Filesystem Resolver plugin to function"
            )
        fs_resolver = FilesystemResolver(lora_cache_dir)
        LoRAResolverRegistry.register_resolver("Filesystem Resolver",
                                                fs_resolver)

    return
