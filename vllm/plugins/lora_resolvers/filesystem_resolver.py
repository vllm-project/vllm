# SPDX-License-Identifier: Apache-2.0
import json
import os
from typing import Optional

import vllm.envs as envs
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry


class FilesystemResolver(LoRAResolver):

    def __init__(self, lora_cache_dir: str):
        self.lora_cache_dir = lora_cache_dir

    async def resolve_lora(self, base_model_name: str,
                           lora_name: str) -> Optional[LoRARequest]:
        lora_path = os.path.join(self.lora_cache_dir, lora_name)
        if os.path.exists(lora_path):
            adapter_config_path = os.path.join(self.lora_cache_dir, lora_name,
                                               "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path) as file:
                    adapter_config = json.load(file)
                if adapter_config["peft_type"] == "LORA" and adapter_config[
                        "base_model_name_or_path"] == base_model_name:
                    lora_request = LoRARequest(lora_name=lora_name,
                                               lora_int_id=abs(
                                                   hash(lora_name)),
                                               lora_path=lora_path)
                    return lora_request
        return None


def register_filesystem_resolver():
    """Register the filesystem LoRA Resolver with vLLM"""

    lora_cache_dir = envs.VLLM_LORA_RESOLVER_CACHE_DIR
    if lora_cache_dir:
        if not os.path.exists(lora_cache_dir) or not os.path.isdir(
                lora_cache_dir):
            raise ValueError(
                "VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory \
                for Filesystem Resolver plugin to function")
        fs_resolver = FilesystemResolver(lora_cache_dir)
        LoRAResolverRegistry.register_resolver("Filesystem Resolver",
                                               fs_resolver)

    return
