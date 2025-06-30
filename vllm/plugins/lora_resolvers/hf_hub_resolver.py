# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

from vllm.lora.request import LoRARequest
from vllm.plugins.lora_resolvers.filesystem_resolver import FilesystemResolver


class HfHubResolver(FilesystemResolver):
    """Similar in usage to the filesystem_resolver, but allows for the use
    of HF hub. This plugin assumes the contents of the HF repo are static."""

    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        repo_files = HfApi().list_repo_files(repo_id=repo_name)
        # valid potential adapters are file path to directories
        # containing the adapter configs
        self.adapter_dirs = [
            name.split("/")[0] for name in repo_files
            if name.endswith("adapter_config.json")
        ]

    async def resolve_lora(self, base_model_name: str,
                           lora_name: str) -> Optional[LoRARequest]:
        if lora_name in self.adapter_dirs:
            repo_path = snapshot_download(repo_id=self.repo_name,
                                          allow_patterns=f"{lora_name}/*")
            lora_path = os.path.join(repo_path, lora_name)

            if os.path.exists(lora_path):
                adapter_config_path = os.path.join(lora_path,
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
