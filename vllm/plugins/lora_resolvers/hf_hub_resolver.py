# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os

from huggingface_hub import HfApi, snapshot_download

import vllm.envs as envs
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolverRegistry
from vllm.plugins.lora_resolvers.filesystem_resolver import FilesystemResolver


class HfHubResolver(FilesystemResolver):
    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        self.adapter_dirs: None | set[str] = None

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        """Resolves potential LoRA requests in a remote repo on HF Hub.
        This is effectively the same behavior as the filesystem resolver, but
        with a snapshot_download on dirs containing an adapter config prior
        to inspecting the cached dir to build a potential LoRA
        request.
        """
        if self.adapter_dirs is None:
            self.adapter_dirs = await self._get_adapter_dirs()
        if lora_name in self.adapter_dirs:
            repo_path = await asyncio.to_thread(
                snapshot_download,
                repo_id=self.repo_name,
                allow_patterns=f"{lora_name}/*",
            )
            lora_path = os.path.join(repo_path, lora_name)
            maybe_lora_request = await self._get_lora_req_from_path(
                lora_name, lora_path, base_model_name
            )
            return maybe_lora_request
        return None

    async def _get_adapter_dirs(self) -> set[str]:
        """Gets the subpaths within a HF repo containing an adapter config."""
        repo_files = await asyncio.to_thread(
            HfApi().list_repo_files, repo_id=self.repo_name
        )
        return {
            os.path.dirname(name)
            for name in repo_files
            if name.endswith("adapter_config.json")
        }


def register_hf_hub_resolver():
    """Register the Hf hub LoRA Resolver with vLLM"""

    hf_repo = envs.VLLM_LORA_RESOLVER_HF_REPO
    if hf_repo:
        hf_hub_resolver = HfHubResolver(hf_repo)
        LoRAResolverRegistry.register_resolver("Hf Hub Resolver", hf_hub_resolver)

    return
