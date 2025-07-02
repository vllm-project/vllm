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
    def __init__(self, repo_list: str):
        self.repo_list = repo_list
        self.adapter_dirs: dict[str, set[str]] = {}

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        """Resolves potential LoRA requests in a remote repo on HF Hub.
        This is effectively the same behavior as the filesystem resolver, but
        with a snapshot_download on dirs containing an adapter config prior
        to inspecting the cached dir to build a potential LoRA
        request.
        """
        # If a LoRA name begins with the repository name, it's disambiguated
        maybe_repo = await self._resolve_repo(lora_name)
        maybe_subpath = await self._resolve_repo_subpath(lora_name, maybe_repo)

        if maybe_repo is None or maybe_subpath is None:
            return None

        # Get the potential valid adapter subpaths in the HF repo if we haven't
        if maybe_repo is not None and maybe_repo not in self.adapter_dirs:
            self.adapter_dirs[maybe_repo] = await self._get_adapter_dirs(maybe_repo)

        repo_path = await asyncio.to_thread(
            snapshot_download,
            repo_id=maybe_repo,
            allow_patterns=f"{maybe_subpath}/*" if maybe_subpath != "." else "*",
        )

        lora_path = os.path.join(repo_path, maybe_subpath)
        maybe_lora_request = await self._get_lora_req_from_path(
            lora_name, lora_path, base_model_name
        )
        return maybe_lora_request

    async def _resolve_repo(self, lora_name: str) -> str | None:
        for potential_repo in self.repo_list:
            if lora_name.startswith(potential_repo) and (
                len(lora_name) == len(potential_repo)
                or lora_name[len(potential_repo)] == "/"
            ):
                return potential_repo
        return None

    async def _resolve_repo_subpath(
        self, lora_name: str, maybe_repo: str | None
    ) -> str | None:
        if maybe_repo is None:
            return None
        repo_len = len(maybe_repo)
        if (
            lora_name == maybe_repo
            or len(lora_name) == repo_len + 1
            and lora_name[-1] == "/"
        ):
            # Resolves to the root of the directory
            return "."
        return lora_name[repo_len + 1 :]

    async def _get_adapter_dirs(self, repo_name: str) -> set[str]:
        """Gets the subpaths within a HF repo containing an adapter config."""
        repo_files = await asyncio.to_thread(HfApi().list_repo_files, repo_id=repo_name)
        adapter_dirs = {
            os.path.dirname(name)
            for name in repo_files
            if name.endswith("adapter_config.json")
        }
        if "adapter_config.json" in repo_files:
            adapter_dirs.add(".")
        return adapter_dirs


def register_hf_hub_resolver():
    """Register the Hf hub LoRA Resolver with vLLM"""

    hf_repo_list = envs.VLLM_LORA_RESOLVER_HF_REPO_LIST
    if hf_repo_list:
        hf_hub_resolver = HfHubResolver(hf_repo_list.split(","))
        LoRAResolverRegistry.register_resolver("Hf Hub Resolver", hf_hub_resolver)

    return
