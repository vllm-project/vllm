# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    OffloadKey,
    get_offload_block_hash,
    get_offload_group_idx,
)

_BASE_PATH_HASH_LEN = 12
_CONFIG_FILENAME = "config.json"


class FileMapper:
    """
    FileMapper maps KV blocks (given by their hash) to file names.
    """

    def __init__(
        self,
        root_dir: str,
        model_name: str,
        hash_block_size: int,
        gpu_blocks_per_file: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
        rank: int,
        dtype: str,
        kv_cache_groups: list[dict] | None = None,
        inference_engine: str = "vllm",
        parallel_agnostic: bool = False,
    ):
        """
        Initialize the file mapper. Each worker constructs its own, but
        `config.json` is shared across workers since rank lives outside the hash.
        When `parallel_agnostic=True`, tp/pp/pcp/dcp are forced to 1 and rank
        to 0 so multiple parallelism layouts collapse into the same folder.
        """
        if parallel_agnostic:
            tp_size = pp_size = pcp_size = dcp_size = 1
            rank = 0
        self.rank: int = rank
        self.fields: dict = {
            "model_name": model_name,
            "hash_block_size": hash_block_size,
            "gpu_blocks_per_file": gpu_blocks_per_file,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "pcp_size": pcp_size,
            "dcp_size": dcp_size,
            "dtype": str(dtype),
            "kv_cache_groups": kv_cache_groups or [],
            "inference_engine": inference_engine,
        }
        self.base_path: str = self._compute_base_path(root_dir, self.fields)

    @classmethod
    def from_vllm_config(
        cls,
        root_dir: str,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        gpu_blocks_per_file: int,
        parallel_agnostic: bool = False,
    ) -> "FileMapper":
        """Build a FileMapper from a vllm VllmConfig + KVCacheConfig."""
        parallel_config = vllm_config.parallel_config
        dtype = str(vllm_config.cache_config.cache_dtype).replace("torch.", "")
        kv_cache_groups = [
            {
                "block_size": group.kv_cache_spec.block_size,
                "layer_names": list(group.layer_names),
            }
            for group in kv_cache_config.kv_cache_groups
        ]
        return cls(
            root_dir=root_dir,
            model_name=vllm_config.model_config.model,
            hash_block_size=vllm_config.cache_config.block_size,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            pcp_size=parallel_config.prefill_context_parallel_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            rank=parallel_config.rank,
            dtype=dtype,
            kv_cache_groups=kv_cache_groups,
            parallel_agnostic=parallel_agnostic,
        )

    def get_file_name(self, key: OffloadKey) -> str:
        """Map an OffloadKey to <base>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin."""
        hash_hex = get_offload_block_hash(key).hex()
        group_idx = get_offload_group_idx(key)
        subfolder1, subfolder2 = hash_hex[:3], hash_hex[3:5]
        return (
            f"{self.base_path}_r{self.rank}"
            f"/{subfolder1}/{subfolder2}_g{group_idx}/{hash_hex}.bin"
        )

    def get_run_config(self) -> dict:
        return dict(self.fields)
    
    def get_config_file_path(self) -> str:
        return f"{self.base_path}/{_CONFIG_FILENAME}"

    @staticmethod
    def _compute_base_path(root_dir: str, fields: dict) -> str:
        """
        Layout: <root_dir>/<safe_model_name>_<sha256-prefix>/.
        safe_model_name replaces '/' with '_' so HuggingFace IDs don't nest.
        """
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :_BASE_PATH_HASH_LEN
        ]
        safe_model_name = fields["model_name"].replace("/", "_")
        return f"{root_dir}/{safe_model_name}_{digest}"
