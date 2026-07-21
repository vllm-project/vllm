# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

from vllm.v1.kv_offload.base import (
    OffloadingSpec,
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
        tokens_per_hash: int,
        blocks_per_file: int,
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
            "tokens_per_hash": tokens_per_hash,
            "blocks_per_file": blocks_per_file,
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
    def from_offloading_spec(
        cls,
        root_dir: str,
        offloading_spec: OffloadingSpec,
        blocks_per_file: int = 1,
        parallel_agnostic: bool = False,
    ) -> "FileMapper":
        """Build a FileMapper from an OffloadingSpec."""
        config = offloading_spec.config
        kv_cache_groups = [
            {
                "tokens_per_block": group.tokens_per_block,
                "layer_names": list(group.layer_names),
            }
            for group in config.groups
        ]
        parallel = config.parallel
        return cls(
            root_dir=root_dir,
            model_name=config.model.name,
            tokens_per_hash=config.cache.tokens_per_hash,
            blocks_per_file=blocks_per_file,
            tp_size=parallel.tp_size,
            pp_size=parallel.pp_size,
            pcp_size=parallel.pcp_size,
            dcp_size=parallel.dcp_size,
            rank=parallel.rank,
            dtype=config.model.dtype,
            kv_cache_groups=kv_cache_groups,
            parallel_agnostic=(parallel_agnostic and parallel.is_parallelism_agnostic),
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
