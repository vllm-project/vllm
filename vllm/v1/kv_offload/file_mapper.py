# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    canonical_format_id,
)
from vllm.v1.kv_offload.base import (
    OffloadingSpec,
    OffloadKey,
    get_offload_block_hash,
    get_offload_group_idx,
)

_BASE_PATH_HASH_LEN = 12
_CONFIG_FILENAME = "config.json"


class FileMapper:
    """FileMapper maps KV blocks (given by their hash) to file names."""

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
        replicated_layout: bool = False,
        canonical_format: str | None = None,
    ):
        """Initialize the file mapper. Each worker constructs its own, but
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
        if not parallel_agnostic:
            self.fields["parallel_agnostic"] = False
        # Only written when True so existing deployments' hashed fields are
        # unchanged (False is the historical default and must not appear).
        if replicated_layout:
            self.fields["replicated_layout"] = True
        # The canonical byte format is not interchangeable with the direct
        # layout (or with other canonical format versions/families), so its
        # identity participates in the storage namespace.
        if canonical_format is not None:
            self.fields["canonical_format"] = canonical_format
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
        canonical_format = None
        if config.canonical_layout:
            assert config.kv_cache_layout is not None
            canonical_format = canonical_format_id(config.kv_cache_layout)
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
            parallel_agnostic=(
                parallel_agnostic
                and (parallel.is_parallelism_agnostic or config.replicated_layout)
            ),
            replicated_layout=(parallel_agnostic and config.replicated_layout),
            canonical_format=canonical_format,
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
        """Layout: <root_dir>/<safe_model_name>_<sha256-prefix>/.
        safe_model_name replaces '/' with '_' so HuggingFace IDs don't nest.
        """
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :_BASE_PATH_HASH_LEN
        ]
        safe_model_name = fields["model_name"].replace("/", "_")
        return f"{root_dir}/{safe_model_name}_{digest}"

    @property
    def num_shards(self) -> int:
        """Number of storage shards (roots) mapped by this FileMapper."""
        return 1

    def get_shard_index(self, key: OffloadKey) -> int:
        """Return the shard index for the given OffloadKey."""
        return 0

    def get_config_file_paths(self) -> list[str]:
        """Return config file paths for all storage roots."""
        return [self.get_config_file_path()]

    def group_by_shard(
        self, keys: list[OffloadKey], chunk_ids: list[int], chunk_size: int
    ) -> list[tuple[list[str], list[int], list[int]]]:
        """Group keys and chunk_ids by storage shard.

        Returns a list of tuples: (paths, offsets, original_indices)
        for each non-empty shard.
        """
        paths = [self.get_file_name(k) for k in keys]
        offsets = [int(cid) * chunk_size for cid in chunk_ids]
        indices = list(range(len(keys)))
        return [(paths, offsets, indices)]


class ShardedFileMapper(FileMapper):
    """ShardedFileMapper shards KV blocks across multiple storage roots."""

    def __init__(
        self,
        root_dirs: list[str],
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
        replicated_layout: bool = False,
        canonical_format: str | None = None,
        path_sharding: str = "by_block_hash",
    ):
        if not root_dirs or any(not path.strip() for path in root_dirs):
            raise ValueError("root_dirs must contain non-empty filesystem paths")
        self.root_dirs = [path.strip() for path in root_dirs]
        normalized_roots = [os.path.normpath(p) for p in self.root_dirs]
        if len(set(normalized_roots)) != len(normalized_roots):
            raise ValueError("root_dirs paths must be distinct")

        if path_sharding != "by_block_hash":
            raise ValueError(
                f"path_sharding must be 'by_block_hash', got {path_sharding!r}"
            )
        self.path_sharding = path_sharding

        self.mappers = [
            FileMapper(
                root_dir=root,
                model_name=model_name,
                tokens_per_hash=tokens_per_hash,
                blocks_per_file=blocks_per_file,
                tp_size=tp_size,
                pp_size=pp_size,
                pcp_size=pcp_size,
                dcp_size=dcp_size,
                rank=rank,
                dtype=dtype,
                kv_cache_groups=kv_cache_groups,
                inference_engine=inference_engine,
                parallel_agnostic=parallel_agnostic,
                replicated_layout=replicated_layout,
                canonical_format=canonical_format,
            )
            for root in self.root_dirs
        ]

        super().__init__(
            root_dir=self.root_dirs[0],
            model_name=model_name,
            tokens_per_hash=tokens_per_hash,
            blocks_per_file=blocks_per_file,
            tp_size=tp_size,
            pp_size=pp_size,
            pcp_size=pcp_size,
            dcp_size=dcp_size,
            rank=rank,
            dtype=dtype,
            kv_cache_groups=kv_cache_groups,
            inference_engine=inference_engine,
            parallel_agnostic=parallel_agnostic,
            replicated_layout=replicated_layout,
            canonical_format=canonical_format,
        )

    @classmethod
    def from_offloading_spec(
        cls,
        root_dir: str | list[str],
        offloading_spec: OffloadingSpec,
        blocks_per_file: int = 1,
        parallel_agnostic: bool = False,
        path_sharding: str = "by_block_hash",
    ) -> "ShardedFileMapper":
        """Build a ShardedFileMapper from an OffloadingSpec."""
        if isinstance(root_dir, str):
            root_dirs = [p.strip() for p in root_dir.split(",")]
        else:
            root_dirs = list(root_dir)

        config = offloading_spec.config
        kv_cache_groups = [
            {
                "tokens_per_block": group.tokens_per_block,
                "layer_names": list(group.layer_names),
            }
            for group in config.groups
        ]
        parallel = config.parallel
        canonical_format = None
        if config.canonical_layout:
            assert config.kv_cache_layout is not None
            canonical_format = canonical_format_id(config.kv_cache_layout)
        return cls(
            root_dirs=root_dirs,
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
            parallel_agnostic=(
                parallel_agnostic
                and (parallel.is_parallelism_agnostic or config.replicated_layout)
            ),
            replicated_layout=(parallel_agnostic and config.replicated_layout),
            canonical_format=canonical_format,
            path_sharding=path_sharding,
        )

    @property
    def num_shards(self) -> int:
        return len(self.mappers)

    def get_shard_index(self, key: OffloadKey) -> int:
        block_hash = get_offload_block_hash(key)
        return int.from_bytes(block_hash, byteorder="big") % len(self.mappers)

    def get_file_name(self, key: OffloadKey) -> str:
        shard_idx = self.get_shard_index(key)
        return self.mappers[shard_idx].get_file_name(key)

    def get_config_file_paths(self) -> list[str]:
        return [m.get_config_file_path() for m in self.mappers]

    def group_by_shard(
        self, keys: list[OffloadKey], chunk_ids: list[int], chunk_size: int
    ) -> list[tuple[list[str], list[int], list[int]]]:
        groups: list[tuple[list[str], list[int], list[int]]] = [
            ([], [], []) for _ in range(self.num_shards)
        ]
        for idx, (key, cid) in enumerate(zip(keys, chunk_ids)):
            shard_idx = self.get_shard_index(key)
            paths, offsets, indices = groups[shard_idx]
            paths.append(self.mappers[shard_idx].get_file_name(key))
            offsets.append(int(cid) * chunk_size)
            indices.append(idx)
        return [g for g in groups if g[0]]
