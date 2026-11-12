# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune warmup cache."""

import gc
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.artifact_store import (
    STORE_BACKENDS,
    ArtifactStore,
    DiskArtifactStore,
    artifact_key,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_warmup_runner,
    is_draft_model_cacheable,
)
from vllm.model_executor.model_loader.weight_cache.warmup_cache.base import WarmupCache
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    flashinfer_autotune_cache_hash,
    flashinfer_autotune_filename,
    flashinfer_autotune_root,
    use_flashinfer_autotune_v2,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        WeightCacheDaemon,
    )
    from vllm.model_executor.model_loader.weight_cache.seed import (
        SeedTransport,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)
_CONFIG_HASH_RE = re.compile(r"[0-9a-f]{64}")


class FlashInferWarmupCache(WarmupCache):
    """FlashInfer's per-rank autotune table (a single JSON file)."""

    kind = "flashinfer_autotune"

    def __init__(self, store_names: Iterable[str] = ()) -> None:
        self._store_names = tuple(store_names)
        self._store_sources: list[ArtifactStore] | None = None

    @property
    def store_sources(self) -> list[ArtifactStore]:
        # Lazily built: resolving the disk root needs flashinfer, which ROCm
        # daemons never have; construction must stay flashinfer-free.
        if self._store_sources is None:
            self._store_sources = [
                DiskArtifactStore(self.kind, flashinfer_autotune_root())
            ]
            # A misspelled store name or a failed build is a startup error.
            for name in self._store_names:
                self._store_sources.append(STORE_BACKENDS[name]())
        return self._store_sources

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        vllm_config = daemon.vllm_config
        # Mirrors the gate in the engine's kernel warmup.
        if (
            vllm_config.kernel_config.enable_flashinfer_autotune is False
            or not has_flashinfer()
            or not current_platform.has_device_capability(90)
            or is_draft_model_cacheable(vllm_config.speculative_config)
        ):
            return
        from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune

        try:
            current_platform.update_block_size_for_backend(vllm_config)
            runner = build_warmup_runner(
                vllm_config,
                daemon.local_rank,
                is_draft=daemon.is_draft,
                model_config=daemon.model_config,
            )
            assert daemon.model is not None, "warmup ran before load_model"
            runner.load_model(model=daemon.model)
            # Shared stores come before D2D: they are built for read fan-out.
            if not self._adopt_from_stores(runner):
                transport.adopt(self, runner)
            # The runner requires skip_attn when no KV cache exists
            flashinfer_autotune(runner, skip_attn=True)
            logger.info(
                "Weight cache %s daemon rank %d tuned FlashInfer; the tuned "
                "configs are in the on-disk autotune cache",
                daemon.role,
                daemon.global_rank,
            )
            self._publish_autotune_table(runner)
        finally:
            # Free the runner and activations; the daemon's weights are untouched.
            gc.collect()
            torch.accelerator.empty_cache()

    def request_params(self, runner: "GPUModelRunner") -> dict[str, Any]:
        return {
            "config_hash": flashinfer_autotune_cache_hash(runner),
            "dp_rank": runner.vllm_config.parallel_config.data_parallel_rank,
            "rank": get_world_group().rank_in_group,
        }

    def _runner_key(self, runner: "GPUModelRunner") -> tuple[str, str]:
        """The (key, filename) naming this rank's table in every store."""
        params = self.request_params(runner)
        filename = flashinfer_autotune_filename(params["dp_rank"], params["rank"])
        key = artifact_key(self.kind, params["config_hash"], filename)
        return key, filename

    def resolve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        config_hash = request.get("config_hash")
        if not isinstance(config_hash, str) or not _CONFIG_HASH_RE.fullmatch(
            config_hash
        ):
            raise ValueError("invalid config_hash")

        # Garbage ranks name a missing key; the hex check keeps us in the root.
        filename = flashinfer_autotune_filename(request["dp_rank"], request["rank"])
        key = artifact_key(self.kind, config_hash, filename)
        for store in self.store_sources:
            if (data := store.get(key)) is not None:
                return {filename: json.loads(data)}
        return None

    def _adopt_from_stores(self, runner: "GPUModelRunner") -> bool:
        """Pull this rank's table from the shared stores; True skips D2D."""
        shared = self.store_sources[1:]
        if not shared or use_flashinfer_autotune_v2(runner):
            return False
        key, filename = self._runner_key(runner)
        if self.store_sources[0].get(key) is not None:
            return False
        for store in shared:
            if (data := store.get(key)) is not None:
                break
        else:
            return False
        try:
            payload = self.validate_payload({filename: json.loads(data)})
        except ValueError as e:
            logger.warning("Rejecting the shared autotune table %s: %s", key, e)
            return False
        self.install(runner, payload)
        logger.info(
            "Adopted the FlashInfer autotune table from a shared store: %s", key
        )
        return True

    def _publish_autotune_table(self, runner: "GPUModelRunner") -> None:
        """Publish this rank's tuned table from the disk store to every store."""
        if len(self.store_sources) == 1:
            return
        key, _ = self._runner_key(runner)
        if (data := self.store_sources[0].get(key)) is None:
            return
        for store in self.store_sources:
            store.put(key, data)
        logger.info("Published the FlashInfer autotune table: %s", key)

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError("expected exactly one autotune table")
        ((filename, table),) = payload.items()
        if not isinstance(table, dict):
            raise ValueError("autotune table must be a JSON object")
        if not filename.startswith("autotune_configs_"):
            raise ValueError(f"unexpected file {filename!r}")
        # Reject tables flashinfer would refuse to load; a mismatch never heals.
        saved_meta = table.get("_metadata")
        if saved_meta is None:
            # Pre-metadata flashinfer file; load_configs accepts these.
            return payload
        try:
            from flashinfer.autotuner.autotuner import (
                _classify_metadata_mismatches,
                _collect_metadata,
            )
        except ImportError:
            logger.warning(
                "flashinfer metadata classifier unavailable; "
                "accepting the remote autotune table unchecked"
            )
            return payload
        hard, soft = _classify_metadata_mismatches(saved_meta, _collect_metadata())
        mismatches = {**hard, **soft}
        if mismatches:
            details = ", ".join(
                f"{key}: remote={old} vs local={new}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(f"autotune table environment mismatch: {details}")
        return payload

    def install(self, runner: "GPUModelRunner", payload: dict[str, Any]) -> None:
        (data,) = payload.values()
        key, _ = self._runner_key(runner)
        self.store_sources[0].put(key, json.dumps(data).encode())
        logger.info("Installed remote FlashInfer autotune table: %s", key)
