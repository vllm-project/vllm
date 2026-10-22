# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon warmup caches, seeded daemon-to-daemon over the shared transport.

Each warmup-cache object (currently only :class:`FlashInferWarmupCache`)
drives its own warmup pass on the daemon; the seeding flow itself
(:class:`seed.SeedTransport`) is shared with future kinds such as weight
shards. A target daemon installs
its rank-matched cache before tuning, so both the daemon's own warmup and
the later engine start skip re-tuning. The disk layout is the index: a
request names the cache by the same key the layout uses (kind, config hash,
dp rank, world rank) and the source resolves it to files under the cache
root. There is no separate artifact store or registry.

The FlashInfer payload is a ``{filename: parsed JSON}`` map rather than a
single document because upcoming kinds (Triton tuned configs) consist of
several JSON files per cache.

TODO: install valid files individually instead of falling back to local
warmup when any file of a multi-file kind fails validation.
"""

import gc
import json
from typing import TYPE_CHECKING, Any, Protocol

import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.seed import Seedable
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_warmup_runner,
    is_draft_model_cacheable,
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
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_CONFIG_HASH_RE = re.compile(r"[0-9a-f]{64}")


class WarmupCache(Seedable, Protocol):
    """One daemon warmup cache: its warmup pass and its cache's semantics.

    The warmup layer; the seeding exchange with peer daemons is the shared
    D2D layer in :mod:`seed` and :mod:`transport`.
    """

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        """Run this rank's warmup pass against the cached model."""
        ...


class FlashInferWarmupCache(WarmupCache):
    """FlashInfer's per-rank autotune table (a single JSON file)."""

    kind = "flashinfer_autotune"

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        """Run the FlashInfer autotune pass against the cached model."""
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
            transport.adopt(self, runner)
            # The V2 runner requires skip_attn when no KV cache exists
            flashinfer_autotune(runner, skip_attn=vllm_config.use_v2_model_runner)
            logger.info(
                "Weight cache %s daemon rank %d tuned FlashInfer; the tuned "
                "configs are in the on-disk autotune cache",
                daemon.role,
                daemon.global_rank,
            )
        finally:
            # Free the runner and the dummy-run activations; the daemon's
            # weights are untouched.
            gc.collect()
            torch.accelerator.empty_cache()

    def request_params(self, runner: "GPUModelRunner") -> dict[str, Any]:
        from vllm.distributed.parallel_state import get_world_group
        from vllm.model_executor.warmup.flashinfer_autotune_cache import (
            flashinfer_autotune_cache_hash,
        )

        return {
            "config_hash": flashinfer_autotune_cache_hash(runner),
            "dp_rank": runner.vllm_config.parallel_config.data_parallel_rank,
            "rank": get_world_group().rank_in_group,
        }

    def resolve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        from vllm.model_executor.warmup.flashinfer_autotune_cache import (
            flashinfer_autotune_filename,
            flashinfer_autotune_root,
        )

        config_hash = request.get("config_hash")
        if not isinstance(config_hash, str) or not _CONFIG_HASH_RE.fullmatch(
            config_hash
        ):
            raise ValueError("invalid config_hash")

        # Garbage ranks simply name a file that does not exist (a miss); the
        # hex check above is what keeps the request inside the cache root.
        filename = flashinfer_autotune_filename(request["dp_rank"], request["rank"])
        path = flashinfer_autotune_root() / config_hash / filename
        if not path.is_file():
            return None
        return {filename: json.loads(path.read_bytes())}

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError("expected exactly one autotune table")
        ((filename, table),) = payload.items()
        if not isinstance(table, dict):
            raise ValueError("autotune table must be a JSON object")
        if not filename.startswith("autotune_configs_"):
            raise ValueError(f"unexpected file {filename!r}")
        # Reject tables flashinfer would refuse to load: a hard _metadata
        # conflict would also never be overwritten, poisoning this cache
        # path into re-tuning on every start.
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
        from vllm.model_executor.warmup.flashinfer_autotune_cache import (
            resolve_rank_flashinfer_autotune_file,
            write_flashinfer_autotune_cache,
        )

        (data,) = payload.values()
        cache_path = resolve_rank_flashinfer_autotune_file(runner)
        write_flashinfer_autotune_cache(cache_path, json.dumps(data).encode())
        logger.info("Installed remote FlashInfer autotune table: %s", cache_path)
