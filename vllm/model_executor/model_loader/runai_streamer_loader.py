# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from collections.abc import Callable, Generator
from pathlib import Path

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.repo_utils import resolve_revision
from vllm.transformers_utils.runai_utils import is_runai_obj_uri, list_safetensors

logger = init_logger(__name__)


def _select_indexed_weight_files(
    files: list[str],
    metadata_dir: str,
    is_unused_weight: Callable[[str], bool],
) -> list[str]:
    # Experimental opt-in: the pinned index has been checked against shard headers.
    try:
        weight_map = json.loads(
            (Path(metadata_dir) / SAFE_WEIGHTS_INDEX_NAME).read_text()
        )["weight_map"]
    except (OSError, ValueError, KeyError, TypeError):
        return files
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        return files
    wanted = {shard for name, shard in weight_map.items() if not is_unused_weight(name)}
    if not wanted or not wanted.issubset({Path(p).name for p in files}):
        return files
    selected = [p for p in files if Path(p).name in wanted]
    logger.info("RunAI model-owned selection: %d/%d shards", len(selected), len(files))
    return selected


class RunaiModelStreamerLoader(BaseModelLoader):
    """Model loader that can load safetensors
    files from local FS, S3, GCS, or Azure Blob Storage.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        self._is_distributed: bool = False
        if load_config.model_loader_extra_config:
            extra_config = load_config.model_loader_extra_config

            allowed_keys = {"distributed", "concurrency", "memory_limit"}
            if unexpected_keys := set(extra_config) - allowed_keys:
                raise ValueError(
                    "Unexpected extra config keys for runai_streamer: "
                    f"{unexpected_keys}"
                )

            if "distributed" in extra_config:
                distributed = extra_config["distributed"]
                if not isinstance(distributed, bool):
                    raise ValueError(f"distributed must be a bool, got {distributed!r}")
                self._is_distributed = distributed

            # Validate every value before mutating os.environ, so a later
            # invalid key cannot leave an earlier one partially applied.
            env_updates: dict[str, str] = {}
            if "concurrency" in extra_config:
                concurrency = extra_config["concurrency"]
                if (
                    isinstance(concurrency, bool)
                    or not isinstance(concurrency, int)
                    or concurrency <= 0
                ):
                    raise ValueError(
                        f"concurrency must be a positive integer, got {concurrency!r}"
                    )
                env_updates["RUNAI_STREAMER_CONCURRENCY"] = str(concurrency)

            if "memory_limit" in extra_config:
                memory_limit = extra_config["memory_limit"]
                if (
                    isinstance(memory_limit, bool)
                    or not isinstance(memory_limit, int)
                    or memory_limit < -1
                ):
                    raise ValueError(
                        f"memory_limit must be an integer >= -1, got {memory_limit!r}"
                    )
                env_updates["RUNAI_STREAMER_MEMORY_LIMIT"] = str(memory_limit)
            os.environ.update(env_updates)

            runai_streamer_s3_endpoint = os.getenv("RUNAI_STREAMER_S3_ENDPOINT")
            aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
            if runai_streamer_s3_endpoint is None and aws_endpoint_url is not None:
                os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = aws_endpoint_url

    def _prepare_weights(
        self, model_name_or_path: str, revision: str | None
    ) -> list[str]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        is_object_storage_path = is_runai_obj_uri(model_name_or_path)
        is_local = os.path.isdir(model_name_or_path)
        safetensors_pattern = "*.safetensors"
        index_file = SAFE_WEIGHTS_INDEX_NAME

        if not is_local and not is_object_storage_path:
            # `model_weights` can point to another repo than the one `revision` was
            # resolved for, which does not pin this one.
            revision = resolve_revision(model_name_or_path, revision)

        hf_folder = (
            model_name_or_path
            if (is_local or is_object_storage_path)
            else download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                [safetensors_pattern],
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        )
        hf_weights_files = list_safetensors(path=hf_folder)

        if not is_local and not is_object_storage_path:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                cache_dir=self.load_config.download_dir,
                revision=revision,
            )

        if not hf_weights_files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights with `{model_name_or_path}`"
            )

        return hf_weights_files

    def _get_weights_iterator(
        self,
        model_or_path: str,
        revision: str | None,
        is_unused_weight: Callable[[str], bool] | None = None,
        metadata_dir: str | None = None,
        cpu_weights_group=None,
        is_shared_cpu_weight=None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_weights_files = self._prepare_weights(model_or_path, revision)
        if is_unused_weight is not None and metadata_dir is not None:
            hf_weights_files = _select_indexed_weight_files(
                hf_weights_files, metadata_dir, is_unused_weight
            )
        return runai_safetensors_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
            self._is_distributed,
            cpu_weights_group=cpu_weights_group,
            is_shared_cpu_weight=is_shared_cpu_weight,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model if necessary."""
        self._prepare_weights(model_config.model, model_config.revision)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        if (
            os.environ.get("VLLM_RUNAI_SINGLE_ENGRAM_READER") == "1"
            and self._is_distributed
        ):
            from vllm.models.deepseek_v41.common.engram import DPSharedEngramStorage

            params = dict(model.named_parameters())
            weight_loader_owners = {
                name: getattr(getattr(param, "weight_loader", None), "__self__", None)
                for name, param in params.items()
            }
            owners = {
                name: owner
                for name, owner in weight_loader_owners.items()
                if isinstance(owner, DPSharedEngramStorage)
            }
            if not owners:
                yield from self._get_weights_iterator(
                    model_weights,
                    model_config.revision,
                    is_unused_weight=getattr(
                        model, "is_unused_checkpoint_weight", None
                    ),
                    metadata_dir=model_config.model,
                )
                return
            group = next(iter(owners.values())).group
            assert all(owner.group is group for owner in owners.values())

            def is_shared(name):
                mapped = model.hf_to_vllm_mapper.apply_list([name])
                return len(mapped) == 1 and mapped[0] in owners

            yield from self._get_weights_iterator(
                model_weights,
                model_config.revision,
                cpu_weights_group=group,
                is_shared_cpu_weight=is_shared,
            )
            return
        is_unused_weight = getattr(model, "is_unused_checkpoint_weight", None)
        if os.environ.get("VLLM_RUNAI_USE_LOCAL_WEIGHT_INDEX") == "1" and callable(
            is_unused_weight
        ):
            yield from self._get_weights_iterator(
                model_weights,
                model_config.revision,
                is_unused_weight=is_unused_weight,
                metadata_dir=model_config.model,
            )
            return
        yield from self._get_weights_iterator(model_weights, model_config.revision)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model."""
        model.load_weights(self.get_all_weights(model_config, model))
