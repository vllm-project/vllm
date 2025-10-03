# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import glob
import os
import time
from collections.abc import Generator
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader, DownloadType
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.s3_utils import glob as s3_glob
from vllm.transformers_utils.utils import is_s3

logger = init_logger(__name__)


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/offline_inference/save_sharded_state.py` for creating a sharded
    checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: str | None):
        if is_s3(model_name_or_path) or os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        local_model_path = model_weights

        rank = get_tensor_model_parallel_rank()
        pattern = os.path.join(
            local_model_path,
            self.pattern.format(rank=rank, part="*"),
        )

        filepaths = []
        if is_s3(local_model_path):
            file_pattern = f"*{self.pattern.format(rank=rank, part='*')}"
            filepaths = s3_glob(path=local_model_path, allow_pattern=[file_pattern])
        else:
            filepaths = glob.glob(pattern)
        if not filepaths:
            # TODO: support un-sharded checkpoints too
            raise ValueError(
                f"Could not find checkpoint files '{pattern}', only "
                f"pre-sharded checkpoints are currently supported!"
            )
        state_dict = self._filter_subtensors(model.state_dict())
        counter_before_loading_weights = time.perf_counter()
        for key, tensor in self.iterate_over_files(filepaths):
            # If loading with LoRA enabled, additional padding may
            # be added to certain parameters. We only load into a
            # narrowed view of the parameter data.
            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_shape:
                logger.warning(
                    "loading tensor of shape %s into parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            param_data.copy_(tensor)
            state_dict.pop(key)
        counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            counter_after_loading_weights - counter_before_loading_weights,
            scope="local",
        )
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

    def iterate_over_files(
        self, paths
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if self.load_config.load_format == "runai_streamer_sharded":
            yield from runai_safetensors_weights_iterator(paths, True)
        else:
            from safetensors.torch import safe_open

            for path in paths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        yield key, tensor

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from safetensors.torch import save_file

        from vllm.distributed import get_tensor_model_parallel_rank

        if pattern is None:
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        total_size = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            state_dict_part[key] = tensor
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )

    def get_download_type(self, model_name_or_path: str) -> DownloadType:
        if os.path.isdir(model_name_or_path):
            return DownloadType.LOCAL_FILE
        if is_s3(model_name_or_path):
            return DownloadType.S3
        return DownloadType.HUGGINGFACE_HUB
