# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import collections
import os

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype


class ServerlessLLMLoader(BaseModelLoader):
    # DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )
        # self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result = {}
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
                    if k2 > k:
                        # Same tensors, keep the one with the longer key.
                        break
                else:
                    result[k] = t
        return result

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        from sllm_store.torch import load_dict

        from vllm.distributed import get_tensor_model_parallel_rank

        assert os.path.isdir(model_config.model)

        rank = get_tensor_model_parallel_rank()

        local_model_path = model_config.model
        local_model_path = os.path.join(local_model_path, f"rank_{rank}")

        def remove_prefix(path, prefix):
            # Normalize the paths to ensure consistency across different platforms
            path = os.path.normpath(path)
            prefix = os.path.normpath(prefix)

            # Check if the path starts with the prefix
            if path.startswith(prefix):
                # Return the path without the prefix
                return path[len(prefix) :].lstrip(os.sep)

            # Return the original path if the prefix doesn't exist
            return path

        # vLLM needs a local model path to read model config but
        # ServerlessLLM Store requires a global model path as the model ID
        storage_path = os.getenv("SLLM_STORAGE_PATH")
        if storage_path is None:
            raise ValueError(
                "Please set the SLLM_STORAGE_PATH environment variable. "
                "This path should point to the root of the ServerlessLLM storage."
            )
        model_path = remove_prefix(local_model_path, storage_path)

        device_id = torch.cuda.current_device()
        target_device = torch.device(f"cuda:{device_id}")

        with set_default_torch_dtype(model_config.dtype):
            # Initialize model on target CUDA device
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            # Load pre-sharded weights from ServerlessLLM Store
            device_map = {"": device_id}
            sllm_state_dict = load_dict(model_path, device_map)

            # Filter out subtensors (tensors sharing memory) before validation
            state_dict = self._filter_subtensors(model.state_dict())
            unloaded_keys = set(state_dict.keys())

            # Directly assign pre-sharded weights to parameters
            param_dict = dict(model.named_parameters())
            for name, loaded_tensor in sllm_state_dict.items():
                if name in param_dict:
                    param = param_dict[name]
                    # Weights are already sharded, directly assign
                    param.data.copy_(loaded_tensor)
                    unloaded_keys.discard(name)

            # Load other weights (e.g., buffers)
            buff_dict = dict(model.named_buffers())
            for name, loaded_tensor in sllm_state_dict.items():
                if name in buff_dict:
                    buff = buff_dict[name]
                    buff.data.copy_(loaded_tensor)
                    unloaded_keys.discard(name)

            for name, buffer in model.named_buffers(recurse=True):
                if buffer.device.type != "cuda":
                    buffer.data = buffer.data.to(f"cuda:{device_id}")

            # Validate that all parameters were loaded
            if unloaded_keys:
                raise ValueError(
                    f"Missing keys {tuple(sorted(unloaded_keys))} in loaded state!"
                )

            # Process weights after loading (initializes attention buffers with correct dtype) #noqa: E501
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model, model_config):
        raise NotImplementedError(
            "ServerlessLLMLoader does not support in-place weight reloading."
        )

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from sllm_store.torch import save_dict

        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        state_dict = ServerlessLLMLoader._filter_subtensors(model.state_dict())

        # move all tensors to CPU
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.cpu().contiguous()

        save_path = os.path.join(path, f"rank_{rank}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dict(state_dict, save_path)
