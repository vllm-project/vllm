# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Generator

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.connector import (ConnectorType, create_remote_connector,
                            get_connector_type)
from vllm.connector.utils import parse_model_name
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.sharded_state_loader import (
    ShardedStateLoader)
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import (
    set_runai_streamer_env)

logger = init_logger(__name__)


class RemoteModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote database."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        set_runai_streamer_env(load_config)

    def _get_weights_iterator_kv(
        self,
        client,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        assert get_connector_type(client) == ConnectorType.KV
        rank = get_tensor_model_parallel_rank()
        return client.weight_iterator(rank)

    def _get_weights_iterator_fs(
        self,
        client,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        assert get_connector_type(client) == ConnectorType.FS
        return client.weight_iterator()

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        model_path: str,
        url: str,
    ) -> None:
        with create_remote_connector(url) as client:
            assert get_connector_type(client) == ConnectorType.KV
            model_name = parse_model_name(url)
            rank = get_tensor_model_parallel_rank()
            state_dict = ShardedStateLoader._filter_subtensors(
                model.state_dict())
            for key, tensor in state_dict.items():
                r_key = f"{model_name}/keys/rank_{rank}/{key}"
                client.set(r_key, tensor)

            for root, _, files in os.walk(model_path):
                for file_name in files:
                    # ignore hidden files
                    if file_name.startswith("."):
                        continue
                    if os.path.splitext(file_name)[1] not in (
                            ".bin",
                            ".pt",
                            ".safetensors",
                            ".jpg",  # ignore jpg file
                    ):
                        file_path = os.path.join(root, file_name)
                        with open(file_path, encoding='utf-8') as file:
                            file_content = file.read()
                            f_key = f"{model_name}/files/{file_name}"
                            client.setstr(f_key, file_content)

    def _load_model_from_remote_kv(self, model: nn.Module, client,
                                   vllm_config: VllmConfig):
        model_config = vllm_config.model_config
        device_config = vllm_config.device_config
        process_weights_after_loading(model, model_config,
                                      device_config.device)
        weights_iterator = self._get_weights_iterator_kv(client)
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        for key, tensor in weights_iterator:
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
                    "loading tensor of shape %s into "
                    "parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            param_data.copy_(tensor)
            state_dict.pop(key)
        if state_dict:
            raise ValueError(
                f"Missing keys {tuple(state_dict)} in loaded state!")

    def _load_model_from_remote_fs(self, model, client,
                                   vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            model.load_weights(self._get_weights_iterator_fs(client))
            process_weights_after_loading(model, model_config, target_device)

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        # RemoteModelLoader totally override load_model() due to
        # process_weights_after_loading() must be called before
        # load_weights() when connector_type is ConnectorType.KV.
        # So we just pass load_weights().
        pass

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        logger.info("Loading weights from remote storage ...")
        start = time.perf_counter()
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config

        assert load_config.load_format == "remote", (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}")

        print("model_config:", model_config)
        model_weights = model_config.model
        if hasattr(model_config, "model_weights"):
            model_weights = model_config.model_weights

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = initialize_model(vllm_config=vllm_config)

            with create_remote_connector(
                    model_weights,
                    device=device_config.device,
                    rank=get_tensor_model_parallel_rank()) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.KV:
                    self._load_model_from_remote_kv(model, client, vllm_config)
                elif connector_type == ConnectorType.FS:
                    self._load_model_from_remote_fs(model, client, vllm_config)

        end = time.perf_counter()
        logger.info("Loaded weights from remote storage in %.2f seconds.",
                    end - start)
        return model.eval()
