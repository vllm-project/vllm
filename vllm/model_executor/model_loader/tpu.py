# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs

from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.tpu_distributed_utils import get_fqn, shard_model
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class TPUModelLoader(DefaultModelLoader):
    """
    A TPU model loader for model loading under SPMD mode.
    """

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        mesh: xs.Mesh | None = None,
    ) -> nn.Module:
        # Initialize model and load weights on CPU. Then, during SPMD partition,
        # weights are sharded and transferred to TPUs.
        self.counter_before_loading_weights = time.perf_counter()
        model_config = vllm_config.model_config
        assert model_config.quantization is None, "Quantization not supported"
        target_device = torch.device("cpu")
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config)

            load_format = vllm_config.load_config.load_format
            if load_format != "dummy":
                weights_to_load = {name for name, _ in model.named_parameters()}
                all_weights = self.get_all_weights(model_config, model)
                loaded_weights = model.load_weights(all_weights)
                self.counter_after_loading_weights = time.perf_counter()
                logger.info(
                    "Loading weights took %.2f seconds",
                    self.counter_after_loading_weights
                    - self.counter_before_loading_weights,
                )
                # We only enable strict check for non-quantized models
                # that have loaded weights tracking currently.
                if model_config.quantization is None and loaded_weights is not None:
                    weights_not_loaded = weights_to_load - loaded_weights
                    if weights_not_loaded:
                        raise ValueError(
                            "Following weights were not initialized from "
                            f"checkpoint: {weights_not_loaded}"
                        )
            else:
                logger.info("Use dummy weight during weight loading.")

            process_weights_after_loading(model, model_config, target_device)

        counter_before_partition = time.perf_counter()
        model = model.eval()
        model = model.to("xla")
        shard_model(model, mesh)
        counter_after_partition = time.perf_counter()
        logger.info(
            "Partition model took %.2f seconds",
            counter_after_partition - counter_before_partition,
        )

        # Ensure the model is properly loaded.
        self._check_model_is_loaded(mesh, model)

        # Need to torch compile after model sharding are done. Because the
        # compiler hints ('xs.mark_sharding') are torch ops.
        if not model_config.is_multimodal_model:
            model.model = torch.compile(model.model, backend="openxla")
        else:
            model.language_model.model = torch.compile(
                model.language_model.model, backend="openxla"
            )
        return model

    def _check_model_is_loaded(self, mesh: xs.Mesh | None, model: nn.Module) -> None:
        """
        Ensure the model is properly loaded.
        1. All model parameters and buffers are on XLA device.
        2. Non-SPMD friendly layers are replaced as expected.
        """
        device = xm.xla_device()
        device_type = str(device.type)

        # Check parameters
        for name, param in model.named_parameters():
            assert param.device.type == device_type, (
                f"Parameter {name} is on {param.device.type} instead of {device_type}"
            )

        # Check buffers
        for name, buffer in model.named_buffers():
            assert buffer.device.type == device_type, (
                f"Buffer {name} is on {buffer.device.type} instead of {device_type}"
            )

        for module in model.modules():
            if (mesh is not None) and (get_fqn(module) == "QKVParallelLinear"):
                raise AssertionError(
                    "QKVParallelLinear should be replaced by \
                            XlaQKVParallelLinear under SPMD mode."
                )
