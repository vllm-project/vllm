# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fake weight loader for compile-only mode.

Initializes the model on the meta device (preserving Parameter subclasses
like ModelWeightParameter), and runs process_weights_after_loading on
meta tensors.

This is used for compile-only mode where we want to run torch.compile
without actually allocating any GPU memory for the model.
"""

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class FakeModelLoader(BaseModelLoader):
    """Model loader that initializes on meta device.

    Model initialization runs on ``meta`` device because FakeTensorMode
    doesn't preserve Parameter subclasses (e.g. ``ModelWeightParameter``
    becomes a plain ``FakeTensor``).  No GPU memory is allocated.
    """

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # No-op: all parameters are already on meta from init.
        pass

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            # Initialize model on meta device — no GPU memory, and
            # Parameter subclasses are preserved (unlike FakeTensorMode
            # which converts them to FakeTensor).
            with torch.device("meta"):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            # Run weight post-processing on meta tensors.
            from vllm.model_executor.model_loader.base_loader import (
                _has_online_quant,
            )
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_processing,
            )

            if _has_online_quant(model):
                finalize_layerwise_processing(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

            # Set fake_device on all parameters and buffers so
            # swap_meta_params_to_fake knows the intended target device.
            # Must happen AFTER process_weights_after_loading which may
            # create new parameters.
            for param in model.parameters():
                param.fake_device = target_device
            for buf in model.buffers():
                buf.fake_device = target_device

        return model.eval()


def swap_meta_params_to_fake(model: nn.Module) -> None:
    """Replace all meta-device parameters and buffers with FakeTensors.

    Called before torch.compile so that Dynamo sees cuda-device tensors
    during tracing.  Parameter subclasses are replaced with plain
    FakeTensors at this point — isinstance checks are no longer needed
    after weight processing is complete.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode()
    with fake_mode:
        for name, param in list(model.named_parameters()):
            device = getattr(param, "fake_device", None)
            assert device is not None and device != torch.device("meta"), (
                f"Parameter {name} missing fake_device or has meta device"
            )
            fake_data = torch.empty(
                param.shape,
                dtype=param.dtype,
                device=device,
            )
            # Navigate to the owning module and replace the parameter.
            *path, attr = name.split(".")
            parent = model.get_submodule(".".join(path)) if path else model
            parent.register_parameter(
                attr,
                nn.Parameter(fake_data, requires_grad=param.requires_grad),
            )
        for name, buf in list(model.named_buffers()):
            device = getattr(buf, "fake_device", buf.device)
            if device == torch.device("meta"):
                device = torch.device("cuda")
            fake_buf = torch.empty(
                buf.shape,
                dtype=buf.dtype,
                device=device,
            )
            *path, attr = name.split(".")
            parent = model.get_submodule(".".join(path)) if path else model
            parent.register_buffer(attr, fake_buf)
