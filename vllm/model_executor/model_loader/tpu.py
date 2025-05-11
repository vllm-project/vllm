# SPDX-License-Identifier: Apache-2.0
import functools
import time

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed.tpu_distributed_utils import shard_model
from vllm.logger import init_logger
from vllm.model_executor.model_loader.loader import (
    DefaultModelLoader, _initialize_model, _process_weights_after_loading)
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


def _create_param_from_meta(meta_param):
    # Create a new parameter with the same shape and dtype as the meta parameter
    return nn.Parameter(torch.empty_like(meta_param, device='cpu'),
                        requires_grad=False)


def _get_fqn(module):
    # Get the fully qualified name of the module
    return module.__class__.__qualname__


def parameter_hook(module, name, param):
    # Ensure model parameters are initialized on the meta device.
    meta_param = torch.nn.Parameter(param.data.to('meta'),
                                    requires_grad=param.requires_grad)
    return meta_param


# Cannot move to meta device because attrs on the original tensor will be lost
# torch.nn.modules.module.register_module_parameter_registration_hook(parameter_hook)


def pre_weight_load_hook(module, name, mesh):
    print("Pre loader hook, module:", module)
    print("Pre loader hook, name:", name)
    meta_param = getattr(module, name)
    setattr(module, name, _create_param_from_meta(meta_param))
    print("check device", getattr(module, name).device)
    return


def post_weight_load_hook(module, name, mesh):
    module_fqn = _get_fqn(module)
    print("check fqn", module_fqn)
    print("Pre loader hook, module:", module)
    print("Post loader hook, name:", name)
    cpu_param = getattr(module, name)
    print(f"check loaded weight {cpu_param}")
    if module_fqn in ["RowParallelLinear", "ColumnParallelLinear"]:
        setattr(module, name,
                nn.Parameter(cpu_param.data.to('xla'), requires_grad=False))


def loader_wrapper(load_weight_func, module, param_name, mesh, *args,
                   **kwargs):
    print("in wrapper, check len(args)", len(args))
    print("load weight for module", module)
    param = args[0]
    weight = args[1]
    print(f"check param {param}")
    print(f"check weight {weight}")
    # pre_weight_load_hook(module, param_name, mesh)
    print(f"before load weight {load_weight_func}")
    load_weight_func(*args, **kwargs)
    print(f"after load weight {load_weight_func}")
    print(f"check param {param}")
    print(f"check param from model {getattr(module, param_name)}")
    post_weight_load_hook(module, param_name, mesh)


def wrap_loader(model, mesh):
    print("wrapping model", model)
    for name, param in model.named_parameters(recurse=False):
        if hasattr(param, "weight_loader"):
            loader = param.weight_loader
        else:
            loader = default_weight_loader
        wrapped_method = functools.partial(loader_wrapper, loader, model, name,
                                           mesh)
        param.weight_loader = wrapped_method


def check_all_buffers_parameters_on_device(module: nn.Module,
                                           device: torch.device) -> None:
    """Check if all buffers and parameters are on the device."""
    for name, buffer in module.named_buffers():
        if buffer.device != device:
            print(
                f"Buffer {name} is on device {buffer.device}, expected to be on {device}"
            )
            raise ValueError(f"Buffer {name} is not on device {buffer.device}")
    for name, param in module.named_parameters(recurse=False):
        if param.device != device:
            print(f"Parameter {name} is not on device {param.device}")
            raise ValueError(
                f"Parameter {name} is not on device {param.device}")
    print("All buffers and parameters are on device 'xla'")


class TPUModelLoader(DefaultModelLoader):

    def load_model(self, vllm_config: VllmConfig, mesh) -> nn.Module:
        self.counter_before_loading_weights = time.perf_counter()
        # device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        assert model_config.quantization is None, "Quantization not supported"
        target_device = torch.device('cpu')
        # target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            # with torch.device('meta'):
            with target_device:
                model = _initialize_model(vllm_config=vllm_config)
            # print("check state dict", model.state_dict())

            # buffer_dict = {}
            # for name, buffers in model.named_buffers():
            #     print(f"Buffer {name} is on device {buffers.device}")
            #     buffer_dict[name] = buffers
            # print("check buffer dict", buffer_dict)
            # buffer_dict = pytree.tree_map_only(torch.Tensor,
            #                       lambda x: x.to('xla'),
            #                       buffer_dict)

            # for name, buffers in model.named_parameters():
            #     print(f"Parameter {name} is on device {buffers.device}")

            # model.apply(functools.partial(wrap_loader, mesh=mesh))
            # logger.info(f"check vllm config {vllm_config}")
            # logger.info("Meta Model initialized")
            weights_to_load = {name for name, _ in model.named_parameters()}
            all_weights = self.get_all_weights(model_config, model)
            loaded_weights = model.load_weights(all_weights)

            # for name, buffers in buffer_dict.items():
            #     setattr(model, name, buffers.to('xla'))
            # model.load_state_dict(buffer_dict, strict=False)
            # model = model.to('xla')
            self.counter_after_loading_weights = time.perf_counter()
            # print("check state dict", model.state_dict())
            # check_all_buffers_parameters_on_device(model, torch.device('xla'))
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights -
                self.counter_before_loading_weights)
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}")

            _process_weights_after_loading(model, model_config, target_device)

        counter_before_partition = time.perf_counter()
        model = model.eval()
        model = model.to('xla')
        shard_model(model, mesh)
        counter_after_partition = time.perf_counter()
        logger.info("Partition model took %.2f seconds",
                    counter_after_partition - counter_before_partition)
        # Need to torch compile after model sharding are done. Because the
        # compiler hints are torch ops.
        if not model_config.is_multimodal_model:
            model.model = torch.compile(model.model, backend="openxla")
        else:
            model.language_model.model = \
                torch.compile(model.language_model.model, backend="openxla")
        return model
