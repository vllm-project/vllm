from torch import nn

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_transfer_utils import (
    maybe_register_PD_disagg_hooks)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


def get_model(*, vllm_config: VllmConfig) -> nn.Module:

    loader = get_model_loader(vllm_config.load_config)

    model = loader.load_model(vllm_config=vllm_config)

    maybe_register_PD_disagg_hooks(model, vllm_config)

    return model


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
