import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.utils import DeviceMemoryProfiler, is_pin_memory_available
from vllm.wde.core.config import DeviceConfig, LoadConfig, ModelConfig
from vllm.wde.core.layers.attention import AttentionBackend
from vllm.wde.prefill_only.config import PrefillOnlySchedulerConfig
from vllm.wde.prefill_only.schema.execute_io import ModelInputForGPU

logger = init_logger(__name__)


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: PrefillOnlySchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        attn_backend: AttentionBackend,
    ):
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.load_config = load_config
        self.attn_backend = attn_backend
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Lazy initialization
        self.model: nn.Module  # Set after load_model

    def load_model(self) -> None:
        from vllm.wde.core.loader.loader import (get_model_loader,
                                                 initialize_model)

        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            loader = get_model_loader(self.load_config)
            self.model = initialize_model(model_config=self.model_config,
                                          load_config=self.load_config,
                                          device_config=self.device_config,
                                          attn_backend=self.attn_backend)

            loader.load_model(self.model,
                              model_config=self.model_config,
                              device_config=self.device_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPU,
    ):
        return self.model(**model_input.to_dict())
