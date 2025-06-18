# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert self.speculative_config is None, "spec decode is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tenosrs()

    def _postprocess_tenosrs(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str,
                           device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                setattr(obj, device_attr_name, cpu_tensor)

        for k, v in vars(self).items():
            if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                replace_tensor(self, k, k[:-4])

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for k, v in vars(self.input_batch.block_table).items():
            if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch.block_table, k, k[:-4])

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.model_config,
                                              self.scheduler_config,
                                              self.lora_config, self.device)

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(max(16, self.max_num_reqs))
        logger.info("Warming up done.")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    import torch._inductor.config

    inductor_config = config.compilation_config.inductor_compile_config
    try:
        # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
        freezing_value = torch._inductor.config.freezing
        if inductor_config.get("max_autotune", False):
            torch._inductor.config.freezing = True
        yield
    finally:
        torch._inductor.config.freezing = freezing_value
