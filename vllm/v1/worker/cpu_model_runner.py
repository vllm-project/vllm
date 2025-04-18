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
        assert not self.use_spec_decode, "spec decode is not supported."
        assert not self.model_config.uses_mrope, "mrope is not supported."
        assert self.lora_config is None, "lora is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tenosrs()

        self.seq_start_loc_cpu = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device="cpu",
        )
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

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

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings():
            self._dummy_run(self.max_num_tokens)
        logger.info("Warming up done.")

    def _init_device_properties(self) -> None:
        pass


@contextmanager
def _set_global_compilation_settings():
    import torch._inductor.config

    # Note: The CPPGEMM backend requires freezing parameters.
    freezing_value = torch._inductor.config.freezing
    torch._inductor.config.freezing = True
    # Note: workaround for "ValueError: fast mode: can't pickle cyclic objects
    # including object type dict"
    force_disable_caches = torch._inductor.config.force_disable_caches
    torch._inductor.config.force_disable_caches = True
    yield
    torch._inductor.config.freezing = freezing_value
    torch._inductor.config.force_disable_caches = force_disable_caches
