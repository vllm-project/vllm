# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.cpu_attn import TorchSDPAMetadataBuilderV1
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert self.speculative_config is None, "spec decode is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tenosrs()

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if len(self.kv_cache_config.kv_cache_groups) > 1:
            raise ValueError("Multiple KVCacheGroups is not"
                             "currently supported with CPU model runner.")

        assert type(self.attn_groups[0]
                    [0].metadata_builder) is TorchSDPAMetadataBuilderV1

        self.attn_groups[0][0].metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)

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

        for block_table in self.input_batch.block_table.block_tables:
            for k, v in vars(block_table).items():
                if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                    replace_tensor(block_table, k, k[:-4])

    def load_model(self, eep_scale_up: bool = False) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.model_config,
                                              self.scheduler_config,
                                              self.lora_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

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
