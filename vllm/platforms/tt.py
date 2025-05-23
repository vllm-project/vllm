from typing import TYPE_CHECKING, Optional

import torch

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


class TTPlatform(Platform):
    _enum = PlatformEnum.TT
    device_name: str = "tt"
    device_type: str = "tt"
    
    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True
    
    @classmethod
    def inference_mode(cls):
        return torch.no_grad()
    
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        assert not vllm_config.scheduler_config.chunked_prefill_enabled, (
            "Chunked prefill is not yet supported for TT backend")
        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TT backend")
        assert vllm_config.parallel_config.tensor_parallel_size == vllm_config.parallel_config.pipeline_parallel_size == 1, (
             "TT backend does not support distributed execution")
        assert not vllm_config.lora_config, "LoRA is not supported for TT backend"

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.worker.tt_worker.TTWorker"
