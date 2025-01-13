from typing import TYPE_CHECKING, Optional

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU
    device_name: str = "hpu"
    device_type: str = "hpu"
    dispatch_key: str = "HPU"
    supported_quantization: list[str] = ["inc"]

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        return _Backend.HPU_ATTN

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:

        scheduler_config = vllm_config.scheduler_config

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                parallel_config.worker_cls = \
                    "vllm.worker.multi_step_hpu_worker.MultiStepHPUWorker"
            elif vllm_config.speculative_config:
                parallel_config.worker_cls = \
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = \
                    "vllm.worker.hpu_worker.HPUWorker"
            else:
                parallel_config.worker_cls = "vllm.worker.hpu_worker.HPUWorker"
