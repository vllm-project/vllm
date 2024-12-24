from typing import Optional

from vllm.config import VllmConfig
from vllm.platforms import Platform, PlatformEnum


class DummyPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_name = "dummy"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_device_name(cls) -> str:
        return "dummy"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = \
            "vllm_add_dummy_platform.my_worker.DummyWorker"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False
