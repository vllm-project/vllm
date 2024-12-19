from vllm.config import VllmConfig
from vllm.platforms import Platform


class DummyPlatform(Platform):
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
