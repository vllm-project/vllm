from vllm.core.scheduler.vllm_scheduler import VLLMScheduler
from vllm.core.scheduler.sarathi_scheduler import SarathiScheduler
from vllm.core.scheduler.dsarathi_scheduler import DSarathiScheduler
from vllm.core.scheduler.scheduler_type import SchedulerType
from vllm.utils.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


SchedulerRegistry.register(SchedulerType.VLLM, VLLMScheduler)
SchedulerRegistry.register(SchedulerType.SARATHI, SarathiScheduler)
SchedulerRegistry.register(SchedulerType.DSARATHI, DSarathiScheduler)