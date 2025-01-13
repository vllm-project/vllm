from typing import Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.executor.uniproc_executor import UniprocExecutor
from vllm.v1.worker.xpu_worker import XPUWorker

logger = init_logger(__name__)


class XPUUniprocExecutor(UniprocExecutor):

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)

    def _create_worker(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> XPUWorker:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return XPUWorker(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )
