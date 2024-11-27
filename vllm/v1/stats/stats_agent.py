import threading
import time
from typing import List


from vllm.logger import init_logger
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.stats.common import EngineStatsUpdate, ms_to_s

logger = init_logger(__name__)


class EngineStatsAgent:
    """
    EngineStatsAgent accumulates stats from the EngineCore and materialize a
    EngineStatsUpdate to be aggregated by the EngineStatsManager.

    - In the multi-process architecture (e.g. AsyncMPClient or SyncMPClient),
      this is running in the background engine core process.
    - In the single-process architecture, this is running in the same process
      as the LLMEngine.

    TODO(rickyx): We could further optimize the stats update by isolating the
    stats polling from the IO thread.
    """

    def __init__(self):
        self._stats_update = self._new_update()

    def _new_update(self) -> EngineStatsUpdate:
        return EngineStatsUpdate(updated_ts_ms=ms_to_s(time.time()))

    def get_update(self) -> EngineStatsUpdate:
        return self._stats_update

    def get_and_reset_update(
        self,
    ) -> EngineStatsUpdate:
        logger.info(f"get_and_reset_update")
        cur_update = self._stats_update
        new_update = self._new_update()
        self._stats_update = new_update
        return cur_update


class ThreadSafeEngineStatsAgent(EngineStatsAgent):
    """
    A thread-safe version of EngineStatsAgent.
    """

    def __init__(self, *args, **kwargs):
        self._lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def on_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ):
        with self._lock:
            self.on_schedule(scheduler_output)

    def on_execute_model(
        self,
        model_runner_output: ModelRunnerOutput,
        engine_outputs: List[EngineCoreOutput],
    ):
        with self._lock:
            self.on_execute_model(model_runner_output, engine_outputs)

    def finalize_update(self) -> EngineStatsUpdate:
        with self._lock:
            return self.finalize_update()
